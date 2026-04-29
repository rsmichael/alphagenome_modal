"""AlphaGenome PyTorch inference API deployed on Modal."""

import base64
from typing import Optional

import modal
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Modal image — PyTorch only, no JAX required for inference
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "xz-utils", "liblzma-dev")
    .pip_install("uv")
    .uv_pip_install(
        "torch",
        "numpy",
        "fastapi",
        "pydantic",
        "safetensors",
        "zstandard",
        "dna-parser",
        "lance",
        "lancedb",
        "pyarrow",
    )
    .run_commands(
        "git clone https://github.com/genomicsxai/alphagenome-pytorch /root/alphagenome-pytorch",
        "pip install /root/alphagenome-pytorch",
    )
)

app = modal.App("alphagenome-inference-torch")

model_volume = modal.Volume.from_name("alphagenome-models")

WEIGHTS_PATH = "/models/model_pytorch.pth"
SEQ_LEN = 131_072
ORGANISM_INDEX = {"human": 0, "mouse": 1}

# All head names the model produces
VALID_HEADS = {
    "atac", "dnase", "procap", "cage", "rna_seq",
    "chip_tf", "chip_histone", "pair_activations",
    "splice_sites_classification", "splice_sites_usage", "splice_sites_junction",
}
# These heads only exist at 128bp resolution
HEADS_128BP_ONLY = {"chip_tf", "chip_histone"}
# These heads require 1bp resolution
SPLICE_HEADS = {"splice_sites_classification", "splice_sites_usage", "splice_sites_junction"}

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _encode(arr) -> dict:
    """Encode a numpy array as zstd-compressed base64 with shape/dtype metadata."""
    import zstandard as zstd
    import numpy as np
    arr = np.asarray(arr)
    compressed = zstd.ZstdCompressor(level=1).compress(arr.tobytes())
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "encoding": "zstd",
    }


def _tensor_to_encoded(t, fp16: bool = True) -> dict:
    import torch
    arr = t.detach().cpu()
    # bfloat16 has no numpy equivalent; cast to float16 (half the payload vs float32)
    if arr.dtype == torch.bfloat16:
        arr = arr.to(torch.float16)
    elif arr.dtype == torch.float32 and fp16:
        arr = arr.to(torch.float16)
    return _encode(arr.numpy())


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    sequence: str = Field(
        ...,
        description="DNA sequence (ACGTN). Padded with N to 131,072 bp if shorter.",
    )
    organism: str = Field(default="human", description="'human' or 'mouse'")
    heads: Optional[list[str]] = Field(
        default=None,
        description=(
            "Which prediction heads to include. None returns all heads available "
            "at the requested resolution."
        ),
    )
    resolution: Optional[int] = Field(
        default=128,
        description=(
            "Output resolution in bp: 1, 128, or null (both). "
            "Defaults to 128 — skips the expensive decoder."
        ),
    )
    return_embeddings: bool = Field(
        default=False,
        description="If true, also return trunk embeddings.",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        invalid = set(v.upper()) - set("ACGTN")
        if invalid:
            raise ValueError(f"Invalid characters in sequence: {invalid}")
        return v.upper()

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v):
        if v.lower() not in ORGANISM_INDEX:
            raise ValueError(f"organism must be one of {list(ORGANISM_INDEX)}")
        return v.lower()

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        if v is not None and v not in (1, 128):
            raise ValueError("resolution must be 1, 128, or null")
        return v

    @field_validator("heads")
    @classmethod
    def validate_heads(cls, v):
        if v is None:
            return v
        unknown = set(v) - VALID_HEADS
        if unknown:
            raise ValueError(
                f"Unknown heads: {unknown}. Valid options: {sorted(VALID_HEADS)}"
            )
        return v


class EmbedRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence (ACGTN).")
    organism: str = Field(default="human", description="'human' or 'mouse'")
    resolution: Optional[int] = Field(
        default=128,
        description="1, 128, or null (both). 128 skips the decoder.",
    )
    center_pos: Optional[int] = Field(
        default=None,
        description=(
            "Center position (0-based) within the original, unpadded sequence for the 1bp crop window. "
            "Sequence-relative: 0 = first base of the input sequence. "
            "Defaults to the middle of the sequence."
        ),
    )
    window_bp: Optional[int] = Field(
        default=None,
        description=(
            "Number of 1bp positions to return, centred on center_pos. "
            "Only applies when resolution=1. If omitted the full 131,072 positions "
            "are returned (large — expect ~768 MB)."
        ),
    )
    use_dna_parser: bool = Field(
        default=False,
        description=(
            "If true, use dna_parser.onehot_encoding_rust for one-hot encoding "
            "on CPU instead of the GPU ASCII lookup table."
        ),
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        invalid = set(v.upper()) - set("ACGTN")
        if invalid:
            raise ValueError(f"Invalid characters: {invalid}")
        return v.upper()

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v):
        if v.lower() not in ORGANISM_INDEX:
            raise ValueError(f"organism must be one of {list(ORGANISM_INDEX)}")
        return v.lower()

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        if v is not None and v not in (1, 128):
            raise ValueError("resolution must be 1, 128, or null")
        return v

    @field_validator("window_bp")
    @classmethod
    def validate_window_bp(cls, v):
        if v is not None and v < 1:
            raise ValueError("window_bp must be a positive integer")
        return v


class EmbedBatchRequest(BaseModel):
    sequences: list[str] = Field(
        ...,
        description="List of DNA sequences (ACGTN). Each padded to 131,072 bp if shorter.",
        min_length=1,
    )
    organism: str = Field(default="human", description="'human' or 'mouse' — shared across the batch.")
    resolution: Optional[int] = Field(
        default=128,
        description="1, 128, or null (both). 128 skips the decoder.",
    )
    center_pos: Optional[int] = Field(
        default=None,
        description="Center position (0-based) within the original, unpadded sequence for 1bp crop. Sequence-relative: 0 = first base of the input sequence. Defaults to sequence midpoint.",
    )
    window_bp: Optional[int] = Field(
        default=None,
        description="Number of 1bp positions to return around center_pos. Only applies when resolution=1.",
    )
    use_dna_parser: bool = Field(
        default=False,
        description=(
            "If true, use dna_parser.onehot_encoding_rust for one-hot encoding "
            "on CPU instead of the GPU ASCII lookup table."
        ),
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v):
        for i, seq in enumerate(v):
            invalid = set(seq.upper()) - set("ACGTN")
            if invalid:
                raise ValueError(f"sequences[{i}] contains invalid characters: {invalid}")
        return [seq.upper() for seq in v]

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v):
        if v.lower() not in ORGANISM_INDEX:
            raise ValueError(f"organism must be one of {list(ORGANISM_INDEX)}")
        return v.lower()

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        if v is not None and v not in (1, 128):
            raise ValueError("resolution must be 1, 128, or null")
        return v

    @field_validator("window_bp")
    @classmethod
    def validate_window_bp(cls, v):
        if v is not None and v < 1:
            raise ValueError("window_bp must be a positive integer")
        return v


# ---------------------------------------------------------------------------
# Stateless sequence-encoding helpers (importable by other Modal modules)
# ---------------------------------------------------------------------------


def _pad(sequence: str) -> str:
    """Center sequence in SEQ_LEN window, padding with N on both sides."""
    if len(sequence) > SEQ_LEN:
        raise ValueError(f"sequence length {len(sequence)} exceeds SEQ_LEN {SEQ_LEN}")
    pad = SEQ_LEN - len(sequence)
    left = pad // 2
    return "N" * left + sequence + "N" * (pad - left)


def _seq_to_gpu(sequences: list[str], onehot_lookup):
    """ASCII lookup path: transfers 1 byte/base to GPU, applies lookup table there."""
    import numpy as np
    import torch

    padded = [_pad(s) for s in sequences]
    raw = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(len(padded), SEQ_LEN).copy()
    byte_tensor = torch.from_numpy(raw).to("cuda", non_blocking=True)
    return onehot_lookup[byte_tensor.long()]  # (B, L, 4) float32


def _seq_to_gpu_dna_parser(sequences: list[str]):
    """dna_parser path: multi-threaded Rust encoding on CPU, int8 transferred to GPU."""
    import numpy as np
    import torch
    import dna_parser

    B = len(sequences)
    arrs = dna_parser.onehot_encoding_rust(sequences, "after", SEQ_LEN, 0)
    stacked = np.stack(arrs)  # (B, SEQ_LEN, 4) int8, columns C,G,A,T
    mat = np.empty((B, SEQ_LEN, 4), dtype=np.int8)
    mat[:, :, 0] = stacked[:, :, 2]  # A was col 2
    mat[:, :, 1] = stacked[:, :, 0]  # C was col 0
    mat[:, :, 2] = stacked[:, :, 1]  # G was col 1
    mat[:, :, 3] = stacked[:, :, 3]  # T was col 3
    return torch.from_numpy(mat).to("cuda", non_blocking=True).float()


def _crop_1bp(emb: dict, resolution, window_bp, center_pos) -> dict:
    """Slice embeddings_1bp to the requested window."""
    if resolution in (1, None) and "embeddings_1bp" in emb and window_bp is not None:
        seq_len = emb["embeddings_1bp"].shape[1]
        center = center_pos if center_pos is not None else seq_len // 2
        start = max(0, center - window_bp // 2)
        end = min(seq_len, start + window_bp)
        emb["embeddings_1bp"] = emb["embeddings_1bp"][:, start:end, :]
    return emb


def _resolutions_tuple(resolution: Optional[int]):
    if resolution == 1:
        return (1,)
    if resolution == 128:
        return (128,)
    return None


# ---------------------------------------------------------------------------
# Modal service class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume.read_only()},
    scaledown_window=300,
    max_containers=10,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
class AlphaGenomeService:
    """Serves the AlphaGenome PyTorch model as a FastAPI web endpoint."""

    @modal.enter()
    def load_model(self):
        import torch
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        print(f"Loading weights from {WEIGHTS_PATH} ...")
        self.model = AlphaGenome.from_pretrained(
            WEIGHTS_PATH,
            dtype_policy=DtypePolicy.mixed_precision(),
            device="cuda",
        )
        self.model.eval()
        print("Compiling model with torch.compile ...")
        self.model = torch.compile(self.model, mode="default", dynamic=True)
        # Build ASCII→one-hot lookup table on GPU.
        # Shape (128, 4): row i = one-hot for ASCII byte i; all-zeros for N/unknown.
        import torch
        lookup = torch.zeros(128, 4, dtype=torch.float32, device="cuda")
        for idx, ch in enumerate("ACGT"):
            lookup[ord(ch)] = torch.eye(4)[idx]
            lookup[ord(ch.lower())] = torch.eye(4)[idx]
        self._onehot_lookup = lookup  # (128, 4) on GPU
        print("Model ready.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _pad(self, sequence: str) -> str:
        if len(sequence) > SEQ_LEN:
            raise ValueError(f"sequence length {len(sequence)} exceeds SEQ_LEN {SEQ_LEN}")
        pad = SEQ_LEN - len(sequence)
        left = pad // 2
        return "N" * left + sequence + "N" * (pad - left)

    def _seq_to_gpu(self, sequences: list[str]):
        """Convert DNA strings to a GPU float32 one-hot tensor via ASCII lookup.

        Transfers raw ASCII bytes (1 byte/base) to GPU instead of float32
        one-hots (16 bytes/base), then applies a lookup table — 16x smaller
        host→device transfer.
        """
        import numpy as np
        import torch

        padded = [self._pad(s) for s in sequences]
        raw = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(len(padded), SEQ_LEN).copy()
        byte_tensor = torch.from_numpy(raw).to("cuda", non_blocking=True)  # (B, L) uint8
        return self._onehot_lookup[byte_tensor.long()]  # (B, L, 4) float32

    def _seq_to_gpu_dna_parser(self, sequences: list[str]):
        """Convert DNA strings to a GPU float32 one-hot tensor via dna_parser.

        Uses dna_parser.onehot_encoding_rust (multi-threaded Rust) for CPU
        encoding, then transfers to GPU.

        dna_parser returns columns in C,G,A,T order; we reindex to A,C,G,T
        to match the model's convention.
        """
        import numpy as np
        import torch
        import dna_parser

        # Returns list of (SEQ_LEN, 4) int8 arrays, columns in C,G,A,T order
        arrs = dna_parser.onehot_encoding_rust(sequences, "after", SEQ_LEN, 0)
        mat = np.stack(arrs)[:, :, [2, 0, 1, 3]]  # (B, SEQ_LEN, 4) reordered to A,C,G,T
        return torch.from_numpy(mat.astype(np.float32)).to("cuda", non_blocking=True)

    def _prepare_input(self, sequence: str, organism: str, use_dna_parser: bool = False):
        import torch

        encode = self._seq_to_gpu_dna_parser if use_dna_parser else self._seq_to_gpu
        dna = encode([sequence])  # (1, L, 4)
        org = torch.tensor([ORGANISM_INDEX[organism]], device="cuda")
        return dna, org

    def _prepare_input_batch(self, sequences: list[str], organism: str, use_dna_parser: bool = False):
        import torch

        encode = self._seq_to_gpu_dna_parser if use_dna_parser else self._seq_to_gpu
        dna = encode(sequences)  # (B, L, 4)
        org = torch.full((len(sequences),), ORGANISM_INDEX[organism], dtype=torch.long, device="cuda")
        return dna, org

    def _crop_1bp(self, emb: dict, resolution, window_bp, center_pos) -> dict:
        """Slice embeddings_1bp to the requested window in-place."""
        if resolution in (1, None) and "embeddings_1bp" in emb and window_bp is not None:
            seq_len = emb["embeddings_1bp"].shape[1]
            center = center_pos if center_pos is not None else seq_len // 2
            start = max(0, center - window_bp // 2)
            end = min(seq_len, start + window_bp)
            emb["embeddings_1bp"] = emb["embeddings_1bp"][:, start:end, :]
        return emb

    def _resolutions_tuple(self, resolution: Optional[int]):
        if resolution == 1:
            return (1,)
        if resolution == 128:
            return (128,)
        return None  # None → model computes both

    def _run_predict(self, req: PredictRequest) -> dict:
        import torch

        dna, org = self._prepare_input(req.sequence, req.organism)
        resolutions = self._resolutions_tuple(req.resolution)

        # Which heads the caller wants
        requested = set(req.heads) if req.heads else VALID_HEADS

        # Drop heads that are incompatible with the requested resolution
        if req.resolution == 128:
            requested -= SPLICE_HEADS
        if req.resolution == 1:
            requested -= HEADS_128BP_ONLY

        with torch.no_grad():
            outputs = self.model.predict(
                dna, org,
                resolutions=resolutions,
                return_embeddings=req.return_embeddings,
            )

        result: dict = {"heads": {}}

        for head_name, head_out in outputs.items():
            if head_name.startswith("embeddings_"):
                continue  # handled below
            if head_name not in requested:
                continue
            if isinstance(head_out, dict):
                # dict keyed by resolution int, e.g. {1: tensor, 128: tensor}
                result["heads"][head_name] = {
                    str(res): _tensor_to_encoded(t)
                    for res, t in head_out.items()
                }
            else:
                result["heads"][head_name] = _tensor_to_encoded(head_out)

        if req.return_embeddings:
            result["embeddings"] = {
                key: _tensor_to_encoded(outputs[key])
                for key in ("embeddings_1bp", "embeddings_128bp", "embeddings_pair")
                if key in outputs
            }

        return result

    def _run_embed(self, req: EmbedRequest) -> dict:
        import torch

        dna, org = self._prepare_input(req.sequence, req.organism, req.use_dna_parser)
        resolutions = self._resolutions_tuple(req.resolution)

        compute_dtype = self.model.dtype_policy.compute_dtype
        use_amp = compute_dtype != torch.float32

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
                emb = self.model.encode(dna, org, resolutions=resolutions)

        center_pos = req.center_pos
        if center_pos is not None:
            pad_left = (SEQ_LEN - len(req.sequence)) // 2
            center_pos = pad_left + center_pos
        emb = self._crop_1bp(emb, req.resolution, req.window_bp, center_pos)
        return {key: _tensor_to_encoded(val) for key, val in emb.items()}

    def _run_embed_batch(self, req: EmbedBatchRequest) -> dict:
        import torch

        dna, org = self._prepare_input_batch(req.sequences, req.organism, req.use_dna_parser)
        resolutions = self._resolutions_tuple(req.resolution)

        compute_dtype = self.model.dtype_policy.compute_dtype
        use_amp = compute_dtype != torch.float32

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
                emb = self.model.encode(dna, org, resolutions=resolutions)

        if req.center_pos is not None:
            results = []
            for i, seq in enumerate(req.sequences):
                pad_left = (SEQ_LEN - len(seq)) // 2
                buf_center = pad_left + req.center_pos
                single_emb = {k: v[i:i+1] for k, v in emb.items()}
                single_emb = self._crop_1bp(single_emb, req.resolution, req.window_bp, buf_center)
                results.append({k: _tensor_to_encoded(v) for k, v in single_emb.items()})
        else:
            emb = self._crop_1bp(emb, req.resolution, req.window_bp, None)

            # Move all tensors to CPU once, then encode in parallel across sequences.
            B = len(req.sequences)
            cpu_emb = {key: val.detach().cpu() for key, val in emb.items()}

            def _encode_seq(i):
                return {key: _tensor_to_encoded(t[i : i + 1]) for key, t in cpu_emb.items()}

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=B) as pool:
                results = list(pool.map(_encode_seq, range(B)))

        return {"results": results}

    # -------------------------------------------------------------------------
    # FastAPI routes — defined inside serve() so they close over `self`
    # -------------------------------------------------------------------------

    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI(
            title="AlphaGenome PyTorch Inference API",
            version="0.1.0",
        )

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "weights": WEIGHTS_PATH}

        @web_app.post("/predict")
        async def predict(req: PredictRequest):
            """Run prediction heads on a DNA sequence.

            Returns base64-encoded arrays with shape and dtype metadata.
            Decode on the client with:
                import numpy as np, base64
                arr = np.frombuffer(base64.b64decode(item["data"]),
                                    dtype=item["dtype"]
                       ).reshape(item["shape"])
            """
            return self._run_predict(req)

        @web_app.post("/embed")
        async def embed(req: EmbedRequest):
            """Extract trunk embeddings for a single sequence."""
            return self._run_embed(req)

        @web_app.post("/embed-batch")
        async def embed_batch(req: EmbedBatchRequest):
            """Extract trunk embeddings for a batch of sequences in one GPU call.

            Stacks all sequences into a single (B, 131072, 4) tensor, runs
            encode() once, then splits results back per sequence.
            Returns {"results": [<embed_dict>, ...]}, one entry per input sequence.
            """
            return self._run_embed_batch(req)

        return web_app
