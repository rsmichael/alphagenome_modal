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
    .pip_install(
        "torch",
        "numpy",
        "fastapi",
        "pydantic",
        "safetensors",
        "zstandard",
    )
    .apt_install("git")
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
            "Center position (0-based) within the sequence for the 1bp crop window. "
            "Required when resolution=1 and window_bp is set. "
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
        description="Center position (0-based) for 1bp crop. Defaults to sequence midpoint.",
    )
    window_bp: Optional[int] = Field(
        default=None,
        description="Number of 1bp positions to return around center_pos. Only applies when resolution=1.",
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
# Modal service class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume.read_only()},
    scaledown_window=300,
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
        print("Model ready.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _pad(self, sequence: str) -> str:
        if len(sequence) < SEQ_LEN:
            return sequence + "N" * (SEQ_LEN - len(sequence))
        return sequence[:SEQ_LEN]

    def _prepare_input(self, sequence: str, organism: str):
        import torch
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

        dna = sequence_to_onehot_tensor(self._pad(sequence), device="cuda").unsqueeze(0)
        org = torch.tensor([ORGANISM_INDEX[organism]], device="cuda")
        return dna, org

    def _prepare_input_batch(self, sequences: list[str], organism: str):
        import numpy as np
        import torch
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot

        # Stack all sequences into one numpy array, then do a single GPU transfer.
        # Faster than B individual sequence_to_onehot_tensor calls + torch.stack.
        onehots = np.stack([sequence_to_onehot(self._pad(s)) for s in sequences])  # (B, SEQ_LEN, 4) uint8
        dna = torch.from_numpy(onehots.astype(np.float32)).cuda()
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

        dna, org = self._prepare_input(req.sequence, req.organism)
        resolutions = self._resolutions_tuple(req.resolution)

        compute_dtype = self.model.dtype_policy.compute_dtype
        use_amp = compute_dtype != torch.float32

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
                emb = self.model.encode(dna, org, resolutions=resolutions)

        emb = self._crop_1bp(emb, req.resolution, req.window_bp, req.center_pos)
        return {key: _tensor_to_encoded(val) for key, val in emb.items()}

    def _run_embed_batch(self, req: EmbedBatchRequest) -> dict:
        import torch

        dna, org = self._prepare_input_batch(req.sequences, req.organism)
        resolutions = self._resolutions_tuple(req.resolution)

        compute_dtype = self.model.dtype_policy.compute_dtype
        use_amp = compute_dtype != torch.float32

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
                emb = self.model.encode(dna, org, resolutions=resolutions)

        emb = self._crop_1bp(emb, req.resolution, req.window_bp, req.center_pos)

        # Split batch dimension: (B, S, C) → list of (1, S, C) encoded dicts
        B = len(req.sequences)
        results = []
        for i in range(B):
            results.append({
                key: _tensor_to_encoded(val[i : i + 1])
                for key, val in emb.items()
            })

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
