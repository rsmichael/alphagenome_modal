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
    """Encode a numpy array as base64 with shape/dtype metadata."""
    import numpy as np
    arr = np.asarray(arr)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _tensor_to_encoded(t) -> dict:
    import torch
    arr = t.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr = arr.float()
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
        self.model = torch.compile(self.model, mode="reduce-overhead")
        print("Model ready.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _prepare_input(self, sequence: str, organism: str):
        import torch
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

        if len(sequence) < SEQ_LEN:
            sequence = sequence + "N" * (SEQ_LEN - len(sequence))
        else:
            sequence = sequence[:SEQ_LEN]

        dna = sequence_to_onehot_tensor(sequence, device="cuda").unsqueeze(0)
        org = torch.tensor([ORGANISM_INDEX[organism]], device="cuda")
        return dna, org

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

        # Crop 1bp embeddings to the requested window before serializing.
        # embeddings_1bp shape: (1, SEQ_LEN, 1536) — SEQ_LEN=131072
        if req.resolution in (1, None) and "embeddings_1bp" in emb and req.window_bp is not None:
            seq_len = emb["embeddings_1bp"].shape[1]
            center = req.center_pos if req.center_pos is not None else seq_len // 2
            half = req.window_bp // 2
            start = max(0, center - half)
            end = min(seq_len, start + req.window_bp)
            emb["embeddings_1bp"] = emb["embeddings_1bp"][:, start:end, :]

        return {key: _tensor_to_encoded(val) for key, val in emb.items()}

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
            """Extract trunk embeddings without running prediction heads.

            Useful for fine-tuning: run this once per sequence, then train
            a lightweight head on the returned embeddings.
            """
            return self._run_embed(req)

        return web_app
