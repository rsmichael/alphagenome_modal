# alphagenome modal 

The goal of this repo is to enable inference from the alphagenome model using microservices deployed on Modal.
Some initial code was written for the original code release from DeepMind, but new code will focus on deployment 
of the alphagenome model as written in pytorch.

A key goal of the microservices is eventually to enable few shot learning for additional tasks not included in 
the alphagenome paper, particularly for RNA properties such as branchpoints and more complicated splicing objectives than
those that were included in the paper. Eventually other models may be included in this project with similar architecture.


## code organization

the alphagenome-pytorch has been vendored at ./alphagenome-pytorch. There is also a directory for marimo notebooks and 
the modal_alphagenome package.

alphagenome-pytorch current code is here: https://github.com/genomicsxai/alphagenome-pytorch/tree/main
docs are here: https://alphagenome-pytorch.readthedocs.io/en/latest/

modal_alphagenome/inference_agtorch.py is the file where I will develop the microservice for alphagenome_pytorch

The old JAX-based inference code is still in modal_alphagenome/inference.py (uses alphagenome_research and JAX).
The new pytorch-based work in inference_agtorch.py is independent of that.


## model API summary (alphagenome-pytorch)

Key facts about the pytorch model, useful for implementation:

**Input format:**
- One-hot encoded DNA: `(B, 131072, 4)` — the standard 131,072 bp window
- Use `alphagenome_pytorch.utils.sequence.sequence_to_onehot_tensor(seq_str)` to encode
- Organism index: 0 = human, 1 = mouse

**Forward pass / inference:**
```python
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

model = AlphaGenome.from_pretrained('model.pth', device='cuda')
model.eval()

dna = sequence_to_onehot_tensor(sequence).unsqueeze(0).to('cuda')  # (1, 131072, 4)
outputs = model.predict(dna, organism_index=0)  # wraps forward with no_grad + autocast
```

**Output heads** (all float32 from `predict()`):
| Head key | Tracks | Resolutions | Notes |
|---|---|---|---|
| `atac` | 256 | 1bp, 128bp | chromatin accessibility |
| `dnase` | 384 | 1bp, 128bp | |
| `procap` | 128 | 1bp, 128bp | |
| `cage` | 640 | 1bp, 128bp | |
| `rna_seq` | 768 | 1bp, 128bp | |
| `chip_tf` | 1664 | 128bp only | |
| `chip_histone` | 1152 | 128bp only | |
| `pair_activations` | 28 | contact map (S/2048 × S/2048) | |
| `splice_sites_classification` | — | 1bp | requires 1bp resolution |
| `splice_sites_usage` | 734/180 | 1bp | human/mouse |
| `splice_sites_junction` | 367/90 | 1bp | human/mouse |

Output tensors are dicts: e.g. `outputs['atac'][1]` is the 1bp atac prediction (shape `(B, S, 256)`),
`outputs['atac'][128]` is the 128bp prediction (shape `(B, S//128, 256)`).

**Embeddings** (for fine-tuning):
- `model.predict(dna, 0, return_embeddings=True)` → also includes `embeddings_1bp` (B, S, 1536), `embeddings_128bp` (B, S//128, 3072)
- `model.encode(dna, 0, resolutions=(128,))` → only embeddings, skips all heads (fast)
- Skipping 1bp resolution (`resolutions=(128,)`) avoids the expensive decoder pass

**Efficiency flags:**
- `resolutions=(128,)` — skip the decoder, only get 128bp-resolution heads (ATAC, CHIP, etc.) and embeddings
- `embeddings_only=True` — skip all prediction heads entirely
- `encoder_only=True` — only CNN encoder (fastest, no transformer/decoder)
- `dtype_policy=DtypePolicy.mixed_precision()` — bfloat16 compute (matches JAX, ~2x faster on GPU)

**Weights:**
- Weights are stored as `.pth` (or `.safetensors`) converted from the JAX checkpoint via `alphagenome-pytorch/scripts/convert_weights.py`
- The JAX checkpoint is already on the `alphagenome-models` Modal volume at `/models/huggingface_cache/`
- The pytorch weights need to be produced by running the conversion script as a Modal function (requires JAX)


## memory requirements (Mac M-series vs GPU)

- Model parameters: ~400–600M params; ~1.6–2.4 GB at float32, ~0.8–1.2 GB at bfloat16
- Peak activation memory at 131,072 bp input: dominated by the 1bp decoder output `(1, 131072, 768)` ≈ 384 MB float32, plus 1bp embeddings `(1, 131072, 1536)` ≈ 768 MB float32
- Rough estimate: 4–8 GB peak RAM for inference at float32; less with bfloat16 + `resolutions=(128,)`
- **Mac M-series (36 GB unified memory): running is feasible.** MPS backend is supported by PyTorch. Use `device='mps'` or `device='cpu'`. Inference will be slow (minutes per sequence) but possible for development/testing.
- **Modal GPU (A10G/H100): recommended for production.** bfloat16 mixed precision fits comfortably.


## development steps (revised)

### Step 1 — Fix pixi.toml

Current issues:
- **Python 3.14** is too new; many packages don't support it yet. Downgrade to **3.12** (required by alphagenome-pytorch, and stable).
- **`jax[cuda12]`** is in pypi-dependencies unconditionally, which will fail on Mac ARM. Remove it from the main dependencies section (JAX is not needed for pytorch inference). If JAX is still needed for the conversion script, add it as a separate pixi feature/environment or just run the conversion on Modal.
- Keep `alphagenome-pytorch = { path = "./alphagenome-pytorch", editable = true }` — this is correct.
- May need to add `safetensors` for `.safetensors` weight loading support.

### Step 2 — Convert JAX weights to PyTorch

The JAX checkpoint already lives on the `alphagenome-models` Modal volume. The pytorch model needs a `.pth` weight file.

**Approach:** Add a new Modal function (e.g. in `modal_alphagenome/model_setup_torch.py`) that:
1. Mounts the `alphagenome-models` volume read-write
2. Uses the JAX-capable image (already defined in `model_setup.py`)
3. Adds the vendored `alphagenome-pytorch` source to the image
4. Runs `scripts/convert_weights.py --input /models/huggingface_cache/... --output /models/model.pth`
5. Commits the volume so `model.pth` is persisted

The conversion requires: JAX, orbax-checkpoint, alphagenome (JAX package), and alphagenome-pytorch (pytorch package).

Output: `/models/model.pth` on the `alphagenome-models` volume, available to inference containers.

### Step 3 — Local testing in a Marimo notebook

Once weights are available (or using random weights), verify the model runs locally.

Target file: `notebooks/alphagenome_pytorch_demo.py` (marimo notebook)

Key things to demonstrate in the notebook:
1. Load model from local `.pth` or with random weights for shape testing
2. Encode a sequence (e.g. 131,072 bp of random bases)
3. Run `model.predict()` with selected outputs
4. Show output shapes for each head
5. Demonstrate `resolutions=(128,)` shortcut for speed
6. Show embedding extraction with `model.encode()`
7. Optionally: run on `device='mps'` to test Mac GPU acceleration

For development without real weights, random weights work fine to verify shapes and code paths.

### Step 4 — Modal microservice (inference_agtorch.py)

Build a FastAPI endpoint in `modal_alphagenome/inference_agtorch.py`.

**Image:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "fastapi", "pydantic", "safetensors")
    .add_local_dir("alphagenome-pytorch/src", "/root/alphagenome_pytorch_src", copy=True)
    .run_commands("pip install /root/alphagenome_pytorch_src")
)
```
Note: No JAX needed for inference — the pytorch model loads from `.pth` only.

**Volume:** Mount the existing `alphagenome-models` volume read-only to access `model.pth`.

**GPU:** A10G is sufficient for inference; H100 for higher throughput or longer sequences.

**Endpoints:**

`POST /predict`
- Input: `{ "sequence": str, "organism": "human"|"mouse", "outputs": ["atac", "rna_seq", ...], "resolution": 1|128, "return_embeddings": bool }`
- Sequence is validated (len must be ≤ 131072, padded with N if shorter, or chunked if longer)
- Output: dict of head name → array (returned as nested JSON or msgpack)
- For large outputs (1bp resolution, all heads), the response can be very large — consider returning only selected heads or only 128bp resolution by default

`POST /embed`
- Input: `{ "sequence": str, "organism": "human"|"mouse", "resolution": 1|128 }`  
- Runs `model.encode()` with `embeddings_only=True` — skips all heads, just returns embeddings
- Useful for the fine-tuning use case (extract features, then train a lightweight head elsewhere)

`GET /health`
- Returns model load status

**Model caching:** Load model once at container startup using `@app.cls` with `@modal.enter()` for clean lifecycle management.

**Serialization:** For numpy arrays, return base64-encoded binary blobs with shape metadata rather than nested JSON lists (much more compact and faster to parse).

### Step 5 — Fine-tuning infrastructure (future)

The alphagenome-pytorch package already includes a full fine-tuning pipeline (`scripts/finetune.py`) supporting:
- Linear probing (frozen backbone + custom head)
- LoRA fine-tuning
- Full fine-tuning
- Multi-modality training
- DDP (multi-GPU)

For the RNA properties use case (branchpoints, novel splicing):
- The `model.encode()` method returns `embeddings_1bp` (B, 131072, 1536) — suitable for branchpoint prediction (1bp resolution)
- Fine-tuning modes: prefer LoRA to avoid catastrophic forgetting; linear probe for quick experiments
- The `extensions/finetuning/heads.py` module provides ready-made head classes
- Training data pipeline (`extensions/finetuning/datasets.py`) handles bigwig files and bed regions

A Modal-based fine-tuning job (separate from inference) would:
1. Mount training data volume
2. Run `scripts/finetune.py` with appropriate flags  
3. Save checkpoints to a training volume
4. The inference service can then load the fine-tuned weights


## open questions / decisions

1. **Sequence chunking for sequences > 131,072 bp:** The model has a fixed 131,072 bp window. For longer genomic regions, need a sliding window approach (already partially implemented in `extensions/inference/full_chromosome.py`). The inference API should clarify whether it accepts only 131,072 bp windows or handles tiling automatically.

2. **Output serialization format:** JSON with nested lists is very slow for large arrays. Options: base64 numpy binary, msgpack, or returning a file download. Decide before building the client.

3. **Track metadata:** The model outputs N tracks per head, but track identities (which tissue, which experiment) come from the JAX metadata files (`nonzero_mean`, track names, etc.). These are extracted during `convert_weights.py` and stored as `track_means`. Need to also store/expose the track metadata DataFrame so callers know which index corresponds to which experiment.

4. **Weight file location on volume:** Needs to be agreed upon before Step 2 — suggest `/models/model_pytorch.pth` or `/models/model_pytorch.safetensors` (safetensors is faster to load).
