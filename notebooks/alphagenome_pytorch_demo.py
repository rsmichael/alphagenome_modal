"""Marimo notebook for exploring the alphagenome-pytorch model locally."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # AlphaGenome PyTorch — Local Inference Demo

    Runs the pytorch model locally (CPU or MPS). Uses random weights unless a `.pth`
    file is provided, which is sufficient to verify shapes and code paths.
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

    # Pick the best available device.
    # MPS is skipped: Conv1d ops with sequence-length in the channel axis hit
    # the MPS hard limit of 65536 output channels (131072 bp > 65536).
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Using device: {DEVICE}")
    return DEVICE, AlphaGenome, DtypePolicy, np, sequence_to_onehot_tensor, torch


@app.cell
def _(mo):
    weights_input = mo.ui.text(
        value="",
        label="Path to model.pth (leave blank for random weights)",
        full_width=True,
    )
    weights_input
    return (weights_input,)


@app.cell
def _(DEVICE, AlphaGenome, DtypePolicy, weights_input):
    import os

    weights_path = weights_input.value.strip()

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path} ...")
        model = AlphaGenome.from_pretrained(
            weights_path,
            dtype_policy=DtypePolicy.full_float32(),
            device=DEVICE,
        )
        print("Weights loaded.")
    else:
        if weights_path:
            print(f"Warning: {weights_path!r} not found — using random weights.")
        else:
            print("No weights path provided — using random weights.")
        model = AlphaGenome(dtype_policy=DtypePolicy.full_float32())
        model.to(DEVICE)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.0f}M)")
    return model, os, weights_path


@app.cell
def _(mo):
    mo.md("""
    ## Sequence encoding

    The model expects a 131,072 bp window. Sequences shorter than that are
    padded with N (all-zeros in one-hot encoding). For quick shape checks a
    shorter test sequence works fine — the architecture is fully convolutional
    down to the trunk, so any multiple of 128 is accepted.
    """)
    return


@app.cell
def _(mo):
    seq_input = mo.ui.text_area(
        value="ATCG" * 512,  # 2048 bp — fast for local testing
        label="DNA sequence (ACGTN)",
        full_width=True,
    )
    seq_input
    return (seq_input,)


@app.cell
def _(DEVICE, seq_input, sequence_to_onehot_tensor, torch):
    SEQ_LEN = 131_072
    raw_seq = seq_input.value.strip().upper()

    # Pad or truncate to SEQ_LEN
    if len(raw_seq) < SEQ_LEN:
        padded_seq = raw_seq + "N" * (SEQ_LEN - len(raw_seq))
    else:
        padded_seq = raw_seq[:SEQ_LEN]

    dna = sequence_to_onehot_tensor(padded_seq, device=DEVICE).unsqueeze(0)  # (1, 131072, 4)
    organism = torch.tensor([0], device=DEVICE)  # 0 = human

    print(f"Input sequence length: {len(raw_seq)} bp (padded to {dna.shape[1]})")
    print(f"dna tensor shape: {dna.shape}, dtype: {dna.dtype}, device: {dna.device}")
    return SEQ_LEN, dna, organism, padded_seq, raw_seq


@app.cell
def _(mo):
    mo.md("""
    ## Forward pass — 128bp resolution only

    `resolutions=(128,)` skips the expensive decoder and 1bp heads.
    This is the fast path: useful for ATAC, ChIP, CAGE at coarse resolution,
    and for embedding extraction.
    """)
    return


@app.cell
def _(dna, model, organism):
    import time

    t0 = time.time()
    outputs_128 = model.predict(dna, organism, resolutions=(128,))
    elapsed_128 = time.time() - t0

    print(f"Elapsed (128bp only): {elapsed_128:.2f}s\n")
    print("Output keys:", list(outputs_128.keys()))
    for _k, _v in outputs_128.items():
        if isinstance(_v, dict):
            for _res, _t in _v.items():
                print(f"  {_k}[{_res}]: {tuple(_t.shape)}")
        elif isinstance(_v, torch.Tensor):
            print(f"  {_k}: {tuple(_v.shape)}")
    return elapsed_128, outputs_128, t0, time


@app.cell
def _(mo):
    mo.md("""
    ## Forward pass — 1bp resolution

    Runs the full decoder. Includes splice heads and 1bp predictions for
    ATAC, DNase, CAGE, RNA-seq, PRO-cap.
    """)
    return


@app.cell
def _(dna, model, organism, time):
    t1 = time.time()
    outputs_1bp = model.predict(dna, organism, resolutions=(1, 128))
    elapsed_1bp = time.time() - t1

    print(f"Elapsed (1bp + 128bp): {elapsed_1bp:.2f}s\n")
    for _k, _v in outputs_1bp.items():
        if isinstance(_v, dict):
            for _res, _t in _v.items():
                print(f"  {_k}[{_res}]: {tuple(_t.shape)}")
        elif isinstance(_v, torch.Tensor):
            print(f"  {_k}: {tuple(_v.shape)}")
    return elapsed_1bp, outputs_1bp, t1


@app.cell
def _(mo):
    mo.md("""
    ## Embedding extraction

    `model.encode()` runs encoder + transformer + decoder + embedders but
    skips all prediction heads. Useful for fine-tuning: extract features
    once, then train a lightweight head.
    """)
    return


@app.cell
def _(dna, model, organism, time):
    # 128bp embeddings only — skip decoder entirely
    t2 = time.time()
    emb_128 = model.encode(dna, organism, resolutions=(128,))
    elapsed_emb_128 = time.time() - t2

    print(f"Elapsed (embed 128bp only): {elapsed_emb_128:.2f}s")
    print(f"  embeddings_128bp: {tuple(emb_128['embeddings_128bp'].shape)}  (B, S//128, 3072)")
    print(f"  embeddings_pair:  {tuple(emb_128['embeddings_pair'].shape)}")
    return elapsed_emb_128, emb_128, t2


@app.cell
def _(dna, model, organism, time):
    # 1bp embeddings — includes decoder
    t3 = time.time()
    emb_1bp = model.encode(dna, organism, resolutions=(1,))
    elapsed_emb_1bp = time.time() - t3

    print(f"Elapsed (embed 1bp): {elapsed_emb_1bp:.2f}s")
    print(f"  embeddings_1bp:   {tuple(emb_1bp['embeddings_1bp'].shape)}  (B, S, 1536)")
    print(f"  embeddings_128bp: {tuple(emb_1bp['embeddings_128bp'].shape)}")
    return elapsed_emb_1bp, emb_1bp, t3


@app.cell
def _(mo):
    mo.md("""
    ## Output inspection

    Peek at values from a single head.
    """)
    return


@app.cell
def _(mo, outputs_128):
    head_options = [f"{k}[{r}]" for k, v in outputs_128.items() if isinstance(v, dict) for r in v]
    head_selector = mo.ui.dropdown(
        options=head_options,
        value=head_options[0] if head_options else None,
        label="Head to inspect",
    )
    head_selector
    return head_options, head_selector


@app.cell
def _(head_selector, np, outputs_128):
    if head_selector.value:
        _head, _res = head_selector.value.split("[")
        _res = int(_res.rstrip("]"))
        _tensor = outputs_128[_head][_res].detach().cpu().numpy()  # (1, S, C)
        _track_means = _tensor[0].mean(axis=0)  # mean over positions, shape (C,)
        print(f"{head_selector.value}  shape={_tensor.shape}")
        print(f"  position-mean across tracks: min={_track_means.min():.4f}  max={_track_means.max():.4f}  mean={_track_means.mean():.4f}")
        print(f"  first position, first 8 tracks: {_tensor[0, 0, :8]}")
    return


if __name__ == "__main__":
    app.run()
