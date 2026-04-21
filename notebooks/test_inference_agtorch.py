"""Quick smoke test for the alphagenome-inference-torch Modal service."""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import base64
    import numpy as np
    import requests

    return base64, mo, np, requests


@app.cell
def _(mo):
    url_input = mo.ui.text(
        value="https://rsmichael--alphagenome-inference-torch-alphagenomese-2f9b47-dev.modal.run",
        label="Modal service URL (from `pixi run serve-torch-temp`)",
        full_width=True,
    )
    url_input
    return (url_input,)


@app.cell
def _(requests, url_input):
    BASE_URL = url_input.value.rstrip("/")

    resp = requests.get(f"{BASE_URL}/health", timeout=60)
    resp.json()
    return (BASE_URL,)


@app.cell
def _(mo):
    mo.md("""
    ## /predict — 128bp resolution, atac head only
    """)
    return


@app.cell
def _(BASE_URL, requests):
    predict_resp = requests.post(
        f"{BASE_URL}/predict",
        json={
            "sequence": "ATCG" * 512,   # 2048 bp, will be padded to 131072
            "organism": "human",
            "heads": ["atac"],
            "resolution": 128,
        },
        timeout=300,
    )
    predict_resp.raise_for_status()
    predict_data = predict_resp.json()
    predict_data["heads"].keys()
    return (predict_data,)


@app.cell
def _(base64, np, predict_data):
    def decode(item):
        raw = base64.b64decode(item["data"])
        if item.get("encoding") == "zstd":
            import zstandard as zstd
            raw = zstd.ZstdDecompressor().decompress(raw)
        return np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"])

    atac_128 = decode(predict_data["heads"]["atac"]["128"])
    print(f"atac[128] shape: {atac_128.shape}")   # expect (1, 1024, 256)
    print(f"mean: {atac_128.mean():.4f}  min: {atac_128.min():.4f}  max: {atac_128.max():.4f}")
    return (decode,)


@app.cell
def _(mo):
    mo.md("""
    ## /embed
    """)
    return


@app.cell
def _(BASE_URL, decode, requests):
    embed_resp = requests.post(
        f"{BASE_URL}/embed",
        json={
            "sequence": "ATCG" * 32768,
            "organism": "human",
            "resolution": 1,
        },
        timeout=300,
    )
    embed_resp.raise_for_status()
    embed_data = embed_resp.json()

    for key, item in embed_data.items():
        arr = decode(item)
        print(f"{key}: {arr.shape}  dtype={arr.dtype}")
    return


@app.cell
def _(BASE_URL, decode, np, requests):
    def _():
        embed_resp = requests.post(
            f"{BASE_URL}/embed",
            json={
                "sequence": "ATCG" * 32768,
                "organism": "human",
                "resolution": 1,
                "window_bp" : 10000
            },
            timeout=300,
        )
        embed_resp.raise_for_status()
        embed_data = embed_resp.json()

        for key, item in embed_data.items():
            arr = decode(item)
            print(np.mean(arr), np.max(arr), np.min(arr))
            print(f"{key}: {arr.shape}  dtype={arr.dtype}")
    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## /embed-batch vs sequential /embed — 1bp resolution

    Memory per sequence at 1bp (encoder intermediates + decoder + 1bp embeddings):
    - Encoder skip connections: ~441 MB
    - Decoder output `(768, 131072)` bfloat16: ~192 MB
    - 1bp embeddings `(1536, 131072)` bfloat16: ~384 MB
    - **Total: ~1 GB/seq**

    A10G has 24 GB; weights take ~2 GB → safe batch size is **B=4**, maybe B=8.
    Using `window_bp=10_000` (central 10K positions) to keep responses manageable.
    """)
    return


@app.cell
def _(BASE_URL, decode, requests):
    import time
    def batch_run(BATCH_SIZE = 4):
        WINDOW = 10_000
        SEQS = ["ATCG" * 512, "GCTA" * 512, "TTAA" * 512, "CCGG" * 512] *8
        SEQS = SEQS[:BATCH_SIZE]

        # --- batched ---
        t0 = time.time()
        r_batch = requests.post(
            f"{BASE_URL}/embed-batch",
            json={
                "sequences": SEQS,
                "organism": "human",
                "resolution": 1,
                "window_bp": WINDOW,
            },
            timeout=600,
        )
        r_batch.raise_for_status()
        t_batch = time.time() - t0
        batch_results = r_batch.json()["results"]

        # --- sequential ---
        t1 = time.time()
        seq_results = []
        for s in SEQS:
            r = requests.post(
                f"{BASE_URL}/embed",
                json={"sequence": s, "organism": "human", "resolution": 1, "window_bp": WINDOW},
                timeout=600,
            )
            r.raise_for_status()
            seq_results.append(r.json())
        t_seq = time.time() - t1

        print(f"Batched  (B={BATCH_SIZE}): {t_batch:.1f}s  ({t_batch/BATCH_SIZE:.1f}s/seq)")
        print(f"Sequential:       {t_seq:.1f}s  ({t_seq/BATCH_SIZE:.1f}s/seq)")
        print(f"Speedup: {t_seq/t_batch:.1f}x")

        for key, item in batch_results[0].items():
            arr = decode(item)
            print(f"  result[0] {key}: {arr.shape}  dtype={arr.dtype}")

    batch_run(4)
    return batch_run, time


@app.cell
def _(batch_run):
    batch_run(4)
    return


@app.cell
def _(batch_run):
    batch_run(3)
    return


@app.cell
def _(batch_run):
    batch_run(16)
    return


@app.cell
def _(BASE_URL, requests, time):
    BATCH_SIZE = 12
    WINDOW = 10_000
    SEQS = ["ATCG" * 512, "GCTA" * 512, "TTAA" * 512, "CCGG" * 512] *4
    SEQS = SEQS[:BATCH_SIZE]

    # --- batched ---
    t0 = time.time()
    r_batch = requests.post(
        f"{BASE_URL}/embed-batch",
        json={
            "sequences": SEQS,
            "organism": "human",
            "resolution": 1,
            "window_bp": WINDOW,
        },
        timeout=600,
    )
    r_batch.raise_for_status()
    t_batch = time.time() - t0
    batch_results = r_batch.json()["results"]
    return


if __name__ == "__main__":
    app.run()
