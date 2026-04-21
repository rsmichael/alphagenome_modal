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
        return np.frombuffer(
            base64.b64decode(item["data"]), dtype=item["dtype"]
        ).reshape(item["shape"])

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
def _(BASE_URL, decode, requests):
    _embed_resp = requests.post(
        f"{BASE_URL}/embed",
        json={
            "sequence": "ATCG" * 32768,
            "organism": "human",
            "resolution": 1,
            "center_pos" : 1000,
            "window" : 10
        },
        timeout=300,
    )
    _embed_resp.raise_for_status()
    _embed_data = _embed_resp.json()

    for _key, _item in _embed_data.items():
        _arr = decode(_item)
        print(f"{_key}: {_arr.shape}  dtype={_arr.dtype}")
    return


if __name__ == "__main__":
    app.run()
