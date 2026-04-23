"""Quick smoke test for the alphagenome-inference-torch Modal service."""

import marimo

__generated_with = "0.23.2"
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
        value="https://rsmichael--alphagenome-inference-torch-alphagenomeservice-serve.modal.run",
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
            "use_dna_parser": False
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
    return (batch_run,)


@app.cell
def _(batch_run):
    batch_run(4)
    return


@app.cell
def _(batch_run):
    batch_run(3)
    return


@app.cell
def _():
    # batch_run(5)
    return


@app.cell
def _():
    # BATCH_SIZE = 12
    # WINDOW = 10_000
    # SEQS = ["ATCG" * 512, "GCTA" * 512, "TTAA" * 512, "CCGG" * 512] *4
    # SEQS = SEQS[:BATCH_SIZE]

    # # --- batched ---
    # t0 = time.time()
    # r_batch = requests.post(
    #     f"{BASE_URL}/embed-batch",
    #     json={
    #         "sequences": SEQS,
    #         "organism": "human",
    #         "resolution": 1,
    #         "window_bp": WINDOW,
    #     },
    #     timeout=600,
    # )
    # r_batch.raise_for_status()
    # t_batch = time.time() - t0
    # batch_results = r_batch.json()["results"]
    return


@app.cell
def _(mo):
    mo.md("""
    ## embed_dataframe — batch-embed a Polars column of sequences

    AlphaGenome embeddings per sequence:

    | Column | Shape | Type |
    |---|---|---|
    | `embeddings_1bp` | `(window_bp, 1536)` | `fixed_shape_tensor(float16)` |
    | `embeddings_128bp` | `(1024, 3072)` | `fixed_shape_tensor(float16)` |
    | `embeddings_pair` | `(64, 64, 128)` | `fixed_shape_tensor(float16)` |

    **In-memory mode** (default): mean-pooled 1bp and 128bp vectors are appended
    as `pl.Array(Float16, D)` columns on the returned DataFrame.

    **Lance mode** (`lance_out` path): every batch is written to a Lance dataset
    incrementally as it completes — nothing accumulates in RAM. Full embedding
    arrays are stored as `pa.fixed_shape_tensor` columns alongside all original
    df columns and a `row_idx` int32 key.
    """)
    return


@app.cell
def _(BASE_URL, decode, np, requests):
    import math
    import threading
    import polars as pl

    def _make_tensor_col(arrays: list, pa) -> "pa.ChunkedArray":
        """Stack a list of float16 ndarrays into a pa.fixed_shape_tensor column."""
        mat = np.stack(arrays)  # (batch, *shape)
        tensor_type = pa.fixed_shape_tensor(pa.float16(), mat.shape[1:])
        storage = pa.FixedSizeListArray.from_arrays(
            pa.array(mat.ravel().tolist(), type=pa.float16()),
            int(np.prod(mat.shape[1:])),
        )
        return pa.field("_", tensor_type), pa.ExtensionArray.from_storage(tensor_type, storage)

    def embed_dataframe(
        df: pl.DataFrame,
        seq_col: str,
        *,
        base_url: str = BASE_URL,
        batch_size: int = 4,
        max_workers: int = 8,
        resolution: int = 1,
        window_bp: int = 10_000,
        organism: str = "human",
        npy_out: str | None = None,
        lance_out: str | None = None,
    ) -> pl.DataFrame:
        """Compute AlphaGenome embeddings for every sequence in `seq_col`.

        Batches are dispatched concurrently (up to `max_workers` in-flight at
        once) so multiple Modal containers are used in parallel.

        Args:
            df:          Input DataFrame. Must contain `seq_col`.
            seq_col:     Name of the column holding DNA sequences (str).
            base_url:    Modal service URL.
            batch_size:  Sequences per /embed-batch call.
            max_workers: Max concurrent requests (= max containers used).
                         Match to the service's `max_containers` setting.
            resolution:  1 or 128 (passed to the service).
            window_bp:   1bp crop window size; only used when resolution=1.
            organism:    'human' or 'mouse'.
            npy_out:     If given, full 1bp embeddings are saved to this path
                         (shape N x window_bp x 1536, float16) with a `npy_row`
                         column added to the returned DataFrame.
            lance_out:   If given, full embeddings are written incrementally to
                         a Lance dataset at this path as each batch completes.
                         Nothing accumulates in RAM. Columns written per row:
                           - `embeddings_1bp`  fixed_shape_tensor(float16, (window_bp, 1536))
                           - `embeddings_128bp` fixed_shape_tensor(float16, (1024, 3072))
                           - `embeddings_pair`  fixed_shape_tensor(float16, (64, 64, 128))
                         Plus all original df columns and `row_idx` int32.
                         Returns the original df unchanged.

        Returns:
            In-memory mode: original DataFrame with embedding columns appended:
              - `emb_1bp_mean`   pl.Array(Float16, 1536)   [if resolution in (1, None)]
              - `emb_128bp_mean` pl.Array(Float16, 3072)   [if resolution in (128, None)]
              - `npy_row`        pl.UInt32                 [if npy_out is set]
            Lance mode (lance_out set): original DataFrame unchanged.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sequences = df[seq_col].to_list()
        n = len(sequences)
        n_batches = math.ceil(n / batch_size)

        # Build all batch payloads upfront, keyed by batch index
        batches = []
        for b in range(n_batches):
            batch = sequences[b * batch_size : (b + 1) * batch_size]
            payload = {"sequences": batch, "organism": organism, "resolution": resolution}
            if resolution == 1:
                payload["window_bp"] = window_bp
            batches.append((b, batch, payload))

        def _call(args):
            b, _, payload = args
            resp = requests.post(f"{base_url}/embed-batch", json=payload, timeout=600)
            resp.raise_for_status()
            return b, resp.json()["results"]

        if lance_out is not None:
            import lance
            import pyarrow as pa

            _lance_lock = threading.Lock()
            _lance_initialized = [False]

            def _write_to_lance(b: int, results: list):
                start_row = b * batch_size
                table = df.slice(start_row, len(results)).to_arrow()
                table = table.append_column(
                    pa.field("row_idx", pa.int32()),
                    pa.array(range(start_row, start_row + len(results)), type=pa.int32()),
                )

                emb_1bp, emb_128bp, emb_pair = [], [], []
                for item_dict in results:
                    if "embeddings_1bp" in item_dict:
                        emb_1bp.append(decode(item_dict["embeddings_1bp"])[0].astype(np.float16))    # (W, 1536)
                    if "embeddings_128bp" in item_dict:
                        emb_128bp.append(decode(item_dict["embeddings_128bp"])[0].astype(np.float16))  # (1024, 3072)
                    if "embeddings_pair" in item_dict:
                        emb_pair.append(decode(item_dict["embeddings_pair"])[0].astype(np.float16))  # (64, 64, 128)

                for col_name, arrays in [
                    ("embeddings_1bp", emb_1bp),
                    ("embeddings_128bp", emb_128bp),
                    ("embeddings_pair", emb_pair),
                ]:
                    if arrays:
                        _, col_arr = _make_tensor_col(arrays, pa)
                        mat = np.stack(arrays)
                        tensor_type = pa.fixed_shape_tensor(pa.float16(), mat.shape[1:])
                        table = table.append_column(pa.field(col_name, tensor_type), col_arr)

                with _lance_lock:
                    mode = "create" if not _lance_initialized[0] else "append"
                    lance.write_dataset(table, lance_out, mode=mode)
                    _lance_initialized[0] = True

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_call, args): args[0] for args in batches}
                for fut in as_completed(futures):
                    b, results = fut.result()
                    _write_to_lance(b, results)
                    print(f"  batch {b + 1}/{n_batches} done ({len(batches[b][1])} seqs) -> lance")

            print(f"Lance dataset written to {lance_out}")
            return df

        # --- in-memory path ---
        # Dispatch concurrently; collect results indexed by batch position
        batch_results = [None] * n_batches
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call, args): args[0] for args in batches}
            for fut in as_completed(futures):
                b, results = fut.result()
                batch_results[b] = results
                print(f"  batch {b + 1}/{n_batches} done ({len(batches[b][1])} seqs)")

        # Flatten in original sequence order
        mean_1bp_rows = []
        mean_128bp_rows = []
        full_1bp_chunks = []

        for results in batch_results:
            for item_dict in results:
                if "embeddings_1bp" in item_dict:
                    arr = decode(item_dict["embeddings_1bp"])  # (1, W, 1536)
                    arr = arr[0].astype(np.float16)            # (W, 1536)
                    mean_1bp_rows.append(arr.mean(axis=0))     # (1536,)
                    if npy_out is not None:
                        full_1bp_chunks.append(arr[np.newaxis])

                if "embeddings_128bp" in item_dict:
                    arr = decode(item_dict["embeddings_128bp"])  # (1, 1024, 3072)
                    arr = arr[0].astype(np.float16)              # (1024, 3072)
                    mean_128bp_rows.append(arr.mean(axis=0))     # (3072,)

        new_cols: dict[str, pl.Series] = {}

        if mean_1bp_rows:
            stacked = np.stack(mean_1bp_rows)  # (N, 1536)
            new_cols["emb_1bp_mean"] = pl.Series(
                "emb_1bp_mean",
                stacked.tolist(),
                dtype=pl.Array(pl.Float16, stacked.shape[1]),
            )

        if mean_128bp_rows:
            stacked = np.stack(mean_128bp_rows)  # (N, 3072)
            new_cols["emb_128bp_mean"] = pl.Series(
                "emb_128bp_mean",
                stacked.tolist(),
                dtype=pl.Array(pl.Float16, stacked.shape[1]),
            )

        if npy_out is not None and full_1bp_chunks:
            full = np.concatenate(full_1bp_chunks, axis=0)  # (N, W, 1536)
            np.save(npy_out, full)
            print(f"Saved full 1bp embeddings -> {npy_out}  shape={full.shape}")
            new_cols["npy_row"] = pl.Series("npy_row", list(range(n)), dtype=pl.UInt32)

        return df.with_columns([v for v in new_cols.values()])

    return embed_dataframe, math, pl


@app.cell
def _():
    return


@app.cell
def _():
    import duckdb


    return (duckdb,)


@app.cell
def _():
    import os
    os.listdir('..')
    return


@app.cell
def _():
    return


@app.cell
def _():

    # dataset = lance.dataset("../demo_embeddings.lance")
    # results = dataset.to_table()
    # conn = duckdb.connect()
    # conn.execute("INSTALL lance; LOAD lance;")
    # conn.register("embeddings", results)

    # db = lancedb.connect("demo_embeddings.lance")
    # duckdb.query("SELECT * FROM 'demo_embeddings.lance' LIMIT 1").to_df()

    # table = db.open_table("my_table")


    # conn.query("""
    # SELECT *
    # FROM embeddings
    # LIMIT 1
    # """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # demo_df = pl.DataFrame({
    #     "name":     ["seq_A", "seq_B", "seq_C", "seq_D", "seq_E", "seq_F"]*10,
    #     "sequence": ["ATCG" * 512, "GCTA" * 512, "TTAA" * 512,
    #                  "CCGG" * 512, "AATT" * 512, "GGCC" * 512]*10,
    # })

    # result_df = embed_dataframe(
    #     demo_df,
    #     seq_col="sequence",
    #     batch_size=3,
    #     resolution=1,
    #     window_bp=10_000,
    #     max_workers=10,
    #     lance_out="demo_embeddings3.lance",
    # )

    # print(result_df.select(["name", "emb_1bp_mean"]))
    return


@app.cell
def _(result_df):
    result_df
    return


@app.cell
def _():
    import lance

    ds = lance.dataset("demo_embeddings.lance")
    print(ds.schema)
    print(f"{ds.count_rows()} rows")
    return ds, lance


@app.cell
def _(lance):
    import lancedb
    db1 = lancedb.connect("./")
    print(db1.list_tables())
    table = db1.open_table("demo_embeddings3")
    dataset1 = lance.dataset("demo_embeddings3.lance")
    result1 = dataset1.scanner(
        filter="name == 'seq_A'",
        columns=None,
        limit = 1
    ).to_table()
    return


@app.cell
def _(duckdb, lance, my_table):
    dataset = lance.dataset("demo_embeddings3.lance")

    # Register and query with DuckDB
    duckdb.register("my_table", dataset)
    result = duckdb.sql("SELECT * FROM my_table LIMIT 10").df()
    return


@app.cell
def _(ds, pl):
    # Read all rows back, sorted by row_idx to match original df order
    tbl = pl.from_arrow(ds.to_table()).sort("row_idx")
    tbl.select(["name", "row_idx", "emb_1bp_mean"])
    return


@app.cell
def _(ds, pl):
    # Filter to a single name
    tbl_filtered = pl.from_arrow(
        ds.to_table(filter="name = 'seq_A'")
    ).sort("row_idx")
    tbl_filtered.select(["name", "row_idx", "emb_1bp_mean"])
    return


@app.cell
def _():
    # import lancedb

    # db = lancedb.connect(".")
    # lance_tbl = db.open_table("demo_embeddings")

    # # Brute-force nearest-neighbour search on emb_1bp_mean
    # query_vec = np.zeros(1536, dtype=np.float32)
    # hits = (
    #     lance_tbl.search(query_vec, vector_column_name="emb_1bp_mean")
    #     .limit(5)
    #     .to_arrow()
    # )
    # pl.from_arrow(hits).select(["name", "row_idx", "_distance"])
    return


@app.cell
def _(mo):
    mo.md("""
    ## embed_to_volume — Modal SDK path (no HTTP embedding transfer)

    Instead of serializing embeddings over HTTP, `EmbedToLance.embed_batch` writes
    directly to a Lance dataset on a Modal volume. Only sequences + metadata travel
    to Modal (via pickle); only row-index metadata comes back. The heavy embedding
    arrays never cross the network.

    Compare with `embed_dataframe(lance_out=...)` which goes:
    GPU → zstd+base64 → HTTP → decode → write local Lance.

    | | HTTP path | Volume path |
    |---|---|---|
    | Data sent to Modal | sequences (JSON) | sequences + metadata (pickle) |
    | Data returned | embeddings (zstd+b64) | row indices only |
    | Lance written by | client | Modal container |
    | Lance location | local | Modal volume |
    """)
    return


@app.cell
def _():
    import modal as _modal
    EmbedToLance = _modal.Cls.from_name("alphagenome-inference-torch", "EmbedToLance")
    return (EmbedToLance,)


@app.cell
def _(EmbedToLance):
    EmbedToLance
    return


@app.cell
def _(EmbedToLance, math):
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed

    def embed_to_volume(
        df,
        seq_col: str,
        dataset_name: str,
        *,
        batch_size: int = 4,
        max_workers: int = 8,
        resolution: int = 1,
        window_bp: int = 10_000,
        organism: str = "human",
        use_dna_parser: bool = False,
    ):
        """Embed sequences and write to a Lance dataset on the Modal volume.

        Args:
            df:           Input Polars DataFrame.
            seq_col:      Column containing DNA sequences.
            dataset_name: Lance dataset name on the volume (no .lance extension).

        Returns:
            List of metadata dicts returned by each batch call.
        """
        sequences = df[seq_col].to_list()
        metadata  = df.drop(seq_col).to_dicts()
        n = len(sequences)
        n_batches = math.ceil(n / batch_size)
        batches = [
            (b, sequences[b*batch_size:(b+1)*batch_size], metadata[b*batch_size:(b+1)*batch_size])
            for b in range(n_batches)
        ]

        def _call(item):
            b, seqs, meta = item
            return EmbedToLance().embed_batch.remote(
                seqs, dataset_name, meta,
                start_row=b * batch_size,
                resolution=resolution,
                window_bp=window_bp,
                organism=organism,
                use_dna_parser=use_dna_parser,
            )

        results = []
        with _TPE(max_workers=max_workers) as pool:
            futures = {pool.submit(_call, item): item[0] for item in batches}
            for fut in _as_completed(futures):
                meta = fut.result()
                results.append(meta)
                print(f"  batch done: {meta}")
        EmbedToLance().commit_volume.remote()
        return results

    return (embed_to_volume,)


@app.cell
def _():
    return


@app.cell
def _(embed_dataframe, embed_to_volume, pl):
    import time as _time

    _demo_df = pl.DataFrame({
        "name":     ["seq_A", "seq_B", "seq_C", "seq_D", "seq_E", "seq_F"] * 40,
        "sequence": ["ATCG" * 512, "GCTA" * 512, "TTAA" * 512,
                     "CCGG" * 512, "AATT" * 512, "GGCC" * 512] * 40,
    })

    _t0 = _time.time()
    embed_dataframe(_demo_df, "sequence", lance_out=f"demo_timing_http_{str(_t0)}.lance",
                    batch_size=3, resolution=1, window_bp=1, max_workers=10)
    _t_http = _time.time() - _t0
    print(_t_http)

    _t0 = _time.time()
    embed_dataframe(_demo_df, "sequence", lance_out=f"demo_timing_http_{str(_t0)}.lance",
                    batch_size=3, resolution=1, window_bp=10_000, max_workers=10)
    _t_http_10k = _time.time() - _t0
    print(_t_http_10k)

    _t0 = _time.time()
    embed_to_volume(_demo_df, "sequence", f"demo_timing_volume_{str(_t0)}",
                    batch_size=3, resolution=1, window_bp=1, max_workers=10)
    _t_vol = _time.time() - _t0


    _t0 = _time.time()
    embed_to_volume(_demo_df, "sequence", f"demo_timing_volume_{str(_t0)}",
                    batch_size=3, resolution=1, window_bp=10_000, max_workers=10)
    _t_vol_10k = _time.time() - _t0

    print(f"\nHTTP + local Lance:  {_t_http:.1f}s, 10k context output: {_t_http_10k:.1f}s")
    print(f"Modal volume Lance: {_t_vol:.1f}s, 10k context output: {_t_vol_10k:.1f}s")
    return


@app.cell
def _(EmbedToLance):
    import pyarrow as _pa
    import polars as _pl

    _raw = EmbedToLance.query.remote(
        "demo_timing_volume",
        filter_expr="name = 'seq_A'",
        columns=["row_idx", "name", "embeddings_128bp"],
    )
    _tbl = _pa.ipc.open_stream(_raw).read_all()
    _pl.from_arrow(_tbl)
    return


if __name__ == "__main__":
    app.run()
