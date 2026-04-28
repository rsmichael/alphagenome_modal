"""Client functions for the AlphaGenome Modal inference service."""

import base64
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import polars as pl
import requests


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def decode(item: dict) -> np.ndarray:
    """Decode a zstd+base64 encoded embedding from an HTTP response dict."""
    raw = base64.b64decode(item["data"])
    if item.get("encoding") == "zstd":
        import zstandard as zstd
        raw = zstd.ZstdDecompressor().decompress(raw)
    return np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"])


def _make_tensor_col(arrays: list, pa):
    """Stack float16 ndarrays into a (tensor_type, ExtensionArray) pair."""
    import pyarrow as pa
    mat = np.stack(arrays)
    tensor_type = pa.fixed_shape_tensor(pa.float16(), mat.shape[1:])
    flat = pa.array(mat.ravel(), type=pa.float16())
    storage = pa.FixedSizeListArray.from_arrays(flat, int(np.prod(mat.shape[1:])))
    return tensor_type, pa.ExtensionArray.from_storage(tensor_type, storage)


# ---------------------------------------------------------------------------
# HTTP path
# ---------------------------------------------------------------------------

def embed_dataframe(
    df: pl.DataFrame,
    seq_col: str,
    *,
    base_url: str,
    batch_size: int = 4,
    max_workers: int = 8,
    resolution: int = 1,
    window_bp: int = 10_000,
    organism: str = "human",
    npy_out: Optional[str] = None,
    lance_out: Optional[str] = None,
    lance_write_mode: str = "overwrite",
) -> pl.DataFrame:
    """Compute AlphaGenome embeddings for every sequence in `seq_col` via HTTP.

    Batches are dispatched concurrently (up to `max_workers` in-flight at once)
    so multiple Modal containers run in parallel.

    Args:
        df:          Input DataFrame. Must contain `seq_col`.
        seq_col:     Column holding DNA sequences.
        base_url:    Modal service URL (from `pixi run serve-torch-temp`).
        batch_size:  Sequences per /embed-batch call.
        max_workers: Max concurrent requests.
        resolution:  1 or 128.
        window_bp:   1bp crop window; only used when resolution=1.
        organism:    'human' or 'mouse'.
        npy_out:     If given, full 1bp embeddings saved here as (N, W, 1536) float16.
        lance_out:        If given, full embeddings written incrementally to a local Lance
                          dataset at this path. Nothing accumulates in RAM. Returns df unchanged.
        lance_write_mode: How to handle an existing Lance dataset at `lance_out`.
                          'overwrite' (default) replaces data but keeps Lance version history.
                          'append' adds rows (use for growing datasets; dedup by embedded_at).
                          'fail' raises an error if the dataset already exists.

    Returns:
        In-memory mode: df with columns appended:
          emb_1bp_mean   pl.Array(Float16, 1536)
          emb_128bp_mean pl.Array(Float16, 3072)
          npy_row        pl.UInt32  [if npy_out set]
        Lance mode: df unchanged.
    """
    sequences = df[seq_col].to_list()
    n = len(sequences)
    n_batches = math.ceil(n / batch_size)

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
        if lance_write_mode not in ("overwrite", "append", "fail"):
            raise ValueError(
                f"lance_write_mode must be 'overwrite', 'append', or 'fail', got {lance_write_mode!r}"
            )

        import lance
        import pyarrow as pa
        from datetime import datetime, timezone

        _embed_ts = datetime.now(timezone.utc)
        _lance_lock = threading.Lock()
        _lance_initialized = [False]

        def _write_to_lance(b: int, results: list):
            start_row = b * batch_size
            table = df.slice(start_row, len(results)).to_arrow()
            table = table.append_column(
                pa.field("row_idx", pa.int32()),
                pa.array(range(start_row, start_row + len(results)), type=pa.int32()),
            )
            table = table.append_column(
                pa.field("embedded_at", pa.timestamp("us", tz="UTC")),
                pa.array([_embed_ts] * len(results), type=pa.timestamp("us", tz="UTC")),
            )

            emb_1bp, emb_128bp, emb_pair = [], [], []
            for item_dict in results:
                if "embeddings_1bp" in item_dict:
                    emb_1bp.append(decode(item_dict["embeddings_1bp"])[0].astype(np.float16))
                if "embeddings_128bp" in item_dict:
                    emb_128bp.append(decode(item_dict["embeddings_128bp"])[0].astype(np.float16))
                if "embeddings_pair" in item_dict:
                    emb_pair.append(decode(item_dict["embeddings_pair"])[0].astype(np.float16))

            for col_name, arrays in [
                ("embeddings_1bp", emb_1bp),
                ("embeddings_128bp", emb_128bp),
                ("embeddings_pair", emb_pair),
            ]:
                if arrays:
                    tensor_type, col_arr = _make_tensor_col(arrays, pa)
                    table = table.append_column(pa.field(col_name, tensor_type), col_arr)

            with _lance_lock:
                if not _lance_initialized[0]:
                    if lance_write_mode == "overwrite":
                        mode = "overwrite"
                    elif lance_write_mode == "append":
                        mode = "append"
                    else:  # "fail"
                        mode = "create"
                else:
                    mode = "append"
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
    batch_results = [None] * n_batches
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, args): args[0] for args in batches}
        for fut in as_completed(futures):
            b, results = fut.result()
            batch_results[b] = results
            print(f"  batch {b + 1}/{n_batches} done ({len(batches[b][1])} seqs)")

    mean_1bp_rows, mean_128bp_rows, full_1bp_chunks = [], [], []
    for results in batch_results:
        for item_dict in results:
            if "embeddings_1bp" in item_dict:
                arr = decode(item_dict["embeddings_1bp"])[0].astype(np.float16)
                mean_1bp_rows.append(arr.mean(axis=0))
                if npy_out is not None:
                    full_1bp_chunks.append(arr[np.newaxis])
            if "embeddings_128bp" in item_dict:
                arr = decode(item_dict["embeddings_128bp"])[0].astype(np.float16)
                mean_128bp_rows.append(arr.mean(axis=0))

    new_cols: dict[str, pl.Series] = {}
    if mean_1bp_rows:
        stacked = np.stack(mean_1bp_rows)
        new_cols["emb_1bp_mean"] = pl.Series(
            "emb_1bp_mean", stacked.tolist(), dtype=pl.Array(pl.Float16, stacked.shape[1])
        )
    if mean_128bp_rows:
        stacked = np.stack(mean_128bp_rows)
        new_cols["emb_128bp_mean"] = pl.Series(
            "emb_128bp_mean", stacked.tolist(), dtype=pl.Array(pl.Float16, stacked.shape[1])
        )
    if npy_out is not None and full_1bp_chunks:
        full = np.concatenate(full_1bp_chunks, axis=0)
        np.save(npy_out, full)
        print(f"Saved full 1bp embeddings -> {npy_out}  shape={full.shape}")
        new_cols["npy_row"] = pl.Series("npy_row", list(range(n)), dtype=pl.UInt32)

    return df.with_columns(list(new_cols.values()))


# ---------------------------------------------------------------------------
# Modal SDK path (writes directly to Modal volume, no embedding transfer)
# ---------------------------------------------------------------------------

def embed_to_volume(
    df: pl.DataFrame,
    seq_col: str,
    dataset_name: str,
    *,
    batch_size: int = 4,
    max_workers: int = 8,
    resolution: int = 1,
    window_bp: int = 10_000,
    organism: str = "human",
    use_dna_parser: bool = False,
) -> list[dict]:
    """Embed sequences and write to a Lance dataset on the Modal volume.

    Uses the Modal Python SDK — sequences and metadata travel to Modal via pickle;
    the embedding arrays never cross the network.

    Args:
        df:           Input Polars DataFrame.
        seq_col:      Column containing DNA sequences.
        dataset_name: Lance dataset name on the volume (no extension).

    Returns:
        List of per-batch metadata dicts returned by each embed_batch call.
    """
    import modal

    EmbedToLance = modal.Cls.from_name("alphagenome-inference-torch", "EmbedToLance")

    sequences = df[seq_col].to_list()
    metadata = df.drop(seq_col).to_dicts()
    n = len(sequences)
    n_batches = math.ceil(n / batch_size)
    batches = [
        (b, sequences[b * batch_size:(b + 1) * batch_size],
            metadata[b * batch_size:(b + 1) * batch_size])
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
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, item): item[0] for item in batches}
        for fut in as_completed(futures):
            meta = fut.result()
            results.append(meta)
            print(f"  batch done: {meta}")

    EmbedToLance().commit_volume.remote()
    return results
