"""AlphaGenome embeddings written as Parquet files to a Modal volume.

Eliminates the HTTP serialization round-trip of the existing /embed-batch endpoint.
Each batch is written as a single Parquet file; queries use DuckDB for SQL filtering.

Embeddings are stored as pa.FixedSizeList(float16) columns (Parquet-native).
Reshape on the client:
    arr = np.array(row["embeddings_1bp"]).reshape(window_bp, 1536)

Client usage (Modal Python SDK):
    import modal
    EmbedToLance = modal.Cls.from_name("alphagenome-inference-torch", "EmbedToLance")

    # Write embeddings to the volume
    meta = EmbedToLance().embed_batch.remote(sequences, "my_dataset", metadata, start_row=0)

    # Query back a filtered subset
    import pyarrow as pa, polars as pl
    raw = EmbedToLance().query.remote("my_dataset", filter_expr="chrom = 'chr1'")
    tbl = pa.ipc.open_stream(raw).read_all()
    pl.from_arrow(tbl)
"""

import modal
from typing import Optional

from modal_alphagenome.inference_agtorch import (
    app,
    image,
    model_volume,
    WEIGHTS_PATH,
    ORGANISM_INDEX,
    _seq_to_gpu,
    _seq_to_gpu_dna_parser,
    _crop_1bp,
    _resolutions_tuple,
)

# ---------------------------------------------------------------------------
# Volume for persisting embeddings
# ---------------------------------------------------------------------------

embeddings_volume = modal.Volume.from_name("alphagenome-embeddings", create_if_missing=True)
EMBEDDINGS_MOUNT = "/embeddings"

# Shape of each embedding type as a function of window_bp
_EMBED_SHAPES = {
    "embeddings_1bp":   lambda window_bp: (window_bp, 1536),
    "embeddings_128bp": lambda window_bp: (1024, 3072),
    "embeddings_pair":  lambda window_bp: (64, 64, 128),
}


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    volumes={
        "/models": model_volume.read_only(),
        EMBEDDINGS_MOUNT: embeddings_volume,
    },
    scaledown_window=300,
    max_containers=10,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
class EmbedToLance:
    """Runs AlphaGenome inference and writes embeddings as Parquet to a Modal volume."""

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
        lookup = torch.zeros(128, 4, dtype=torch.float32, device="cuda")
        for idx, ch in enumerate("ACGT"):
            lookup[ord(ch)] = torch.eye(4)[idx]
            lookup[ord(ch.lower())] = torch.eye(4)[idx]
        self._onehot_lookup = lookup
        print("Model ready.")

    @modal.method()
    def embed_batch(
        self,
        sequences: list[str],
        dataset_name: str,
        metadata: list[dict],
        *,
        start_row: int = 0,
        organism: str = "human",
        resolution: int = 1,
        window_bp: int = 10_000,
        use_dna_parser: bool = False,
    ) -> dict:
        """Run inference and write embeddings as a Parquet file on the volume.

        Each call writes one file: /embeddings/<dataset_name>/<start_row:010d>.parquet.
        Concurrent containers write different files — no locking needed.

        Args:
            sequences:    DNA strings for this batch.
            dataset_name: Dataset name; files land in /embeddings/<name>/.
            metadata:     Per-row metadata dicts stored as extra columns.
            start_row:    Global row index of sequences[0]; used for the filename and row_idx.
            organism:     'human' or 'mouse'.
            resolution:   1 or 128.
            window_bp:    1bp crop window size (resolution=1 only).
            use_dna_parser: Use dna_parser CPU encoding instead of GPU lookup.

        Returns:
            {"dataset_name": ..., "row_indices": [...], "n_written": N}
        """
        import os
        import numpy as np
        import torch
        import pyarrow as pa
        import pyarrow.parquet as pq

        dna = _seq_to_gpu_dna_parser(sequences) if use_dna_parser else _seq_to_gpu(sequences, self._onehot_lookup)
        org = torch.full((len(sequences),), ORGANISM_INDEX[organism], dtype=torch.long, device="cuda")
        resolutions = _resolutions_tuple(resolution)

        compute_dtype = self.model.dtype_policy.compute_dtype
        use_amp = compute_dtype != torch.float32
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_amp):
                emb = self.model.encode(dna, org, resolutions=resolutions)

        emb = _crop_1bp(emb, resolution, window_bp, None)

        B = len(sequences)
        row_indices = list(range(start_row, start_row + B))

        # Build PyArrow table: row_idx + metadata columns + embedding columns
        cols: dict[str, pa.Array] = {
            "row_idx": pa.array(row_indices, type=pa.int32()),
        }
        if metadata:
            for key in metadata[0]:
                cols[key] = pa.array([row[key] for row in metadata])
        table = pa.table(cols)

        # Append one FixedSizeList column per embedding type (Parquet-native)
        for emb_key, tensor in emb.items():
            arr = tensor.detach().to(torch.float16).cpu().numpy()  # (B, *spatial)
            shape_fn = _EMBED_SHAPES.get(emb_key)
            shape = shape_fn(window_bp) if shape_fn else tuple(arr.shape[1:])
            flat_size = int(np.prod(shape))
            flat = pa.array(arr.reshape(B, -1).ravel(), type=pa.float16())
            col = pa.FixedSizeListArray.from_arrays(flat, flat_size)
            table = table.append_column(pa.field(emb_key, col.type), col)

        dataset_dir = os.path.join(EMBEDDINGS_MOUNT, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        pq.write_table(table, os.path.join(dataset_dir, f"{start_row:010d}.parquet"))

        return {"dataset_name": dataset_name, "row_indices": row_indices, "n_written": B}

    @modal.method()
    def commit_volume(self):
        """Commit the embeddings volume. Call once after all embed_batch calls finish."""
        embeddings_volume.commit()

    @modal.method()
    def query(
        self,
        dataset_name: str,
        *,
        filter_expr: Optional[str] = None,
        columns: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> bytes:
        """Query the Parquet dataset on the volume and return Arrow IPC bytes.

        Uses DuckDB for SQL-style filtering directly over the Parquet files.
        Only requested columns and filtered rows cross the network.

        filter_expr uses SQL syntax, e.g. "chrom = 'chr1' AND pos > 1000".

        Client deserializes with:
            import pyarrow as pa, polars as pl
            tbl = pa.ipc.open_stream(result_bytes).read_all()
            pl.from_arrow(tbl)
        """
        import io
        import os
        import duckdb
        import pyarrow as pa

        dataset_dir = os.path.join(EMBEDDINGS_MOUNT, dataset_name)
        col_list = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql = f"SELECT {col_list} FROM read_parquet('{dataset_dir}/*.parquet')"
        if filter_expr:
            sql += f" WHERE {filter_expr}"
        if limit is not None:
            sql += f" LIMIT {limit}"

        tbl = duckdb.connect().execute(sql).arrow()

        buf = io.BytesIO()
        with pa.ipc.new_stream(buf, tbl.schema) as writer:
            writer.write_table(tbl)
        return buf.getvalue()
