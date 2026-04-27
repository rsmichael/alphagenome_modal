"""AlphaGenome embeddings written directly to a Lance dataset on a Modal volume (v2).

Eliminates the HTTP serialization round-trip of the existing /embed-batch endpoint:
instead of zstd+base64 encoding, network transfer, and client-side decoding, embeddings
are written as pa.fixed_shape_tensor columns directly on Modal infrastructure.

Requires Modal volume v2, which supports the atomic rename operations used by lance's
commit protocol.

Client usage (Modal Python SDK):
    import modal
    EmbedToLance = modal.Cls.from_name("alphagenome-inference-torch", "EmbedToLance")

    # Write embeddings to the volume
    meta = EmbedToLance().embed_batch.remote(sequences, "my_dataset", metadata, start_row=0)

    # Commit once after all batches finish
    EmbedToLance().commit_volume.remote()

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
# Volume for persisting embeddings (v2 required for lance atomic commits)
# ---------------------------------------------------------------------------

embeddings_volume = modal.Volume.from_name(
    "alphagenome-embeddings-v2", create_if_missing=True, version=2
)
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
    """Runs AlphaGenome inference and writes embeddings to Lance on a Modal volume (v2)."""

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
        """Run inference and write embeddings to a Lance dataset on the volume.

        Args:
            sequences:    DNA strings for this batch.
            dataset_name: Lance dataset name (stored at /embeddings/<name>).
            metadata:     Per-row metadata dicts stored as extra columns.
            start_row:    Global row index of sequences[0]; used to build row_idx column.
            organism:     'human' or 'mouse'.
            resolution:   1 or 128.
            window_bp:    1bp crop window size (resolution=1 only).
            use_dna_parser: Use dna_parser CPU encoding instead of GPU lookup.

        Returns:
            {"dataset_name": ..., "row_indices": [...], "n_written": N}
            No embedding data is returned — it stays on the volume.
        """
        import numpy as np
        import torch
        import lancedb
        import pyarrow as pa

        dna = org = emb = None
        try:
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

            # Build PyArrow table: row_idx + metadata columns + embedding tensor columns
            cols: dict[str, pa.Array] = {
                "row_idx": pa.array(row_indices, type=pa.int32()),
            }
            if metadata:
                for key in metadata[0]:
                    cols[key] = pa.array([row[key] for row in metadata])
            table = pa.table(cols)

            # Append one fixed_shape_tensor column per embedding type
            for emb_key, tensor in emb.items():
                arr = tensor.detach().to(torch.float16).cpu().numpy()  # (B, *spatial)
                shape_fn = _EMBED_SHAPES.get(emb_key)
                shape = shape_fn(window_bp) if shape_fn else tuple(arr.shape[1:])
                flat_size = int(np.prod(shape))
                tensor_type = pa.fixed_shape_tensor(pa.float16(), shape)
                storage = pa.FixedSizeListArray.from_arrays(
                    pa.array(arr.reshape(B, -1).ravel(), type=pa.float16()),
                    flat_size,
                )
                table = table.append_column(
                    pa.field(emb_key, tensor_type),
                    pa.ExtensionArray.from_storage(tensor_type, storage),
                )

            db = lancedb.connect(EMBEDDINGS_MOUNT)
            try:
                db.open_table(dataset_name).add(table)
            except Exception:
                db.create_table(dataset_name, table)

            return {"dataset_name": dataset_name, "row_indices": row_indices, "n_written": B}
        finally:
            del dna, org, emb
            torch.cuda.empty_cache()

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
        """Query the Lance dataset on the volume and return Arrow IPC bytes.

        Only the requested columns and filtered rows cross the network.

        Client deserializes with:
            import pyarrow as pa, polars as pl
            tbl = pa.ipc.open_stream(result_bytes).read_all()
            pl.from_arrow(tbl)
        """
        import io
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(EMBEDDINGS_MOUNT)
        lance_ds = db.open_table(dataset_name).to_lance()
        tbl = lance_ds.to_table(filter=filter_expr, columns=columns)
        if limit is not None:
            tbl = tbl.slice(0, limit)

        buf = io.BytesIO()
        with pa.ipc.new_stream(buf, tbl.schema) as writer:
            writer.write_table(tbl)
        return buf.getvalue()
