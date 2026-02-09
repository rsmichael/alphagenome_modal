"""AlphaGenome inference API with FastAPI endpoints."""

import os
from pathlib import Path
from typing import Optional

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Create Modal app
app = modal.App("alphagenome-inference")

# Use same image as model_setup
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "pkg-config",
    )
    .pip_install(
        "jax[cuda12]",
        "huggingface-hub",
        "fastapi",
        "pydantic",
        "alphagenome",  # Required dependency for alphagenome_research
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    .add_local_dir(
        local_path="modal_alphagenome/alphagenome_research",
        remote_path="/root/alphagenome_research",
        copy=True,
    )
    .run_commands(
        "pip install /root/alphagenome_research"
    )
)

# Model volume (read-only)
model_volume = modal.Volume.from_name(
    "alphagenome-models",
    create_if_missing=True,
)

# Create FastAPI app
web_app = FastAPI(
    title="AlphaGenome Inference API",
    description="Genomic sequence analysis using the AlphaGenome model",
    version="0.1.0",
)

# Global model cache (per container)
model_cache = {}


def get_model():
    """Load and cache the AlphaGenome model."""
    if "model" in model_cache:
        return model_cache["model"]

    # Set HuggingFace cache to volume
    cache_dir = Path("/models/huggingface_cache")
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

    # Check if cache directory exists
    if not cache_dir.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Cache directory not found in volume. Please download the model first by running:\n"
                "pixi run setup-model"
            ),
        )

    # Get HuggingFace token (don't call login() - volume is read-only)
    token = os.environ.get("HF_TOKEN")

    # Load model from local snapshot
    try:
        from huggingface_hub import snapshot_download
        from alphagenome_research.model import dna_model

        print("Loading AlphaGenome model from cache...")

        # Find the downloaded model path
        local_model_path = snapshot_download(
            repo_id="google/alphagenome-all-folds",
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=True,  # Only use local files, don't download
        )
        print(f"Model path: {local_model_path}")

        # Load model from the local path
        model = dna_model.create(
            checkpoint_path=local_model_path,
            organism_settings=None,
            device=None,
        )
        print("✓ Model loaded successfully")

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model from cache: {str(e)}\nPlease run: pixi run setup-model",
        )

    model_cache["model"] = model
    return model


# Request/Response models
class PredictionRequest(BaseModel):
    """Request for sequence predictions."""

    sequences: list[str] = Field(
        ...,
        description="List of DNA sequences (ATCG characters)",
        min_length=1,
    )
    organism: str = Field(
        default="human",
        description="Organism type: 'human' or 'mouse'",
    )
    outputs: list[str] = Field(
        default=["ATAC", "RNA_SEQ"],
        description="Output modalities to predict",
    )
    tissues: Optional[list[str]] = Field(
        default=None,
        description="Optional ontology terms to filter by tissue type",
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v):
        """Validate DNA sequences."""
        for i, seq in enumerate(v):
            # Check length
            if len(seq) > 65536:
                raise ValueError(
                    f"Sequence {i} is too long ({len(seq)} bp). "
                    "Maximum supported length is 65,536 bp."
                )
            if len(seq) == 0:
                raise ValueError(f"Sequence {i} is empty.")

            # Check characters
            invalid_chars = set(seq.upper()) - set("ATCGN")
            if invalid_chars:
                raise ValueError(
                    f"Sequence {i} contains invalid characters: {invalid_chars}. "
                    "Only ATCG and N are allowed."
                )

        return v

    @field_validator("organism")
    @classmethod
    def validate_organism(cls, v):
        """Validate organism."""
        if v.lower() not in ["human", "mouse"]:
            raise ValueError(
                f"Invalid organism '{v}'. Must be 'human' or 'mouse'."
            )
        return v.lower()


class OutputData(BaseModel):
    """Output data for a single modality."""

    values: list[list[float]] = Field(
        ...,
        description="Prediction values as 2D array",
    )
    shape: list[int] = Field(
        ...,
        description="Shape of the prediction array",
    )
    resolution: int = Field(
        ...,
        description="Base pair resolution of predictions",
    )
    tracks: list[str] = Field(
        ...,
        description="Track names/metadata",
    )


class SequencePrediction(BaseModel):
    """Prediction result for a single sequence."""

    sequence_index: int = Field(
        ...,
        description="Index of the sequence in the request",
    )
    sequence_length: int = Field(
        ...,
        description="Length of the input sequence in base pairs",
    )
    outputs: dict[str, OutputData] = Field(
        ...,
        description="Predictions for each requested output type",
    )


class PredictionResponse(BaseModel):
    """Response containing predictions for all sequences."""

    predictions: list[SequencePrediction] = Field(
        ...,
        description="Predictions for each input sequence",
    )
    organism: str = Field(
        ...,
        description="Organism used for predictions",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    message: str


# API Endpoints
@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are ready."""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="AlphaGenome model loaded and ready for inference",
        )
    except HTTPException as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message=e.detail,
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            model_loaded=False,
            message=f"Error loading model: {str(e)}",
        )


@web_app.get("/debug-volume")
async def debug_volume():
    """Debug endpoint to check what's in the volume."""
    from pathlib import Path

    volume_root = Path("/models")
    cache_dir = Path("/models/huggingface_cache")

    # Check volume root
    volume_exists = volume_root.exists()
    volume_is_dir = volume_root.is_dir() if volume_exists else False

    # Check cache dir
    cache_exists = cache_dir.exists()
    cache_is_dir = cache_dir.is_dir() if cache_exists else False

    # List files in volume root
    volume_root_files = []
    if volume_exists:
        try:
            volume_root_files = [str(p.relative_to(volume_root)) for p in volume_root.iterdir()]
        except Exception as e:
            volume_root_files = [f"Error listing: {str(e)}"]

    # List files in cache dir (first 100)
    cache_files = []
    cache_file_count = 0
    if cache_exists:
        try:
            all_cache_files = list(cache_dir.glob("**/*"))
            cache_file_count = len(all_cache_files)
            # Show first 100 files
            cache_files = [str(p.relative_to(cache_dir)) for p in all_cache_files[:100]]
        except Exception as e:
            cache_files = [f"Error listing: {str(e)}"]

    return {
        "volume_root": str(volume_root),
        "volume_exists": volume_exists,
        "volume_is_dir": volume_is_dir,
        "volume_root_contents": volume_root_files,
        "cache_dir": str(cache_dir),
        "cache_exists": cache_exists,
        "cache_is_dir": cache_is_dir,
        "cache_file_count": cache_file_count,
        "cache_files_sample": cache_files[:20] if cache_files else [],
        "message": f"Found {cache_file_count} files in cache directory" if cache_exists else "Cache directory not found",
    }


@web_app.get("/predict-test")
async def predict_test():
    """Test prediction endpoint with hardcoded inputs for debugging."""
    import traceback

    try:
        print("=== PREDICT TEST STARTED ===")

        # Step 1: Load model
        print("Step 1: Loading model...")
        try:
            model = get_model()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {str(e)}")
            return {
                "status": "error",
                "step": "model_loading",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Step 2: Import modules
        print("Step 2: Importing modules...")
        try:
            from alphagenome.models import dna_output
            from alphagenome_research.model import dna_model as dm
            print("✓ Modules imported successfully")
        except Exception as e:
            print(f"✗ Import failed: {str(e)}")
            return {
                "status": "error",
                "step": "imports",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Step 3: Prepare inputs
        print("Step 3: Preparing inputs...")
        try:
            test_sequence = "ATCGATCG" * 256  # 2048 bp
            organism_enum = dm.Organism.HOMO_SAPIENS
            output_types = [dna_output.OutputType.ATAC]
            print(f"✓ Inputs prepared: {len(test_sequence)} bp sequence")
        except Exception as e:
            print(f"✗ Input preparation failed: {str(e)}")
            return {
                "status": "error",
                "step": "input_preparation",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Step 4: Run prediction
        print("Step 4: Running prediction...")
        try:
            result = model.predict_sequence(
                test_sequence,
                organism=organism_enum,
                requested_outputs=output_types,
                ontology_terms=None,
            )
            print("✓ Prediction completed")
        except Exception as e:
            print(f"✗ Prediction failed: {str(e)}")
            return {
                "status": "error",
                "step": "prediction",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Step 5: Extract output
        print("Step 5: Extracting output...")
        try:
            atac_output = result.atac
            if atac_output is None:
                print("✗ ATAC output is None")
                return {
                    "status": "error",
                    "step": "output_extraction",
                    "error": "ATAC output is None",
                }

            values_shape = atac_output.values.shape
            resolution = atac_output.resolution
            print(f"✓ Output extracted: shape={values_shape}, resolution={resolution}")
        except Exception as e:
            print(f"✗ Output extraction failed: {str(e)}")
            return {
                "status": "error",
                "step": "output_extraction",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Step 6: Return success
        print("=== PREDICT TEST COMPLETED SUCCESSFULLY ===")
        return {
            "status": "success",
            "sequence_length": len(test_sequence),
            "output_shape": list(values_shape),
            "resolution": resolution,
            "message": "Test prediction completed successfully",
        }

    except Exception as e:
        print(f"=== PREDICT TEST FAILED WITH UNEXPECTED ERROR ===")
        print(f"Error: {str(e)}")
        return {
            "status": "error",
            "step": "unexpected",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@web_app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run AlphaGenome inference on DNA sequences.

    Supports batch inference on multiple sequences. Sequences shorter than
    the model's processing length will be automatically padded with 'N'.

    Args:
        request: Prediction request with sequences and parameters

    Returns:
        Predictions for each sequence and requested output type
    """
    try:
        print("=== PREDICT STARTED ===")

        # Load model
        print("Step 1: Loading model...")
        try:
            model = get_model()
            print("✓ Model loaded")
        except HTTPException:
            raise
        except Exception as e:
            print(f"✗ Model loading failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}",
            )

        # Import necessary modules
        print("Step 2: Importing modules...")
        try:
            from alphagenome.models import dna_output
            from alphagenome_research.model import dna_model as dm
            print("✓ Modules imported")
        except ImportError as e:
            print(f"✗ Import failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error importing alphagenome modules: {str(e)}",
            )

        # Convert organism string to enum
        print(f"Step 3: Preparing inputs for {len(request.sequences)} sequence(s)...")
        organism_enum = (
            dm.Organism.HOMO_SAPIENS
            if request.organism == "human"
            else dm.Organism.MUS_MUSCULUS
        )
        print(f"✓ Organism: {request.organism}")

        # Convert output strings to enums
        try:
            output_types = [
                getattr(dna_output.OutputType, output_name.upper())
                for output_name in request.outputs
            ]
            print(f"✓ Output types: {request.outputs}")
        except AttributeError as e:
            print(f"✗ Invalid output type: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output type. Valid options: ATAC, CAGE, DNASE, "
                f"RNA_SEQ, PROCAP, CHIP_HISTONE, CHIP_TF, SPLICE_SITES, "
                f"SPLICE_SITE_USAGE, SPLICE_JUNCTIONS, CONTACT_MAPS",
            )

        # Process each sequence
        all_predictions = []

        for seq_idx, sequence in enumerate(request.sequences):
            original_length = len(sequence)
            print(f"\nStep 4.{seq_idx+1}: Processing sequence {seq_idx} ({original_length} bp)...")
            try:
                # Pad sequence to required length (model needs specific multiples)
                import math

                # Determine required length (power of 2, minimum 2048)
                if original_length <= 2048:
                    required_length = 2048
                else:
                    # Find next power of 2
                    required_length = 2 ** math.ceil(math.log2(original_length))

                # Pad if needed
                if original_length < required_length:
                    padding_needed = required_length - original_length
                    sequence = sequence + ('N' * padding_needed)
                    print(f"  Padded sequence: {original_length} → {required_length} bp (added {padding_needed} N's)")
                else:
                    print(f"  Sequence length OK: {original_length} bp")

                # Run prediction
                print(f"  Running prediction...")
                result = model.predict_sequence(
                    sequence,
                    organism=organism_enum,
                    requested_outputs=output_types,
                    ontology_terms=request.tissues,
                )
                print(f"  ✓ Prediction completed")

                # Extract outputs
                print(f"  Extracting {len(request.outputs)} output type(s)...")
                sequence_outputs = {}

                for output_name in request.outputs:
                    output_name_lower = output_name.lower()
                    print(f"    Processing {output_name} (attr: {output_name_lower})...")

                    # Get the output from result
                    output_data = getattr(result, output_name_lower, None)

                    if output_data is None:
                        print(f"    ⚠ {output_name} output is None, skipping")
                        continue

                    # Convert to JSON-serializable format
                    print(f"    Converting arrays to JSON (shape: {output_data.values.shape})...")
                    values_array = output_data.values
                    values_list = values_array.tolist()
                    print(f"    ✓ Converted to list")

                    # Get track metadata
                    tracks = []
                    if hasattr(output_data, 'metadata') and output_data.metadata is not None:
                        # Extract track names from metadata DataFrame
                        if hasattr(output_data.metadata, 'index'):
                            tracks = output_data.metadata.index.tolist()
                        elif hasattr(output_data.metadata, 'to_dict'):
                            tracks = list(output_data.metadata.to_dict().keys())
                    else:
                        # Fallback: generate generic track names
                        num_tracks = values_array.shape[1] if len(values_array.shape) > 1 else 1
                        tracks = [f"track_{i}" for i in range(num_tracks)]

                    # Convert all track names to strings (metadata may return integers)
                    tracks = [str(track) for track in tracks]

                    sequence_outputs[output_name] = OutputData(
                        values=values_list,
                        shape=list(values_array.shape),
                        resolution=output_data.resolution,
                        tracks=tracks,
                    )
                    print(f"    ✓ {output_name} extracted successfully")

                print(f"  Building response object...")
                all_predictions.append(
                    SequencePrediction(
                        sequence_index=seq_idx,
                        sequence_length=original_length,  # Return original length, not padded
                        outputs=sequence_outputs,
                    )
                )
                print(f"  ✓ Sequence {seq_idx} processed successfully")

            except Exception as e:
                import traceback
                print(f"  ✗ Error processing sequence {seq_idx}: {str(e)}")
                print(f"  Traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing sequence {seq_idx}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}",
                )

        print(f"\nStep 5: Building final response...")
        response = PredictionResponse(
            predictions=all_predictions,
            organism=request.organism,
        )
        print(f"✓ Response built successfully")
        print(f"=== PREDICT COMPLETED SUCCESSFULLY ===")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        print(f"=== PREDICT FAILED WITH HTTP EXCEPTION ===")
        raise
    except Exception as e:
        # Catch any unexpected errors and return them with details
        import traceback
        print(f"=== PREDICT FAILED WITH UNEXPECTED ERROR ===")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}",
        )


# Deploy as Modal web endpoint
@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": model_volume.read_only()},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    keep_warm=1,  # Keep one instance warm for faster responses
    container_idle_timeout=300,  # Keep containers alive for 5 minutes
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app as a Modal web endpoint."""
    return web_app
