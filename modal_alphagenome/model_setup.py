"""AlphaGenome model setup with H100 GPU for downloading and loading weights."""

import modal

app = modal.App("alphagenome-model-setup")

# Build image with CUDA, JAX, and AlphaGenome dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11"  # AlphaGenome may not support 3.14 yet
    )
    .apt_install(
        "git",
        "build-essential",  # C/C++ compilers (gcc, g++)
        "clang",            # Clang compiler (needed by sorted_nearest)
        "pkg-config",       # Package configuration
    )
    .pip_install(
        "jax[cuda12]",
        "huggingface-hub",
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    # Copy and install alphagenome_research from local directory
    .add_local_dir(
        local_path="modal_alphagenome/alphagenome_research",
        remote_path="/root/alphagenome_research",
        copy=True,  # Copy into image to allow further build steps
    )
    .run_commands(
        "pip install /root/alphagenome_research"
    )
)

# Create persistent volume for model storage
model_volume = modal.Volume.from_name(
    "alphagenome-models",
    create_if_missing=True,
)


@app.function(
    image=image,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="H100",
    timeout=3600,  # 1 hour for large downloads
)
def download_model():
    """Download AlphaGenome model weights to persistent volume.

    This function:
    1. Authenticates with HuggingFace using HF_TOKEN
    2. Downloads the full AlphaGenome model from HuggingFace
    3. Stores the model in a Modal volume for persistence
    4. Commits the volume to ensure the model is saved

    Returns:
        Status message indicating success or failure
    """
    import os
    from pathlib import Path
    from huggingface_hub import login

    # Set HuggingFace cache to our volume
    cache_dir = Path("/models/huggingface_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

    # Authenticate with HuggingFace
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "ERROR: HF_TOKEN not found. Create Modal secret first."

    print(f"Authenticating with HuggingFace (token: {token[:10]}...)")
    login(token=token)

    # Check if model already exists in cache
    cache_files = list(cache_dir.glob("**/*"))
    if len(cache_files) > 10:  # Some files already cached
        print(f"Found {len(cache_files)} cached files")
        print("Model may already be downloaded")

    # Download model using AlphaGenome's official method
    print(f"Downloading AlphaGenome model to {cache_dir}...")
    print("This may take a while depending on model size...")

    try:
        from alphagenome_research.model import dna_model

        # Download model - this will cache to HF_HOME
        model = dna_model.create_from_huggingface('all_folds')

        print("✓ Model downloaded successfully")
        print(f"Model loaded into memory. Type: {type(model)}")

        # Check what was downloaded
        cache_files_after = list(cache_dir.glob("**/*"))
        print(f"Total files in cache: {len(cache_files_after)}")

        # Commit volume to persist changes
        print("Committing volume to persist model...")
        model_volume.commit()

        return f"SUCCESS: Model downloaded and cached ({len(cache_files_after)} files)"

    except Exception as e:
        import traceback
        return f"ERROR: Failed to download model: {str(e)}\n{traceback.format_exc()}"


@app.function(
    image=image,
    volumes={"/models": model_volume.read_only()},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="H100",
    timeout=600,
)
def verify_model():
    """Verify that the AlphaGenome model loads correctly from the volume.

    This function:
    1. Checks that model files exist in the volume
    2. Attempts to load the model
    3. Prints model information and structure
    4. Verifies the model is ready for inference

    Returns:
        Dict with verification status and model info
    """
    import os
    from pathlib import Path
    from huggingface_hub import login

    # Set HuggingFace cache to our volume (same as download)
    cache_dir = Path("/models/huggingface_cache")
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

    # Authenticate with HuggingFace
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # Check if cache directory exists
    if not cache_dir.exists():
        return {
            "success": False,
            "error": "Cache directory not found. Run download_model first.",
        }

    # List files in cache directory
    cache_files = list(cache_dir.glob("**/*"))
    file_count = len([f for f in cache_files if f.is_file()])
    print(f"Found {file_count} files in HuggingFace cache")

    if file_count == 0:
        return {
            "success": False,
            "error": "No model files found in cache. Run download_model first.",
        }

    try:
        print("Loading AlphaGenome model from cache...")
        from alphagenome_research.model import dna_model

        # Load model from cache
        model = dna_model.create_from_huggingface('all_folds')

        print("✓ Model loaded successfully")
        print(f"Model type: {type(model)}")

        # Try to get model information
        model_info = {
            "type": str(type(model)),
            "cached_files": file_count,
        }

        # Check if model has common attributes
        if hasattr(model, '__dict__'):
            attrs = list(model.__dict__.keys())
            print(f"Model has {len(attrs)} attributes")
            if attrs:
                print(f"First few attributes: {attrs[:5]}...")

        return {
            "success": True,
            "info": model_info,
            "message": "Model verified and ready for inference",
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Failed to load model: {str(e)}\n{traceback.format_exc()}",
        }


@app.local_entrypoint()
def main():
    """Download and verify AlphaGenome model."""
    print("=" * 70)
    print("AlphaGenome Model Setup (H100 GPU)")
    print("=" * 70)

    # Step 1: Download model
    print("\n[1/2] Downloading model to Modal volume...")
    print("-" * 70)
    download_result = download_model.remote()
    print(f"\nResult: {download_result}")

    # Step 2: Verify model
    print("\n" + "=" * 70)
    print("[2/2] Verifying model loads correctly...")
    print("-" * 70)
    verify_result = verify_model.remote()

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS:")
    print("=" * 70)

    if verify_result.get("success"):
        print("✓ SUCCESS!")
        print(f"\n{verify_result['message']}")
        print(f"\nModel info:")
        for key, value in verify_result.get("info", {}).items():
            print(f"  {key}: {value}")
    else:
        print("✗ VERIFICATION FAILED!")
        print(f"\nError: {verify_result.get('error')}")

    print("=" * 70)
