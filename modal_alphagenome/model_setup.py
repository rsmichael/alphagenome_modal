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

    print("=" * 70)
    print("BEFORE DOWNLOAD - Volume Inspection")
    print("=" * 70)

    # Check environment variables
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    print(f"XDG_CACHE_HOME: {os.environ.get('XDG_CACHE_HOME', 'not set')}")

    # List /models/ root
    models_root = Path("/models")
    if models_root.exists():
        print(f"\nContents of {models_root}:")
        for item in models_root.iterdir():
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"  [{item_type}] {item.name}")

    # Check cache directory before download
    if cache_dir.exists():
        cache_files_before = list(cache_dir.glob("**/*"))
        cache_files_list = [f for f in cache_files_before if f.is_file()]
        print(f"\nCache dir exists with {len(cache_files_list)} files (before download):")
        for f in cache_files_list[:10]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(cache_dir)} ({size_mb:.2f} MB)")

    # Authenticate with HuggingFace
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "ERROR: HF_TOKEN not found. Create Modal secret first."

    print(f"\nAuthenticating with HuggingFace (token: {token[:10]}...)")
    login(token=token)

    # Download model using explicit local download
    print("\n" + "=" * 70)
    print("DOWNLOADING MODEL")
    print("=" * 70)
    print(f"Target cache directory: {cache_dir}")
    print("This may take a while depending on model size...")

    try:
        from huggingface_hub import snapshot_download
        from alphagenome_research.model import dna_model

        # Explicitly download all files locally
        print("Downloading all model files locally with snapshot_download...")
        local_model_path = snapshot_download(
            repo_id="google/alphagenome-all-folds",
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=False,
            ignore_patterns=None,  # Download everything
        )
        print(f"✓ Model files downloaded to: {local_model_path}")

        # Load model from the downloaded local path
        print("Loading model from local files...")
        model = dna_model.create(
            checkpoint_path=local_model_path,
            organism_settings=None,
            device=None,
        )

        print("✓ Model loaded into memory")
        print(f"Model type: {type(model)}")

        print("\n" + "=" * 70)
        print("AFTER DOWNLOAD - Volume Inspection")
        print("=" * 70)

        # Check entire /models/ tree
        print(f"\nContents of {models_root}:")
        for item in models_root.iterdir():
            item_type = "DIR" if item.is_dir() else "FILE"
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  [{item_type}] {item.name} ({size_mb:.2f} MB)")
            else:
                # Count files in subdirectory
                try:
                    subfiles = list(item.glob("**/*"))
                    file_count = len([f for f in subfiles if f.is_file()])
                    print(f"  [{item_type}] {item.name}/ ({file_count} files)")
                except:
                    print(f"  [{item_type}] {item.name}/")

        # Check cache directory after download
        cache_files_after = list(cache_dir.glob("**/*"))
        cache_files_list = [f for f in cache_files_after if f.is_file()]
        print(f"\nCache directory: {len(cache_files_list)} files")
        print("Sample of files (first 20):")
        for f in cache_files_list[:20]:
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(cache_dir)} ({size_mb:.2f} MB)")
            except:
                print(f"  {f.relative_to(cache_dir)}")

        # Look for large model files anywhere in /models/
        print("\nSearching for large files (>10 MB) in /models/:")
        large_files = []
        for item in models_root.glob("**/*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                if size_mb > 10:
                    large_files.append((item, size_mb))

        large_files.sort(key=lambda x: x[1], reverse=True)
        for file_path, size_mb in large_files[:10]:
            print(f"  {file_path.relative_to(models_root)} ({size_mb:.2f} MB)")

        if not large_files:
            print("  (No large files found)")

        # Commit volume to persist changes
        print("\n" + "=" * 70)
        print("Committing volume to persist model...")
        model_volume.commit()
        print("✓ Volume committed")

        return f"SUCCESS: Model downloaded. Cache has {len(cache_files_list)} files. Found {len(large_files)} large files in volume."

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
        from huggingface_hub import snapshot_download
        from alphagenome_research.model import dna_model

        # Find the downloaded model path
        local_model_path = snapshot_download(
            repo_id="google/alphagenome-all-folds",
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=True,  # Only use local files, don't download
        )
        print(f"Found model at: {local_model_path}")

        # Load model from the local path
        model = dna_model.create(
            checkpoint_path=local_model_path,
            organism_settings=None,
            device=None,
        )

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
