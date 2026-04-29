"""CPU-only test app to verify HuggingFace authentication and model access."""

import modal

app = modal.App("alphagenome-download-test")

# Lightweight image for testing authentication
image = modal.Image.debian_slim().pip_install("huggingface-hub")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300,
)
def test_authentication():
    """Test HuggingFace authentication and verify access to AlphaGenome model.

    This function runs on CPU only and tests:
    1. HF_TOKEN is correctly configured in Modal secrets
    2. Token is valid and can authenticate
    3. User has accepted the AlphaGenome license terms
    4. Model metadata can be accessed

    Returns:
        Dict with authentication status and model info
    """
    import os
    from huggingface_hub import login, model_info, HfApi

    # Check if token exists
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {
            "success": False,
            "error": "HF_TOKEN not found in environment. Make sure Modal secret is created.",
        }

    print(f"Found HF_TOKEN: {token[:10]}...")

    # Test authentication
    try:
        login(token=token)
        print("✓ HuggingFace authentication successful")
    except Exception as e:
        return {
            "success": False,
            "error": f"Authentication failed: {str(e)}",
        }

    # Test model access
    model_id = "google/alphagenome-all-folds"
    try:
        info = model_info(model_id, token=token)
        print(f"✓ Successfully accessed model: {model_id}")
        print(f"  Model card: {info.cardData}")
        print(f"  Last modified: {info.lastModified}")

        # Check if gated
        if hasattr(info, 'gated') and info.gated:
            print("  ⚠ Model is gated - terms must be accepted at:")
            print(f"    https://huggingface.co/{model_id}")

        return {
            "success": True,
            "model_id": model_id,
            "gated": getattr(info, 'gated', False),
            "last_modified": str(info.lastModified),
            "message": "Authentication successful! Ready to download model.",
        }

    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "access" in error_msg.lower():
            return {
                "success": False,
                "error": (
                    f"Cannot access model. You may need to:\n"
                    f"1. Go to https://huggingface.co/{model_id}\n"
                    f"2. Log in with your HuggingFace account\n"
                    f"3. Accept the license terms\n"
                    f"Original error: {error_msg}"
                ),
            }
        else:
            return {
                "success": False,
                "error": f"Failed to access model: {error_msg}",
            }


@app.local_entrypoint()
def main():
    """Run authentication test and print results."""
    print("=" * 60)
    print("Testing HuggingFace Authentication for AlphaGenome")
    print("=" * 60)

    result = test_authentication.remote()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if result["success"]:
        print("✓ SUCCESS!")
        print(f"\n{result['message']}")
        if result.get("gated"):
            print("\nNote: Model is gated. Ensure you've accepted the terms.")
    else:
        print("✗ FAILED!")
        print(f"\nError: {result['error']}")
        print("\nPlease fix the issue and try again.")

    print("=" * 60)
