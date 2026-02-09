"""Basic Modal application for AlphaGenome."""

import modal

# Create a Modal app
app = modal.App("alphagenome-hello")


@app.function()
def hello(name: str = "World") -> str:
    """A simple hello world function that runs on Modal.

    Args:
        name: The name to greet (default: "World")

    Returns:
        A greeting message
    """
    return f"Hello, {name}! This is AlphaGenome running on Modal."


@app.local_entrypoint()
def main():
    """Local entrypoint that calls the hello function."""
    result = hello.remote("AlphaGenome")
    print(result)
