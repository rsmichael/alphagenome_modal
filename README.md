# AlphaGenome Modal

Inference microservice and training pipeline for the AlphaGenome model using Modal.

## Overview

This project provides a Modal-based infrastructure for running AlphaGenome model inference and training. Modal allows you to run serverless Python code in the cloud with minimal configuration.

## Prerequisites

### 1. Install Pixi

Pixi is a package manager that handles Python and dependencies. Install it following the instructions at [https://pixi.sh](https://pixi.sh).

For macOS/Linux:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Modal Account and Authentication

You'll need a Modal account to run this application.

#### Create a Modal Account

1. Go to [https://modal.com](https://modal.com)
2. Sign up for a free account
3. Verify your email address

#### Install Modal CLI and Authenticate

After setting up your pixi environment (see below), authenticate with Modal:

```bash
pixi run modal setup
```

This will:
- Open a browser window for authentication
- Create a Modal API token
- Store the token locally in `~/.modal.toml`

You only need to do this once per machine. The token will be used for all subsequent Modal operations.

#### Verify Authentication

To verify your Modal authentication is working:

```bash
pixi run modal token show
```

This should display your current Modal token and workspace information.

## Setup

1. Clone this repository:
```bash
git clone git@github.com:rsmichael/alphagenome_modal.git
cd alphagenome_modal
```

2. Install dependencies using pixi:
```bash
pixi install
```

This will create a virtual environment and install all required dependencies including Modal.

## Running the Application

### Run the Basic Hello World App

To test that everything is working:

```bash
pixi run modal-run
```

This will execute the basic hello world function on Modal's infrastructure.

### Available Commands

- `pixi run modal-run` - Run the Modal app locally (executes functions remotely on Modal)
- `pixi run modal-deploy` - Deploy the app to Modal (creates a persistent deployment)
- `pixi run modal-shell` - Open an interactive shell in the Modal environment

## Project Structure

```
modal_alphagenome/
├── pixi.toml                      # Pixi configuration and dependencies
├── README.md                      # This file
└── modal_alphagenome/             # Main Python package
    ├── __init__.py                # Package initialization
    └── app.py                     # Basic Modal application
```

## Next Steps

This is a basic setup with a hello world function. Future development will include:

- Model inference endpoints
- Training pipeline
- GPU support
- Model storage and versioning
- API endpoints for inference

## Troubleshooting

### Modal Authentication Issues

If you encounter authentication errors:

1. Check if you're authenticated: `pixi run modal token show`
2. Re-authenticate: `pixi run modal setup`
3. Verify your Modal account is active at [https://modal.com](https://modal.com)

### Pixi Issues

If pixi commands fail:

1. Ensure pixi is installed: `pixi --version`
2. Try reinstalling dependencies: `pixi install --force`
3. Check that you're in the project directory

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Pixi Documentation](https://pixi.sh/latest/)
- [AlphaGenome Project](https://github.com/rsmichael/alphagenome_modal)
