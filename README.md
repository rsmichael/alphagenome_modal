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

### 3. HuggingFace Account and AlphaGenome Access

AlphaGenome is a gated model on HuggingFace that requires accepting license terms.

#### Create HuggingFace Account

1. Go to [https://huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Verify your email address

#### Accept AlphaGenome License Terms

**IMPORTANT:** You must accept the license terms before downloading the model.

1. Visit [https://huggingface.co/google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds)
2. Log in to your HuggingFace account
3. Read and accept the license agreement

**License Restrictions:** The AlphaGenome model is for **non-commercial use only**:
- Available to universities, nonprofits, and research institutes
- Cannot be used by commercial entities
- Cannot be used to train derivative models
- Derivatives must follow the same restrictions

#### Create HuggingFace Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "modal-alphagenome")
4. Select "Read" access
5. Click "Generate token"
6. **Copy the token** (it starts with `hf_...`)

#### Configure Modal Secret with HuggingFace Token

After installing dependencies (see Setup section below), create a Modal secret:

```bash
modal secret create huggingface-secret
```

When prompted, enter:
```
HF_TOKEN=hf_your_token_here
```

Replace `hf_your_token_here` with your actual HuggingFace token.

You only need to do this once. The secret will be available to all Modal functions.

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

### Step 1: Test HuggingFace Authentication (CPU Only)

Before downloading the large model files, verify your HuggingFace authentication:

```bash
pixi run test-download
```

This runs a CPU-only test that verifies:
- Your HF_TOKEN is correctly configured
- You have access to the AlphaGenome model
- You've accepted the license terms

**Expected output:** Should show "✓ SUCCESS!" and confirm model access.

**If it fails:** Check that you've:
1. Accepted the license at https://huggingface.co/google/alphagenome-all-folds
2. Created the Modal secret with your HF token
3. Used the correct token format (starts with `hf_...`)

### Step 2: Download AlphaGenome Model (H100 GPU)

Once authentication is verified, download the full model:

```bash
pixi run setup-model
```

This will:
- Download AlphaGenome weights from HuggingFace
- Store them in a persistent Modal volume
- Verify the model loads correctly
- Use an H100 GPU for loading (required for this model)

**Note:** This may take a while depending on model size and takes advantage of Modal's H100 GPU.

### Step 3: Run the Basic Hello World App

To test that basic Modal functionality is working:

```bash
pixi run modal-run
```

This will execute the basic hello world function on Modal's infrastructure.

### Step 4: Run Inference API

Once the model is downloaded, you can start the inference API:

#### Option A: Temporary Endpoint (for testing)

```bash
pixi run serve-temp
```

This creates a temporary web endpoint that runs on Modal's H100 GPU. The endpoint:
- Stays active while the command is running
- Shuts down when you press Ctrl+C
- Provides a URL you can use to make requests
- Good for development and testing

#### Option B: Persistent Deployment (for production)

```bash
pixi run serve-deploy
```

This deploys a persistent web service that:
- Stays running even after you close your terminal
- Gets a permanent URL
- Automatically scales based on traffic
- Good for production use

The API will be available at a URL like `https://rsmichael--alphagenome-inference-fastapi-app.modal.run`

## API Usage

### Available Commands

**Setup:**
- `pixi run test-download` - Test HuggingFace authentication (CPU only, fast)
- `pixi run setup-model` - Download and verify AlphaGenome model (H100 GPU, slow)

**Inference API:**
- `pixi run serve-temp` - Start temporary inference API endpoint (for testing)
- `pixi run serve-deploy` - Deploy persistent inference API (for production)

**Other:**
- `pixi run modal-run` - Run the basic Modal hello world app
- `pixi run modal-deploy` - Deploy the app to Modal (creates a persistent deployment)
- `pixi run modal-shell` - Open an interactive shell in the Modal environment

## API Reference

### Endpoints

The inference API provides two endpoints:

#### GET /health

Check if the API and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "AlphaGenome model loaded and ready for inference"
}
```

#### POST /predict

Run AlphaGenome inference on DNA sequences.

**Request Body:**
```json
{
  "sequences": ["ATCGATCG..."],
  "organism": "human",
  "outputs": ["ATAC", "RNA_SEQ"],
  "tissues": ["UBERON:0001157"]
}
```

**Parameters:**
- `sequences` (required): List of DNA sequences (ATCG characters, max 65,536 bp each)
- `organism` (optional): "human" or "mouse" (default: "human")
- `outputs` (optional): List of output types (default: ["ATAC", "RNA_SEQ"])
- `tissues` (optional): List of ontology terms to filter by tissue type

**Available Output Types:**
- `ATAC` - ATAC-seq chromatin accessibility (1bp resolution)
- `CAGE` - CAGE transcription start sites (1bp resolution)
- `DNASE` - DNase-seq chromatin accessibility (1bp resolution)
- `RNA_SEQ` - RNA-seq gene expression (1bp resolution)
- `PROCAP` - PRO-cap nascent transcription (1bp resolution)
- `CHIP_HISTONE` - Histone ChIP-seq modifications (128bp resolution)
- `CHIP_TF` - Transcription factor ChIP-seq (128bp resolution)
- `SPLICE_SITES` - Splice site classification (1bp resolution)
- `SPLICE_SITE_USAGE` - Splice site usage quantification (1bp resolution)
- `SPLICE_JUNCTIONS` - Junction predictions between splice sites
- `CONTACT_MAPS` - 3D chromatin contact maps (2048bp resolution)

**Response:**
```json
{
  "predictions": [
    {
      "sequence_index": 0,
      "sequence_length": 2048,
      "outputs": {
        "ATAC": {
          "values": [[0.1, 0.2], [0.3, 0.4], ...],
          "shape": [2048, 2],
          "resolution": 1,
          "tracks": ["tissue1_+", "tissue1_-"]
        },
        "RNA_SEQ": {
          "values": [[0.5, 0.6], ...],
          "shape": [2048, 2],
          "resolution": 1,
          "tracks": ["tissue1_+", "tissue1_-"]
        }
      }
    }
  ],
  "organism": "human"
}
```

### Usage Examples

#### cURL

```bash
# Get API health status
curl https://your-modal-url.modal.run/health

# Make a prediction
curl -X POST https://your-modal-url.modal.run/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["ATCGATCGATCGATCG"],
    "organism": "human",
    "outputs": ["ATAC", "RNA_SEQ"]
  }'
```

#### Python

```python
import requests

# API endpoint URL (get this from Modal after deploying)
API_URL = "https://your-modal-url.modal.run"

# Check health
response = requests.get(f"{API_URL}/health")
print(response.json())

# Make a prediction
sequence = "ATCG" * 512  # 2048 bp sequence

response = requests.post(
    f"{API_URL}/predict",
    json={
        "sequences": [sequence],
        "organism": "human",
        "outputs": ["ATAC", "RNA_SEQ"],
        "tissues": ["UBERON:0001157"],  # Optional: colon tissue
    }
)

result = response.json()
predictions = result["predictions"][0]

# Access ATAC predictions
atac_values = predictions["outputs"]["ATAC"]["values"]
print(f"ATAC shape: {predictions['outputs']['ATAC']['shape']}")
print(f"First prediction: {atac_values[0]}")
```

#### Batch Inference

```python
# Process multiple sequences in one request
sequences = [
    "ATCG" * 512,  # Sequence 1: 2048 bp
    "GCTA" * 256,  # Sequence 2: 1024 bp (will be auto-padded)
    "TTAA" * 1024, # Sequence 3: 4096 bp
]

response = requests.post(
    f"{API_URL}/predict",
    json={
        "sequences": sequences,
        "organism": "human",
        "outputs": ["ATAC"],
    }
)

result = response.json()

# Process each sequence's predictions
for pred in result["predictions"]:
    seq_idx = pred["sequence_index"]
    seq_len = pred["sequence_length"]
    atac = pred["outputs"]["ATAC"]
    print(f"Sequence {seq_idx} ({seq_len} bp): {atac['shape']}")
```

### Tissue Filtering with Ontology Terms

You can filter predictions by specific cell or tissue types using ontology terms:

```python
# Predict for specific tissue type
response = requests.post(
    f"{API_URL}/predict",
    json={
        "sequences": [sequence],
        "outputs": ["RNA_SEQ"],
        "tissues": [
            "UBERON:0001157",  # colon
            "UBERON:0002048",  # lung
        ],
    }
)
```

Common ontology terms:
- `UBERON:0001157` - colon
- `UBERON:0002048` - lung
- `UBERON:0000948` - heart
- `UBERON:0002037` - cerebellum
- `UBERON:0002107` - liver

For more ontology terms, see the [Uberon Ontology](http://uberon.github.io/).

## Project Structure

```
alphagenome_modal/
├── pixi.toml                      # Pixi configuration and dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── alphagenome_modal/             # Main Python package
    ├── __init__.py                # Package initialization
    ├── app.py                     # Basic Modal hello world app
    ├── download_test.py           # CPU-only HuggingFace auth test
    ├── model_setup.py             # H100 model download and verification
    ├── inference.py               # Inference API with FastAPI endpoints
    └── alphagenome_research/      # AlphaGenome research code (vendored)
```

## Model Storage

The AlphaGenome model weights are stored in a Modal Volume named `alphagenome-models`. This volume:
- Persists across function calls and deployments
- Is shared between all functions in the app
- Only needs to be populated once (via `pixi run setup-model`)
- Can be mounted read-only for inference to prevent accidental modifications

## Next Steps

The inference API is now ready for DNA sequence analysis. Future development can include:

- Variant effect prediction endpoint
- Genomic interval-based predictions (requires reference genome)
- Longer sequence support with windowing (>65K bp)
- Training pipeline integration
- Result caching and storage
- Model versioning and updates
- Async job processing for very long sequences

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

### HuggingFace Authentication Issues

If `pixi run test-download` fails:

1. **"HF_TOKEN not found"**: Create the Modal secret
   ```bash
   modal secret create huggingface-secret
   # Then enter: HF_TOKEN=hf_your_token_here
   ```

2. **"Cannot access model" or "gated" error**:
   - Go to https://huggingface.co/google/alphagenome-all-folds
   - Log in and accept the license terms
   - Wait a few minutes for permissions to propagate

3. **"Authentication failed"**:
   - Verify your token at https://huggingface.co/settings/tokens
   - Make sure the token has "Read" access
   - Create a new token if needed and update the Modal secret

4. **Check secret exists**:
   ```bash
   modal secret list
   ```
   Should show `huggingface-secret` in the list

### Model Download Issues

If `pixi run setup-model` fails:

1. **"Model directory not found" during verification**:
   - The download may have failed
   - Check Modal logs for error messages
   - Try running `pixi run setup-model` again

2. **Timeout errors**:
   - The model is large and may take time to download
   - The timeout is set to 1 hour, but you can increase it if needed
   - Check your network connection

3. **GPU availability**:
   - H100 GPUs may not be immediately available
   - Modal will queue your request until a GPU is available
   - You can check GPU availability in the Modal dashboard

4. **Out of memory errors**:
   - The H100 has 80GB of memory which should be sufficient
   - If issues persist, check the Modal logs for specific error messages

### Viewing Modal Logs

To see detailed logs from Modal functions:

```bash
modal app logs alphagenome-download-test
modal app logs alphagenome-model-setup
```

## Additional Resources

- [AlphaGenome Model (HuggingFace)](https://huggingface.co/google/alphagenome-all-folds)
- [AlphaGenome Research Code](https://github.com/google-deepmind/alphagenome_research)
- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Pixi Documentation](https://pixi.sh/latest/)
- [HuggingFace Tokens](https://huggingface.co/settings/tokens)
