"""Marimo notebook for testing AlphaGenome inference API."""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import json

    return json, mo, requests


@app.cell
def _(mo):
    mo.md("""
    # AlphaGenome Inference API Test

    This notebook tests the AlphaGenome inference API deployed on Modal.

    **Setup:** Make sure you have the API running with `pixi run serve-temp` or deployed with `pixi run serve-deploy`
    """)
    return


@app.cell
def _(mo):
    # API URL input - user should paste their Modal URL here
    api_url_input = mo.ui.text(
        value="https://rsmichael--alphagenome-inference-torch-alphagenomese-2f9b47-dev.modal.run",
        label="API URL",
        full_width=True,
    )
    api_url_input
    return (api_url_input,)


@app.cell
def _(api_url_input):
    api_url = api_url_input.value
    return (api_url,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Health Check
    """)
    return


@app.cell
def _(api_url, requests):
    # Test health endpoint
    try:
        _response = requests.get(f"{api_url}/health", timeout=120)
        health_status = _response.json()

        if health_status.get("status") == "healthy":
            print(f"✅ **API is healthy!** {health_status.get('message')}")
        else:
            print(f"⚠️ **API is unhealthy:** {health_status.get('message')}")
    except requests.exceptions.RequestException as e:
        health_status = {"error": str(e)}
        print(f"❌ **Failed to connect to API:** {str(e)}")
    return


@app.cell
def _(api_url_input, requests):
    _response = requests.get(f"{api_url_input.value}/predict-test")
    _response.json()
    return


@app.cell
def _(api_url_input, requests):
    _response = requests.get(f"{api_url_input.value}/debug-volume")
    _response.json()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Test Sequence Prediction

    Let's test the API with a simple DNA sequence.
    """)
    return


@app.cell
def _():
    # 2**20/len("ATCGATCGATCG")
    return


@app.cell
def _(mo):
    # Input controls for prediction
    sequence_input = mo.ui.text_area(
        value="AT" * 2**14,  # 2048 bp
        label="DNA Sequence",
        full_width=True,
    )

    organism_dropdown = mo.ui.dropdown(
        options=["human", "mouse"],
        value="human",
        label="Organism",
    )

    outputs_multiselect = mo.ui.multiselect(
        options=[
            "ATAC",
            "CAGE",
            "DNASE",
            "RNA_SEQ",
            "PROCAP",
            "CHIP_HISTONE",
            "CHIP_TF",
            "SPLICE_SITES",
            "SPLICE_SITE_USAGE",
            "SPLICE_JUNCTIONS",
            "CONTACT_MAPS",
        ],
        value=["ATAC", "RNA_SEQ"],
        label="Output Types",
    )

    mo.vstack([sequence_input, organism_dropdown, outputs_multiselect])
    return organism_dropdown, outputs_multiselect, sequence_input


@app.cell
def _():
    return


@app.cell
def _(mo):
    # Run prediction button
    run_prediction_button = mo.ui.run_button(label="Run Prediction")
    run_prediction_button
    return (run_prediction_button,)


@app.cell
def _(
    api_url,
    mo,
    organism_dropdown,
    outputs_multiselect,
    requests,
    run_prediction_button,
    sequence_input,
):
    # Make prediction when button is clicked
    prediction_result = None

    if run_prediction_button.value:
        _request_data = {
            "sequences": [sequence_input.value],
            "organism": organism_dropdown.value,
            "outputs": list(outputs_multiselect.value),
        }

        try:
            mo.status.spinner(title="Running inference on Modal H100 GPU...")
            _response = requests.post(
                f"{api_url}/predict",
                json=_request_data,
                timeout=300,
            )
            _response.raise_for_status()
            prediction_result = _response.json()
            mo.md("✅ **Prediction completed successfully!**")
        except requests.exceptions.RequestException as e:
            prediction_result = {"error": str(e)}
            mo.md(f"❌ **Prediction failed:** {str(e)}")
    print(prediction_result)
    return (prediction_result,)


@app.cell
def _(prediction_result):
    import numpy as np
    np.array(prediction_result['predictions'][0]['outputs']['RNA_SEQ']['values'][0])
    return


@app.cell
def _(mo, prediction_result):
    # Display prediction results
    if prediction_result and "predictions" in prediction_result:
        _pred = prediction_result["predictions"][0]

        mo.md(
            f"""
            ## 3. Prediction Results

            - **Sequence Index:** {_pred['sequence_index']}
            - **Sequence Length:** {_pred['sequence_length']} bp
            - **Organism:** {prediction_result['organism']}
            - **Number of Outputs:** {len(_pred['outputs'])}
            """
        )
    elif prediction_result and "error" in prediction_result:
        mo.md(f"**Error:** {prediction_result['error']}")
    else:
        mo.md("*No predictions yet. Click 'Run Prediction' above.*")
    return


@app.cell
def _(mo, prediction_result):
    # Display output shapes and metadata
    if prediction_result and "predictions" in prediction_result:
        _pred = prediction_result["predictions"][0]
        _outputs = _pred["outputs"]

        _rows = []
        for output_name, output_data in _outputs.items():
            _rows.append({
                "Output Type": output_name,
                "Shape": str(output_data["shape"]),
                "Resolution": f"{output_data['resolution']} bp",
                "Num Tracks": len(output_data["tracks"]),
            })

        mo.ui.table(_rows, label="Output Summary")
    else:
        mo.md("")
    return


@app.cell
def _(mo, prediction_result):
    # Show track names for each output
    if prediction_result and "predictions" in prediction_result:
        _pred = prediction_result["predictions"][0]
        _outputs = _pred["outputs"]

        _output_selector = mo.ui.dropdown(
            options=list(_outputs.keys()),
            value=list(_outputs.keys())[0] if _outputs else None,
            label="Select Output to View",
        )
        _output_selector
    return


@app.cell
def _(mo, prediction_result):
    # Display selected output details
    if prediction_result and "predictions" in prediction_result and _output_selector.value:
        _pred = prediction_result["predictions"][0]
        _selected_output = _pred["outputs"][_output_selector.value]

        mo.md(
            f"""
            ### {_output_selector.value} Details

            - **Shape:** {_selected_output['shape']}
            - **Resolution:** {_selected_output['resolution']} bp
            - **Tracks:** {len(_selected_output['tracks'])}
            """
        )
    return


@app.cell
def _(mo, prediction_result):
    # Display track names
    if prediction_result and "predictions" in prediction_result and _output_selector.value:
        _pred = prediction_result["predictions"][0]
        _selected_output = _pred["outputs"][_output_selector.value]

        mo.md("**Track Names:**")
        mo.ui.table(
            [{"Track": track} for track in _selected_output["tracks"][:10]],
            label=f"First 10 tracks (of {len(_selected_output['tracks'])})",
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Batch Prediction Test

    Test with multiple sequences in a single request.
    """)
    return


@app.cell
def _(api_url, mo, requests):
    # Test batch prediction
    _batch_sequences = [
        "ATCG" * 512,  # 2048 bp
        "GCTA" * 256,  # 1024 bp (will be padded)
        "TTAA" * 400,  # 1600 bp
    ]

    _batch_request = {
        "sequences": _batch_sequences,
        "organism": "human",
        "outputs": ["ATAC"],
    }

    try:
        _response = requests.post(
            f"{api_url}/predict",
            json=_batch_request,
            timeout=120,
        )
        _response.raise_for_status()
        batch_result = _response.json()

        _batch_summary = []
        for _pred in batch_result["predictions"]:
            _batch_summary.append({
                "Sequence": _pred["sequence_index"],
                "Length": f"{_pred['sequence_length']} bp",
                "Outputs": len(_pred["outputs"]),
            })

        mo.vstack([
            mo.md("✅ **Batch prediction completed!**"),
            mo.ui.table(_batch_summary, label="Batch Results Summary"),
        ])
    except requests.exceptions.RequestException as e:
        batch_result = {"error": str(e)}
        mo.md(f"❌ **Batch prediction failed:** {str(e)}")
    except Exception as e:
        batch_result = None
        mo.md(f"⚠️ **Note:** Run the API first to test batch predictions")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Raw JSON Response

    View the complete API response in JSON format.
    """)
    return


@app.cell
def _(json, mo, prediction_result):
    # Show raw JSON
    if prediction_result:
        mo.ui.code_editor(
            value=json.dumps(prediction_result, indent=2),
            language="json",
            disabled=True,
        )
    else:
        mo.md("*No prediction results to display*")
    return


if __name__ == "__main__":
    app.run()
