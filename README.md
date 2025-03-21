# DialysisDuETT

**DialysisDuETT** is a customized adaptation of the DuETT (Dual Event Time Transformer) model, tailored specifically for analyzing and predicting clinical data from dialysis patients.

## Environment Configuration

To set up the environment, the following variables must be defined in a `.env` file:

### `PYTORCH_URL`

Provide the URL for the appropriate PyTorch version, which can be found on the [official PyTorch website](https://pytorch.org/get-started/locally/). Ensure compatibility with your system's CUDA drivers.

### `VOLUME_PATH`

Specify the path to the host machine’s volume that will be mounted into the container or environment.

### `JUPYTER_TOKEN`

Define the access token required to authenticate with the Jupyter Notebook server.