FROM jupyter/minimal-notebook:latest

USER ${NB_UID}

RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install --no-cache-dir \
    pandas \
    scipy \
    numpy \
    matplotlib \
    scikit-learn \
    jupyterlab

ARG PYTORCH_URL
RUN pip install --pre torch torchvision torchaudio --index-url ${PYTORCH_URL}