FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "bmc", "/bin/bash", "-c"]
RUN conda install -c conda-forge manim

ENV PATH="/opt/conda/envs/bmc/bin:$PATH"

COPY . /app

RUN pip install -e .

SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN echo "conda activate bmc" >> ~/.bashrc