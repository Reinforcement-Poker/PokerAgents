FROM python:3.10

RUN apt update && apt install -y git

# Install project dependecies
COPY ./pyproject.toml /
COPY README.md /
RUN pip install .

# Jupyter kernel
RUN python3 -m pip install ipykernel
RUN python3 -m ipykernel install