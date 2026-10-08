# MS-PO training / evaluation image (CUDA 12.4, PyTorch 2.5.1).
#   docker build -t ms-po .
#   docker compose run --rm ms-po bash scripts/run_pipeline.sh
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    MSPO_ROOT=/workspace/ms-po

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ms-po
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
CMD ["bash"]
