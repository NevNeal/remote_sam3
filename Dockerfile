FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY segmentation_pipeline.py .

# Mount your output directory to /app/<output_folder> at runtime.
# Pass HF_TOKEN as an env var to access gated models:
#   docker run --gpus all -e HF_TOKEN=hf_... -v /host/output:/app/output ...

ENTRYPOINT ["python", "segmentation_pipeline.py"]
