FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY segmentation_pipeline.py test_on_chtc.py ./

# No ENTRYPOINT — the caller specifies the command. Examples:
#   docker run --gpus all -e HF_TOKEN=hf_... remote_sam3 \
#       python segmentation_pipeline.py 591507 output --limit 1
#   docker run --gpus all -e HF_TOKEN=hf_... remote_sam3 \
#       python test_on_chtc.py --taxon-id 591507
# CHTC's HTCondor container universe passes its own command, which would
# otherwise conflict with a fixed ENTRYPOINT.
