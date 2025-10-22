FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    safetensors \
    basicsr \
    realesrgan \
    gfpgan \
    Pillow \
    xformers \
    scipy \
    opencv-python

COPY inference.py /workspace/inference.py
RUN chmod +x /workspace/inference.py
