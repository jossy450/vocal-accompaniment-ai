# Use PyTorch image with CUDA (even if you don’t use GPU, it has torch pre-installed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install audio and MIDI dependencies
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fluidsynth \
    libasound2-dev \
    libavcodec-extra \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python dependencies
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "start.py"]
