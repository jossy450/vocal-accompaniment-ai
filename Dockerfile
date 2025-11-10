FROM python:3.11-slim

# System deps for ffmpeg, fluidsynth, midi rendering
RUN apt-get update && apt-get install -y ffmpeg fluidsynth libasound2-dev libavcodec-extra python3-dev build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "start.py"]
