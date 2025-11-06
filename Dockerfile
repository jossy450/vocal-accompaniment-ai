FROM python:3.11-slim

# system deps: ffmpeg for pydub, fluidsynth for pretty_midi rendering
RUN apt-get update && apt-get install -y ffmpeg fluidsynth && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# run via python so PORT is always read correctly
CMD ["python", "start.py"]
