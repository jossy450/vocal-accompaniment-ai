FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY . /app
RMN pip install --no-cache-dir q requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "-port", "7860"]
