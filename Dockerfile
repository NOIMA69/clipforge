FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the whisper model so first request is fast
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

COPY . .

RUN mkdir -p uploads outputs

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
