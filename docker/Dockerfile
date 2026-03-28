FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY run_demo.py .

VOLUME ["/app/data"]
VOLUME ["/app/outputs"]

ENV ENVIRONMENT=docker

ENTRYPOINT ["python3", "run_demo.py", "--env", "docker"]