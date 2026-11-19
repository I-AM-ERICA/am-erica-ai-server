FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/cache/huggingface TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/policy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r /app/requirements.txt && pip check
COPY services/policy/upstream /app/upstream
COPY contracts /app/contracts
COPY runtime /app/runtime
RUN python -m runtime.generate_proto policy \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /cache /data && chown -R app:app /cache /data
ENV PYTHONPATH=/app:/app/upstream
WORKDIR /app/upstream
USER app
EXPOSE 50051
CMD ["python", "-m", "runtime.serve", "policy"]
