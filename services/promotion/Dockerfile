FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 U2NET_HOME=/cache/u2net POSTER_DATA_DIR=/data/promotion
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/promotion/upstream/requirements-serving.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt grpcio-tools==1.85.1 && pip check
COPY services/promotion/upstream /app/upstream
COPY contracts /app/contracts
COPY runtime /app/runtime
RUN python -m runtime.generate_proto promotion \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /cache /data && chown -R app:app /cache /data
ENV PYTHONPATH=/app:/app/upstream
WORKDIR /app/upstream
USER app
EXPOSE 50052 8081
CMD ["python", "-m", "runtime.serve", "promotion"]
