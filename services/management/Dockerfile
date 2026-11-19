FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/management/constraints.txt /app/constraints.txt
COPY services/management/upstream /app/upstream
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c /app/constraints.txt '/app/upstream[management]' && pip check
COPY contracts /app/contracts
COPY runtime /app/runtime
RUN python -m runtime.generate_proto management \
    && useradd --uid 10001 --create-home app
ENV PYTHONPATH=/app:/app/upstream/src
WORKDIR /app/upstream
USER app
EXPOSE 50054
CMD ["python", "-m", "runtime.serve", "management"]
