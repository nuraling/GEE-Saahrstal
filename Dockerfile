FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-depth.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-depth.txt
# Bake the Depth Anything weights into the image: no download on cold start,
# and no dependency on huggingface.co being reachable from Cloud Run.
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')"
COPY . .
ENV PYTHONUNBUFFERED=1 PORT=8080 MPLBACKEND=Agg
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "3600", "app:app"]
