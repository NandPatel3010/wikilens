FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/models TOKENIZERS_PARALLELISM=false
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY search ./search
COPY benchmark ./benchmark
RUN python -c "from search.models import Models; Models()"
ENV HF_HUB_OFFLINE=1
USER 10001
CMD ["sh", "-c", "uvicorn search.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
