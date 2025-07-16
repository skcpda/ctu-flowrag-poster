FROM python:3.11-slim

WORKDIR /app

COPY . /app

ARG EXTRAS=0
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$EXTRAS" = "1" ]; then pip install --no-cache-dir -r requirements-extra.txt; fi

CMD ["python", "scripts/run_full_pipeline.py", "--help"] 