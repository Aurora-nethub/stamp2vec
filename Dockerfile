# Dockerfile
FROM python:3.12

RUN apt-get update\
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

WORKDIR /app

RUN python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && python -m pip config set global.trusted-host mirrors.aliyun.com \
    && python -m pip config set global.timeout 120 \
    && python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
COPY requirements*.txt ./

RUN if [ -f "pyproject.toml" ]; then \
        UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" \
        uv sync; \
    elif [ -f "requirements.txt" ]; then \
        uv pip install \
        --no-cache-dir \
        -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com; \
    else \
        echo "No dependency files found, skipping dependency installation"; \
    fi

COPY . /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chmod +x /app/run.sh && \
    chmod +x /app/scripts/*.py 2>/dev/null || true

RUN python -m scripts.init_model

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["/bin/bash", "-c", "./run.sh"]