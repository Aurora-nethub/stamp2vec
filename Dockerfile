FROM python:3.12

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . /app

RUN if [ ! -f pyproject.toml ]; then uv init; fi \
    && uv sync

ENV PATH="/app/.venv/bin:${PATH}"

RUN /bin/bash -lc "source /app/.venv/bin/activate"

CMD ["/bin/bash", "-lc", "python -m scripts.init_model && python -m scripts.init_milvus && ./run.sh"]
