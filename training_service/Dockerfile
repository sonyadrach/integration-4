FROM python:3.14-slim
WORKDIR /workspace
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY training_service/ /workspace/training_service/
WORKDIR /workspace/training_service
CMD ["uv", "run", "python", "-m", "train"]