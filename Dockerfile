# syntax=docker/dockerfile:1
#
# Build from the REPO ROOT, not this directory - the skill markdown lives in a sibling
# app and has to come along:
#
#   docker build -f apps/agentic-system/Dockerfile -t hivek-agentic .
#
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer caches across source edits.
COPY apps/agentic-system/pyproject.toml apps/agentic-system/README.md ./
COPY apps/agentic-system/src/ ./src/
RUN pip install --no-cache-dir .

# The skill registry reads these at runtime. SKILLS_DIR points at them explicitly
# because the default path is derived from the monorepo layout, which does not exist
# inside the image.
COPY apps/server-ai/SKILL/ /app/skills/
ENV SKILLS_DIR=/app/skills

RUN useradd --create-home --uid 1001 hivek && chown -R hivek:hivek /app
USER hivek

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "hivek_agent.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
