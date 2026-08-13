# Multi-stage build, run from the repo root:
#   docker build -f web/backend/Dockerfile -t alpha-engine-dash .
# (fly.toml points `build.dockerfile`/`build.context` at this same repo
# root, since the image needs both `src/`+`config/` and `web/`.)

FROM node:20-slim AS frontend-build
WORKDIR /app/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

# alpha_engine's own dependencies (pandas/numpy/matplotlib) need these to
# build/run on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY web/backend/ ./web/backend/
RUN pip install --no-cache-dir ".[web]"

# Built frontend lands at web/backend/static -- see STATIC_DIR in
# web/backend/app/main.py, which serves it (with SPA-route fallback to
# index.html) if and only if this directory exists.
COPY --from=frontend-build /app/web/frontend/dist ./web/backend/static

WORKDIR /app/web/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips=*: Fly terminates TLS at its edge
# and forwards plain HTTP internally, and uvicorn only trusts the
# X-Forwarded-Proto header (which says the original request was https)
# from peers it's told to trust -- without this, request.url_for() builds
# "http://..." redirect URIs (e.g. the GitHub OAuth callback), which
# GitHub then rejects since the registered callback is https. "*" is safe
# here since Fly's internal network is already the trust boundary -- this
# process only ever receives traffic from fly-proxy, never directly from
# the internet.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
