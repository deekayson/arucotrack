# ArUcoTrack, CPU image: Python 3.12, ffmpeg from Debian, the uv-locked environment. No CUDA. Decoding is in
# software (the `auto` decoder finds no GPU here). For NVDEC decoding see docs/ENVIRONMENT.md.
#
#   docker build -t arucotrack .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data -w /data arucotrack \
#       run input/configs/tilt.yaml --data-root input --out out --work work
#
# Base image pinned by digest (python:3.12-slim, Python 3.12.13).
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# ffmpeg / ffprobe decode the video. The headless OpenCV wheel needs no further system library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv pinned by the digest of its multi-platform image index
COPY --from=ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1
# Dependencies first (cached layer), then the package itself, both from uv.lock exactly.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH
RUN useradd --create-home --uid 1000 arucotrack
USER arucotrack
ENTRYPOINT ["arucotrack"]
CMD ["--help"]
