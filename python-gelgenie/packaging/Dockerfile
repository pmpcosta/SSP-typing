# Standalone SSP-typing runtime image (ONNX-only — no torch, no gelgenie).
#
# Build from python-gelgenie/:
#   docker build -f packaging/Dockerfile -t ssp-typing:latest .
#
# Run (see packaging/docker-compose.yml for the volume setup):
#   docker run --rm -p 127.0.0.1:8765:8765 \
#     -v ssptyping-models:/home/ssp/.cache/ssptyping \
#     -v ssptyping-data:/home/ssp/.local/share/ssptyping \
#     ssp-typing:latest
#
# The default model (~57 MB, Apache-2.0) is fetched automatically on first
# `serve` and cached in the models volume. Tray sheets are uploaded through
# the web UI; nothing vendor-specific is baked into this image.

FROM python:3.12-slim

# runtime deps only — mirrors setup.py extras_require['ssp'] + ['webapp']
RUN pip install --no-cache-dir \
        "numpy>=1.21" \
        "scikit-image>=0.19" \
        "imageio>=2.19" \
        "pillow>=9.1" \
        "onnxruntime>=1.16" \
        "rich-click>=1.6" \
        "openpyxl>=3.1" \
        "fastapi>=0.110" \
        "uvicorn>=0.29" \
        "python-multipart>=0.0.9"

WORKDIR /app

# the ssp package runs straight off PYTHONPATH (python -m ssp.cli);
# installing via setup.py would drag in the gelgenie/torch training stack
COPY ssp/ /app/ssp/
ENV PYTHONPATH=/app

RUN useradd -m -u 1000 ssp
USER ssp

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8765/api/sessions', timeout=4)" || exit 1

ENTRYPOINT ["python", "-m", "ssp.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
