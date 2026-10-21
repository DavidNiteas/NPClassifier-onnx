# Build from the repository root:
#   docker build -f tools/convert/Dockerfile -t npclassifier-convert .
#
# Run (mount the official HDF5 files and an output directory; no weights are
# baked into the image):
#   docker run --rm \
#     -v /path/to/official/hdf5:/data:ro \
#     -v /path/to/output:/out \
#     npclassifier-convert \
#     --hdf5-dir /data --index /data/index_v1.json --output-dir /out --verify
FROM python:3.11-slim

WORKDIR /app
COPY tools/convert/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY tools/convert /app/tools/convert

ENTRYPOINT ["python", "/app/tools/convert/convert_npc.py"]
