FROM mambaorg/micromamba:1.5.8

# Hugging Face Spaces runs as root, so set user
USER root

# Create working directory
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

# ── Step 1: Install ALL binary/geo packages via conda-forge in ONE layer ──
# This ensures GDAL, rasterio, SQLite, PROJ all come from the same channel
# and are linked against the same shared libraries.
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    "numpy=1.26.*" \
    "xarray=2024.7.*" \
    "gdal=3.9.*" \
    "rasterio=1.4.*" \
    rioxarray \
    geopandas \
    pyproj \
    shapely \
    "stackstac=0.5.*" \
    leafmap \
    localtileserver \
    pandas \
    matplotlib \
    scipy \
    openpyxl \
    && micromamba clean --all --yes

# ── Step 2: Activate conda env for all subsequent RUN commands ──
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# ── Step 3: pip-only packages (no binary geo deps) ──
COPY requirements.txt .
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    solara==1.57.3 \
    solara-enterprise==1.57.3 \
    solara-server==1.57.3 \
    solara-ui==1.57.3 \
    pystac-client \
    planetary-computer \
    omnicloudmask \
    sentinelhub \
    s2cloudless \
    lonboard \
    folium==0.20.0 \
    Authlib==1.6.9 \
    starlette==0.36.3 \
    Flask==3.1.3 \
    && pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# ── Step 4: Copy app files ──
RUN mkdir -p pages
COPY pages/ ./pages/

# ── Step 5: Environment variables ──
ENV PROJ_LIB=/opt/conda/share/proj
ENV GDAL_DATA=/opt/conda/share/gdal

EXPOSE 7860

# Hugging Face Spaces expects port 7860
CMD ["solara", "run", "./pages", "--host=0.0.0.0", "--port=7860"]
