###############################################################################
#  iMAD Change Detection — Unified Solara App
#  Modes:
#    1. Single Scene  with    CEA
#    2. Single Scene  without CEA
#    3. Multi-Scene   with    CEA
#    4. Multi-Scene   without CEA
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library & third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import base64
import datetime
import io
import math
import os
import shutil
import tempfile
import threading
import uuid
import warnings
import zipfile
from collections import defaultdict

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import requests
import rioxarray as rxr
import stackstac
import xarray as xr
from omnicloudmask import predict_from_array
from PIL import Image as PILImage
from rasterio.features import shapes
from scipy import ndimage
from scipy.linalg import eigh
from scipy.stats import chi2
from sentinelhub import SHConfig
from shapely.geometry import mapping, shape
from starlette.responses import FileResponse, Response
from starlette.routing import Route

import solara
import solara.lab
import solara.server.starlette as _solara_starlette
from solara_enterprise import auth

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Significant no-data areas detected")

# ─────────────────────────────────────────────────────────────────────────────
#  Starlette download route (registered once at module load)
# ─────────────────────────────────────────────────────────────────────────────

_download_registry: dict = {}


async def _zip_download_handler(request):
    token = request.path_params["token"]
    entry = _download_registry.get(token)
    if not entry or not os.path.exists(entry["path"]):
        return Response("File not found or expired.", status_code=404)
    return FileResponse(path=entry["path"], media_type="application/zip",
                        filename=entry["filename"])


_solara_starlette.app.routes.insert(
    0, Route("/download/{token}", endpoint=_zip_download_handler)
)


def _register_zip(zip_path: str, filename: str) -> str:
    token = str(uuid.uuid4())
    _download_registry[token] = {"path": zip_path, "filename": filename}
    return f"/download/{token}"


def _cleanup_zip(token: str):
    entry = _download_registry.pop(token, None)
    if entry and os.path.exists(entry["path"]):
        try:
            os.unlink(entry["path"])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def trunc(values, dec=3):
    return np.trunc(values * 10**dec) / (10**dec)


def _safe_login_url() -> str:
    try:
        return auth.get_login_url()
    except Exception:
        return "/_solara/auth/login"


def _safe_logout_url() -> str:
    try:
        return auth.get_logout_url()
    except Exception:
        return "/_solara/auth/logout"


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel extractor — Single Scene (requires full coverage)
# ─────────────────────────────────────────────────────────────────────────────

def sentinel_extractor_single(roi, collection, t1a, t1b,
                               stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
                               cloud_percent=5):
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
             "B8A", "B09", "B10", "B11", "B12", "SCL"]
    roi_4326 = roi.to_crs("EPSG:4326")
    catalog = pystac_client.Client.open(url=stac_url,
                                         modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[collection],
        intersects=mapping(roi_4326.union_all()),
        datetime=f"{t1a}/{t1b}",
        query={"eo:cloud_cover": {"lt": cloud_percent}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )
    items = search.item_collection()
    full_cover = [i for i in items if shape(i.geometry).contains(roi_4326.union_all())]
    if not full_cover:
        raise ValueError("No images covering full ROI in the given date range.")
    item = full_cover[0]
    date = item.datetime.date()
    cloud_cover = item.properties.get("eo:cloud_cover", "?")

    xmin, ymin, xmax, ymax = roi_4326.total_bounds
    longitude = (xmin + xmax) / 2
    zone = math.floor((longitude + 180) / 6) + 1
    roi_reproj = roi.to_crs(f"EPSG:78{zone}")

    stacked = (
        stackstac.stack(items=[item], assets=bands,
                        epsg=int(f"78{zone}"), chunksize=1024, resolution=10)
        .isel(time=0)
    )
    stacked_clipped = stacked.rio.clip(roi_reproj.geometry, roi_reproj.crs)
    stacked_scaled = (stacked_clipped * 0.0001).rio.write_crs(f"EPSG:78{zone}")
    return (str(date), item, stacked, stacked_scaled)


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel extractor — Multi-Scene (median composite, partial OK)
# ─────────────────────────────────────────────────────────────────────────────

def sentinel_extractor_multi(roi, collection, t1a, t1b,
                              stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
                              cloud_percent=5):
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
             "B8A", "B09", "B10", "B11", "B12", "SCL"]
    roi_4326 = roi.to_crs("EPSG:4326")
    catalog = pystac_client.Client.open(url=stac_url,
                                         modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[collection],
        intersects=mapping(roi_4326.union_all()),
        datetime=f"{t1a}/{t1b}",
        query={"eo:cloud_cover": {"lt": cloud_percent}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )
    items = search.item_collection()
    by_date = defaultdict(list)
    for i in items:
        by_date[i.properties["datetime"][:10]].append(i)
    best_date, best_items = max(by_date.items(), key=lambda x: len(x[1]))

    xmin, ymin, xmax, ymax = roi_4326.total_bounds
    longitude = (xmin + xmax) / 2
    zone = math.floor((longitude + 180) / 6) + 1
    roi_reproj = roi.to_crs(f"EPSG:78{zone}")

    stacked = (
        stackstac.stack(items=best_items, assets=bands,
                        epsg=int(f"78{zone}"), chunksize=1024, resolution=10)
        .rio.write_crs(f"EPSG:78{zone}")
    )
    stacked = stacked.median(dim="time")
    stacked_clipped = stacked.rio.clip(roi_reproj.geometry, roi_reproj.crs)
    stacked_scaled = stacked_clipped * 0.0001
    return (best_date, best_items, stacked, stacked_scaled)


# ─────────────────────────────────────────────────────────────────────────────
#  Cloud masking
# ─────────────────────────────────────────────────────────────────────────────

def omnicloudmask(da):
    da_values = da.sel(band=["B04", "B03", "B08"]).values.astype("float32")
    da_values = np.nan_to_num(da_values, nan=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Patch size too large")
        mask = predict_from_array(da_values)
    mask_da = xr.DataArray(
        data=mask[0], dims=["y", "x"],
        coords={"y": da.y, "x": da.x}
    ).rio.write_crs(da.rio.crs)
    return mask_da.isin([1, 2, 3]).astype("uint8")


def scl_masking(da):
    scl_classes = [3, 8, 9, 10]
    scl_band = da.sel(band="SCL") * 10000
    return scl_band.isin(scl_classes).astype("uint8")


# ─────────────────────────────────────────────────────────────────────────────
#  iMAD core
# ─────────────────────────────────────────────────────────────────────────────

def covarw(X, weights=None):
    if weights is None:
        weights = np.ones(X.shape[0])
    w_sum = weights.sum()
    if w_sum < 1e-12:
        raise RuntimeError("Weight sum is effectively zero.")
    w = weights / w_sum
    mu = (w[:, None] * X).sum(axis=0)
    Xc = X - mu
    n = X.shape[0]
    Xcw = Xc * np.sqrt(weights[:, None])
    cov_ = (Xcw.T @ Xcw) * n / w_sum
    return Xc, cov_


def corr(cov):
    sigma_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    c = sigma_inv @ cov @ sigma_inv
    return [[trunc(v) for v in row] for row in c]


def geneiv(C, B):
    lambdas, vecs = eigh(C, B)
    return lambdas, vecs


def chi2cdf(Z, df):
    return chi2.cdf(Z, df)


def imad1(X, weights, N):
    Xc, cov = covarw(X, weights)
    s11 = cov[:N, :N]; s22 = cov[N:, N:]
    s12 = cov[:N, N:]; s21 = cov[N:, :N]
    c1 = np.linalg.solve(s22, s21).T @ s21
    c2 = np.linalg.solve(s11, s12).T @ s12
    lambdas1, A = geneiv(c1, s11)
    _,        B = geneiv(c2, s22)
    lambdas1 = np.clip(lambdas1, 0.0, 1.0)
    rhos = np.sqrt(lambdas1)
    idx = np.argsort(rhos)[::-1]
    rhos = rhos[idx]; A = A[:, idx]; B = B[:, idx]
    tmp = np.diag(1.0 / np.sqrt(np.diag(s11)))
    s = (tmp @ s11 @ A).sum(axis=0)
    A = A * np.sign(s)
    tmp2 = np.diag(np.sign(np.diag(A.T @ s12 @ B)))
    B = B @ tmp2
    U = Xc[:, :N] @ A; V = Xc[:, N:] @ B
    iMAD = U - V
    sigma2 = 2.0 * (1.0 - rhos)
    sigma2 = np.where(sigma2 < 1e-12, 1e-12, sigma2)
    Z = (iMAD**2 / sigma2).sum(axis=1)
    return iMAD, Z, rhos, A, B


def run_imad(image1, image2, mask=None, scale=20, maxiter=100, tol=1e-4):
    def _to_array(img):
        if isinstance(img, xr.DataArray):
            arr = img.values
            if arr.ndim == 3:
                arr = np.moveaxis(arr, 0, -1)
            return arr
        return np.array(img)

    arr1 = _to_array(image1).astype(np.float64)
    arr2 = _to_array(image2).astype(np.float64)
    H, W = arr1.shape[:2]
    N = arr1.shape[2] if arr1.ndim == 3 else 1
    X1 = arr1.reshape(-1, N); X2 = arr2.reshape(-1, N)
    if mask is not None:
        valid = mask.ravel().astype(bool)
    else:
        finite1 = np.all(np.isfinite(X1), axis=1) & np.all(X1 > 0, axis=1)
        finite2 = np.all(np.isfinite(X2), axis=1) & np.all(X2 > 0, axis=1)
        valid = finite1 & finite2
    n_valid = valid.sum()
    print(f"  Valid pixels: {n_valid:,} of {H*W:,} ({100*n_valid/(H*W):.1f}%)")
    if n_valid == 0:
        raise RuntimeError("No valid pixels found.")
    X = np.hstack([X1[valid], X2[valid]])
    weights = np.ones(n_valid)
    allrhos = [np.ones(N)]
    Z = np.zeros(n_valid)
    iMAD_flat = np.zeros((n_valid, N))
    for iteration in range(1, maxiter + 1):
        iMAD_flat, Z, rhos, A, B = imad1(X, weights, N)
        allrhos.append(rhos.copy())
        delta = np.max(np.abs(rhos - allrhos[-2]))
        print(f"  iter {iteration:3d}  rhos={np.round(rhos,5)}  Δ={delta:.2e}")
        if delta < tol:
            print(f"  Converged after {iteration} iterations.")
            break
        weights = np.maximum(1.0 - chi2cdf(Z, N), 1e-6)
    else:
        print(f"  Reached maximum iterations ({maxiter}) without convergence.")
    iMAD_out = np.full((H*W, N), np.nan)
    Z_out = np.full(H*W, np.nan)
    iMAD_out[valid] = iMAD_flat
    Z_out[valid] = Z
    return {"iMAD": iMAD_out.reshape(H, W, N), "Z": Z_out.reshape(H, W),
            "rhos": rhos, "allrhos": allrhos, "niter": iteration}


def run_imad_da(da1, da2, **kwargs):
    result = run_imad(da1, da2, **kwargs)
    N = result["iMAD"].shape[-1]
    y_clean = np.asarray(da1.y.values, dtype=np.float64).ravel()
    x_clean = np.asarray(da1.x.values, dtype=np.float64).ravel()
    crs = da1.rio.crs
    band_names = [f"iMAD{i+1}" for i in range(N)] + ["Z"]
    data_arrays = []
    for i in range(N):
        da = xr.DataArray(result["iMAD"][..., i].astype(np.float64),
                          dims=("y", "x"), coords={"y": y_clean, "x": x_clean},
                          name=band_names[i], attrs={})
        data_arrays.append(da)
    da_z = xr.DataArray(result["Z"].astype(np.float64),
                        dims=("y", "x"), coords={"y": ("y", y_clean), "x": ("x", x_clean)},
                        name="Z", attrs={})
    data_arrays.append(da_z)
    result_da = xr.concat(data_arrays, dim=xr.IndexVariable("band", band_names),
                          coords="minimal", compat="override", join="override")
    if crs is not None:
        result_da = result_da.rio.write_crs(crs)
    result_da.name = "iMAD"
    result_da.attrs.update({"description": "iMAD variates and chi-square change statistic",
                             "rhos": str(result["rhos"].tolist()), "niter": result["niter"]})
    return result_da


# ─────────────────────────────────────────────────────────────────────────────
#  Binary cleanup
# ─────────────────────────────────────────────────────────────────────────────

def clean_binary_xarray(da: xr.DataArray, min_size, closing_then_opening=True) -> xr.DataArray:
    structure = ndimage.generate_binary_structure(2, 1)
    binary = da.values.astype(bool)
    if closing_then_opening:
        binary = ndimage.binary_closing(binary, structure=structure, iterations=1)
    binary = ndimage.binary_opening(binary, structure=structure, iterations=1)
    if min_size > 1:
        labeled, num = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(num+1))
        mask = sizes < min_size
        binary[labeled * mask[labeled]] = False
    return da.copy(data=binary.astype(da.dtype))


# ─────────────────────────────────────────────────────────────────────────────
#  Change detection (shared logic returns gdf_raw + optional gdf_clipped)
# ─────────────────────────────────────────────────────────────────────────────

def run_change_detection(ds1, ds2, cea=None):
    """
    Run iMAD and return (result_da, iMAD_1, gdf_raw, gdf_clipped | None).
    If cea is a GeoDataFrame, results are clipped to it; otherwise gdf_clipped is None.
    """
    result_da = run_imad_da(ds1, ds2, maxiter=100)
    iMAD_1 = result_da.isel(band=0)
    mean = float(iMAD_1.mean())
    stdev = float(iMAD_1.std())
    threshold = mean - 3.5 * stdev

    y_vals = result_da.y.values
    x_vals = result_da.x.values
    ones = np.where(iMAD_1.values < threshold, 1, 0)
    ones_da = xr.DataArray(data=ones, dims=["y", "x"],
                           coords={"y": y_vals, "x": x_vals}
                           ).rio.write_crs(result_da.rio.crs)

    cleaned = clean_binary_xarray(ones_da, min_size=5)
    cleaned_arr = np.where(cleaned.values == 0, np.nan, 1)
    cleaned_da = xr.DataArray(data=cleaned_arr, dims=["y", "x"],
                               coords={"y": y_vals, "x": x_vals})

    mask_arr = cleaned_da.notnull().astype(np.uint8).values
    transform = ones_da.rio.transform()
    shapes_gen = shapes(mask_arr, transform=transform, connectivity=8)
    geoms = [{"geometry": g, "properties": {"value": v}}
             for g, v in shapes_gen if v == 1]

    gdf_raw = gpd.GeoDataFrame.from_features(geoms, crs=ones_da.rio.crs)
    gdf_raw["area_ha"] = gdf_raw.geometry.area / 10000
    gdf_raw = gdf_raw[gdf_raw["area_ha"] >= 0.2].copy()

    gdf_clipped = None
    if cea is not None:
        gdf_clipped = gdf_raw.clip(cea)
        gdf_clipped["area_ha"] = gdf_clipped.geometry.area / 10000
        gdf_clipped = gdf_clipped[gdf_clipped["area_ha"] >= 0.2][["geometry", "area_ha"]].copy()

    return result_da, iMAD_1, gdf_raw, gdf_clipped


# ─────────────────────────────────────────────────────────────────────────────
#  Folium map builder
# ─────────────────────────────────────────────────────────────────────────────

def da_to_image_overlay(da: xr.DataArray, band_list: list, vmin=None, vmax=None):
    subset = da.sel(band=band_list).compute()
    if vmin is None or vmax is None:
        arr_raw = subset.values.astype(np.float32)
        valid = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]
        vmin = float(np.percentile(valid, 2)) if vmin is None else vmin
        vmax = float(np.percentile(valid, 98)) if vmax is None else vmax
    subset_4326 = subset.rio.reproject("EPSG:4326")
    arr = subset_4326.values.astype(np.float32)
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = np.moveaxis(arr, 0, -1)
    alpha = np.any(arr > 0, axis=-1).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha])
    img = PILImage.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    west, south, east, north = subset_4326.rio.bounds()
    return f"data:image/png;base64,{b64}", [[south, west], [north, east]]


def build_folium_map(im1, im2, roi_gdf, gdf_raw, gdf_clipped=None):
    bounds_4326 = im1.rio.transform_bounds("EPSG:4326")
    west, south, east, north = bounds_4326
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    img1_b64, fb1 = da_to_image_overlay(im1, ["B04", "B03", "B02"])
    img2_b64, fb2 = da_to_image_overlay(im2, ["B04", "B03", "B02"])

    folium.raster_layers.ImageOverlay(image=img1_b64, bounds=fb1, opacity=1.0,
                                       name="Beginning Image", interactive=False).add_to(m)
    folium.raster_layers.ImageOverlay(image=img2_b64, bounds=fb2, opacity=1.0,
                                       name="End Image", interactive=False).add_to(m)

    folium.GeoJson(roi_gdf.to_crs("EPSG:4326").__geo_interface__, name="Project Boundary",
                   style_function=lambda x: {"color": "#FFD700", "fillColor": "none",
                                             "fillOpacity": 0, "weight": 2}).add_to(m)

    folium.GeoJson(gdf_raw.to_crs("EPSG:4326").__geo_interface__,
                   name="Disturbances (all)",
                   style_function=lambda x: {"color": "#00FF04", "fillColor": "#00FF04",
                                             "fillOpacity": 1, "weight": 1}).add_to(m)

    if gdf_clipped is not None:
        folium.GeoJson(gdf_clipped.to_crs("EPSG:4326").__geo_interface__,
                       name="Intersected Disturbances (CEA)",
                       style_function=lambda x: {"color": "#FF0000", "fillColor": "#FF0000",
                                                 "fillOpacity": 1, "weight": 1},
                       tooltip=folium.GeoJsonTooltip(fields=["area_ha"],
                                                      aliases=["Area (ha)"])).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  ZIP builder — handles both single & multi, with & without CEA
# ─────────────────────────────────────────────────────────────────────────────

def build_zip_file(im1, im2, im1_metadata, im2_metadata, imad,
                   gdf_raw, project_code, project_id, im1_date, im2_date,
                   gdf_clipped=None, multi_scene=False):
    """
    multi_scene=True  → im1_metadata / im2_metadata are lists of STAC items.
    multi_scene=False → they are single STAC items.
    gdf_clipped       → include intersected CEA shapefile when provided.
    """
    def _date_label(d):
        parts = str(d).split("-")
        return "".join(parts)

    im1_label = _date_label(im1_date)
    im2_label = _date_label(im2_date)

    with tempfile.TemporaryDirectory() as tmpdir:
        im1.rio.to_raster(
            os.path.join(tmpdir, f"{project_id}_{project_code}_BeginningImage_{im1_label}_Clipped.tif"),
            driver="GTiff")
        im2.rio.to_raster(
            os.path.join(tmpdir, f"{project_id}_{project_code}_EndImage_{im2_label}_Clipped.tif"),
            driver="GTiff")

        # Metadata XML — single vs multi
        def _save_meta(meta_items, label):
            if not multi_scene:
                meta_items = [meta_items]
            for item in meta_items:
                try:
                    url = item.assets["product-metadata"].href
                    tile_id = item.properties.get("s2:mgrs_tile", "unknown")
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    fname = f"{project_id}_{project_code}_{label}_MTD_MSIL2A_T{tile_id}.xml"
                    with open(os.path.join(tmpdir, fname), "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    print(f"Warning: could not fetch {label} metadata XML: {e}")

        _save_meta(im1_metadata, "BeginningImage")
        _save_meta(im2_metadata, "EndImage")

        imad.rio.to_raster(
            os.path.join(tmpdir, f"{project_id}_{project_code}_imad1.tif"), driver="GTiff")

        gdf_raw.to_file(
            os.path.join(tmpdir, f"{project_id}_{project_code}_PreClipCEA_Disturbances.shp"),
            driver="ESRI Shapefile")

        if gdf_clipped is not None:
            gdf_clipped.to_file(
                os.path.join(tmpdir, f"{project_id}_{project_code}_Intersected_Disturbances.shp"),
                driver="ESRI Shapefile")

        zip_tmp = tempfile.NamedTemporaryFile(
            suffix=f"_{project_id}_{project_code}_Change_Detection.zip",
            prefix="imad_dl_", delete=False)
        try:
            with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(tmpdir):
                    zf.write(os.path.join(tmpdir, fname), arcname=fname)
        finally:
            zip_tmp.close()
        return zip_tmp.name


# ─────────────────────────────────────────────────────────────────────────────
#  CEA file loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cea_from_bytes(file_bytes: bytes, file_name: str) -> gpd.GeoDataFrame:
    name_lower = file_name.lower()
    if name_lower.endswith(".gpkg"):
        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False, prefix="cea_")
        try:
            tmp.write(file_bytes); tmp.flush(); os.fsync(tmp.fileno()); tmp.close()
            return gpd.read_file(tmp.name)
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass
    if name_lower.endswith(".zip"):
        extract_dir = tempfile.mkdtemp(prefix="cea_zip_")
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                zf.extractall(extract_dir)
            for root, dirs, _ in os.walk(extract_dir):
                for d in dirs:
                    if d.lower().endswith(".gdb"):
                        return gpd.read_file(os.path.join(root, d))
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".shp"):
                        return gpd.read_file(os.path.join(root, f))
            raise ValueError("ZIP has no recognised spatial file (.shp or .gdb).")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    raise ValueError(f"Unsupported CEA format: '{file_name}'.")


# ─────────────────────────────────────────────────────────────────────────────
#  Download ZIP button component
# ─────────────────────────────────────────────────────────────────────────────

@solara.component
def DownloadZipButton(im1, im2, im1_metadata, im2_metadata, imad,
                      gdf_raw, project_id, project_code, im1_date, im2_date,
                      gdf_clipped=None, multi_scene=False):
    download_url = solara.use_reactive(None)
    is_building  = solara.use_reactive(True)
    build_error  = solara.use_reactive(None)

    def build_in_background():
        try:
            zip_path = build_zip_file(
                im1, im2, im1_metadata, im2_metadata, imad,
                gdf_raw, project_code, project_id, im1_date, im2_date,
                gdf_clipped=gdf_clipped, multi_scene=multi_scene)
            filename = f"{project_id}_{project_code}_Change_Detection.zip"
            download_url.set(_register_zip(zip_path, filename))
        except Exception as e:
            build_error.set(str(e))
        finally:
            is_building.set(False)

    solara.use_effect(
        lambda: threading.Thread(target=build_in_background, daemon=True).start(), [])

    if is_building.value:
        with solara.Row(style={"margin-top": "16px", "align-items": "center"}):
            solara.Text("  Preparing download package — this may take a moment...")
    elif build_error.value:
        solara.Error(f"❌ Failed to build ZIP: {build_error.value}")
    elif download_url.value:
        filename = f"{project_id}_{project_code}_Change_Detection.zip"
        solara.HTML(tag="div", style={"margin-top": "16px"}, unsafe_innerHTML=f"""
            <a href="{download_url.value}" download="{filename}"
               style="display:inline-flex;align-items:center;gap:8px;
                      padding:10px 22px;background-color:#2e7d32;color:white;
                      text-decoration:none;border-radius:6px;font-weight:bold;
                      font-size:15px;box-shadow:0 2px 6px rgba(0,0,0,0.3);">
               ⬇&nbsp; Download Results as ZIP
            </a>
            <p style="color:#aaa;font-size:12px;margin-top:6px;">
              File is ready — click to download.
            </p>""")


# ─────────────────────────────────────────────────────────────────────────────
#  App bar
# ─────────────────────────────────────────────────────────────────────────────

def app_bar():
    with solara.AppBar():
        solara.HTML(tag="div", unsafe_innerHTML="""
            <img src="https://raw.githubusercontent.com/nikhil-eq/EQ-GC-/refs/heads/main/Logo_white.png"
                 style="width:150px;position:absolute;left:16px;top:50%;transform:translateY(-50%);" />
        """)
        solara.AppBarTitle(" ")
        if auth.user.value:
            name = auth.user.value.get("userinfo", {}).get("name", "User")
            solara.Text(name, style={"color": "#aaa", "font-size": "13px", "margin-right": "10px"})
            solara.Button("Logout", icon_name="mdi-logout",
                          href=_safe_logout_url(), color="#360000",
                          style={"font-size": "12px"})


# ─────────────────────────────────────────────────────────────────────────────
#  Title banner
# ─────────────────────────────────────────────────────────────────────────────

def title():
    solara.HTML(tag="div", unsafe_innerHTML="""
        <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
        <div style="background:linear-gradient(135deg,#0a0f16 0%,#0d1a2a 60%,#081a10 100%);
                    border-bottom:1px solid #1a2a3a;padding:28px 32px 24px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
                      border:1px solid #4ade8018;border-radius:50%;"></div>
          <div style="position:absolute;top:-10px;right:-10px;width:120px;height:120px;
                      border:1px solid #4ade8030;border-radius:50%;"></div>
          <div style="font-family:Arial;font-size:10px;color:#4ade80;letter-spacing:0.2em;
                      text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:20px;height:1px;background:#4ade80;"></span>
            Forest Disturbance Monitor
          </div>
          <div style="font-family:Arial;font-size:26px;font-weight:300;color:#e8f4ff;
                      letter-spacing:-0.01em;line-height:1.2;margin-bottom:6px;">
            <strong style="font-weight:500;color:#ffffff;">iMAD</strong> Change Detection
          </div>
          <div style="font-size:13px;color:#6a8aaa;font-weight:300;max-width:520px;font-family:Arial;">
            Iteratively re-weighted Multivariate Alteration Detection using Sentinel-2 L2A imagery via Planetary Computer.
          </div>
        </div>""")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode options
# ─────────────────────────────────────────────────────────────────────────────

MODES = [
    "Single Scene  with    CEA",
    "Single Scene  without CEA",
    "Multi-Scene   with    CEA",
    "Multi-Scene   without CEA",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Main processing panel
# ─────────────────────────────────────────────────────────────────────────────

_CARD_STYLE = {
    "background": "#0d1520", "border": "1px solid #1a2a3a",
    "border-radius": "6px", "flex": "1", "padding": "20px",
}

_SECTION_LABEL = lambda text: f"""
    <div style="font-family:Arial;font-size:9px;letter-spacing:0.18em;
                text-transform:uppercase;color:#4ade80;margin-bottom:14px;
                display:flex;align-items:center;gap:8px;">
      {text}
      <span style="flex:1;height:1px;background:#1a2a3a;display:block;"></span>
    </div>"""


@solara.component
def ProcessingPanel(mode: str):
    multi_scene = "Multi" in mode
    with_cea    = "with    CEA" in mode

    xl_sheet = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "Change Detection Tracker.xlsx"))
    proj_list = xl_sheet["Project name"].dropna().tolist()

    selected         = solara.use_reactive(None)
    date1            = solara.use_reactive(None)
    date2            = solara.use_reactive(None)
    cea_file_content = solara.use_reactive(None)
    cea_file_name    = solara.use_reactive(None)
    is_processing    = solara.use_reactive(False)
    status_message   = solara.use_reactive(None)
    error_message    = solara.use_reactive(None)
    progress_value   = solara.use_reactive(0)
    results          = solara.use_reactive(None)
    map_object       = solara.use_reactive(None)

    def on_cea_upload(files):
        if files:
            f = files[0]
            file_obj = f["file_obj"]
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            data = file_obj.read()
            if not data:
                error_message.set("❌ Uploaded CEA file appears to be empty.")
                return
            cea_file_content.set(data)
            cea_file_name.set(f["name"])

    # ── Active mode badge ─────────────────────────────────────────────────────
    cea_badge_color  = "#4ade80" if with_cea else "#6a8aaa"
    cea_badge_border = "#1a4a2a" if with_cea else "#1a2a3a"
    cea_badge_bg     = "#091a10" if with_cea else "#0d1520"
    scene_label      = "Multi-Scene" if multi_scene else "Single Scene"
    cea_label        = "CEA" if with_cea else "No CEA"

    solara.HTML(tag="div", unsafe_innerHTML=f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding:14px 32px 0;background:#080c10;font-family:Arial;">
          <span style="font-size:9px;letter-spacing:0.15em;text-transform:uppercase;
                       color:#6a8aaa;">Active mode:</span>
          <span style="background:#0a1a2a;border:1px solid #1a3a5a;border-radius:4px;
                       padding:5px 12px;font-size:11px;color:#7ab8e8;letter-spacing:0.08em;">
            {scene_label}
          </span>
          <span style="background:{cea_badge_bg};border:1px solid {cea_badge_border};
                       border-radius:4px;padding:5px 12px;font-size:11px;
                       color:{cea_badge_color};letter-spacing:0.08em;">
            {'✓ ' if with_cea else ''}{cea_label}
          </span>
        </div>""")

    # ── Input cards row ──────────────────────────────────────────────────────
    with solara.Row(style={"gap": "16px", "padding": "16px 32px 24px",
                           "align-items": "flex-start", "background": "#080c10"}):

        # Left card: project selector + optional CEA uploader
        with solara.Card(style=_CARD_STYLE):
            solara.HTML(tag="div", unsafe_innerHTML=_SECTION_LABEL("Select the Project Boundary"))
            solara.Select(label="Project Boundary", values=proj_list,
                          value=selected.value, on_value=selected.set)

            if with_cea:
                solara.HTML(tag="div", unsafe_innerHTML=_SECTION_LABEL("Upload CEA File") + """
                    <div style="font-family:Arial;font-size:11px;color:#4a6a5a;
                                margin-bottom:10px;line-height:1.6;">
                      Accepted: <span style="color:#4ade80;font-weight:600;">.gpkg</span>,
                      <span style="color:#4ade80;font-weight:600;">.zip</span> (Shapefile or FileGDB)
                    </div>""")
                solara.FileDropMultiple(
                    label="Drop .gpkg or Shapefile / GDB .zip here",
                    on_total_progress=lambda *_: None,
                    on_file=on_cea_upload,
                    lazy=False,
                )
                if cea_file_name.value:
                    fn_lower = cea_file_name.value.lower()
                    fmt_label = "GeoPackage" if fn_lower.endswith(".gpkg") else "ZIP archive"
                    fmt_color = "#0a4a2a" if fn_lower.endswith(".gpkg") else "#0a2a4a"
                    fmt_border = "#1a6a3a" if fn_lower.endswith(".gpkg") else "#1a3a6a"
                    solara.HTML(tag="div", unsafe_innerHTML=f"""
                        <div style="display:flex;align-items:center;gap:8px;margin-top:10px;
                                    background:{fmt_color};border:1px solid {fmt_border};
                                    border-radius:4px;padding:7px 12px;font-size:12px;
                                    color:#4ade80;font-family:Arial;">
                          ✓ &nbsp;{cea_file_name.value}
                          <span style="margin-left:auto;font-size:10px;letter-spacing:0.1em;
                                       color:#2abe60;background:{fmt_border};
                                       padding:2px 6px;border-radius:3px;">{fmt_label}</span>
                        </div>""")

        # Right card: date range
        with solara.Card(style=_CARD_STYLE):
            solara.HTML(tag="div", unsafe_innerHTML=_SECTION_LABEL("Select Date Range"))
            solara.lab.InputDate(date1, label="Start Date", open_value=False)
            solara.HTML(tag="div", style={"margin-top": "12px"}, unsafe_innerHTML="")
            solara.lab.InputDate(date2, label="End Date")
            if date1.value and date2.value:
                solara.HTML(tag="div", unsafe_innerHTML=f"""
                    <div style="background:#0a1520;border:1px solid #1a2a3a;
                                border-left:2px solid #4ade80;border-radius:4px;
                                padding:10px 14px;font-family:Arial;font-size:10px;
                                color:#6a9a7a;letter-spacing:0.08em;margin-top:14px;">
                      RANGE → {date1.value} to {date2.value}
                    </div>""")

    # ── Check all required inputs are filled ─────────────────────────────────
    inputs_ready = (
        selected.value is not None and
        date1.value is not None and
        date2.value is not None and
        (not with_cea or cea_file_content.value is not None)
    )

    if not inputs_ready:
        return

    # ── Run button ────────────────────────────────────────────────────────────
    def run_processing():
        is_processing.set(True)
        status_message.set(None)
        error_message.set(None)
        results.set(None)
        map_object.set(None)
        try:
            config = SHConfig()
            config.sh_client_id = "0e39a266-3f64-4c11-ab55-e1176d5af787"
            config.sh_client_secret = "fq4XAvpH3wqhTeSkiqezoLaedEldNZ3D"
            config.download_timeout_seconds = 600

            selected_proj = xl_sheet[xl_sheet["Project name"] == str(selected.value)]
            roi_path      = str(selected_proj["Boundary"].iloc[0])
            project_id    = int(selected_proj["Project ID"].iloc[0])
            project_code  = str(selected_proj["Project Code"].iloc[0])

            roi_gdf = gpd.read_file(roi_path).dissolve()

            xmin, ymin, xmax, ymax = roi_gdf.to_crs("EPSG:4326").total_bounds
            longitude = (xmin + xmax) / 2
            zone      = math.floor((longitude + 180) / 6) + 1

            cea = None
            if with_cea:
                cea_raw = load_cea_from_bytes(cea_file_content.value, cea_file_name.value)
                cea = cea_raw.to_crs(f"EPSG:78{zone}")

            status_message.set(f"Identified MGA zone EPSG:78{zone}. Fetching images...")
            progress_value.set(15)

            # Date offsets: Start date ±2 days; End date ±3 months
            t1a = pd.to_datetime(date1.value)
            t2   = pd.to_datetime(date2.value)

            if multi_scene:
                t1b  = t1a + pd.DateOffset(days=5)
                t2a  = t2  + pd.DateOffset(months=-3)
                t2b  = t2  + pd.DateOffset(months=3)
                extractor = sentinel_extractor_multi
            else:
                t1b  = t1a + pd.DateOffset(days=2)
                t2a  = t2  + pd.DateOffset(months=-3)
                t2b  = t2  + pd.DateOffset(months=3)
                extractor = sentinel_extractor_single

            t1a_s = str(t1a.date()); t1b_s = str(t1b.date())
            t2a_s = str(t2a.date()); t2b_s = str(t2b.date())

            im1_date, im1_metadata, im1_extent, im1 = extractor(
                roi_gdf, "sentinel-2-l2a", t1a_s, t1b_s)
            im2_date, im2_metadata, im2_extent, im2 = extractor(
                roi_gdf, "sentinel-2-l2a", t2a_s, t2b_s)

            status_message.set("Images fetched. Running cloud masking...")
            progress_value.set(35)

            def _mask_image(im):
                scl_mask   = scl_masking(im).astype("uint8")
                omni_mask  = omnicloudmask(im)
                combined   = scl_mask | omni_mask
                return im.where(combined == 0)

            im1_masked = _mask_image(im1)
            im2_masked = _mask_image(im2)

            status_message.set("Cloud masking done. Running iMAD...")
            progress_value.set(55)

            ds1 = im1_masked.drop_sel(band="SCL").compute()
            ds2 = im2_masked.drop_sel(band="SCL").compute()

            _, iMAD_1, gdf_raw, gdf_clipped = run_change_detection(ds1, ds2, cea=cea)

            # Metadata id handling (single item vs list)
            def _item_id(meta):
                if multi_scene:
                    return ", ".join(i.id for i in meta[:3]) + ("..." if len(meta) > 3 else "")
                return meta.id

            results.set({
                "im1": im1_masked, "im2": im2_masked,
                "im1_metadata": im1_metadata, "im2_metadata": im2_metadata,
                "gdf_raw": gdf_raw, "gdf_clipped": gdf_clipped,
                "imad": iMAD_1,
                "project_id": project_id, "project_code": project_code,
                "im1_id": _item_id(im1_metadata), "im2_id": _item_id(im2_metadata),
                "im1_date": str(im1_date), "im2_date": str(im2_date),
                "roi_gdf": roi_gdf,
            })

            status_message.set("iMAD complete! Rendering map...")
            progress_value.set(80)

            m = build_folium_map(im1_masked, im2_masked, roi_gdf, gdf_raw,
                                  gdf_clipped=gdf_clipped)
            map_object.set(m)
            status_message.set("Map rendered! Download the results below.")
            progress_value.set(100)

        except Exception as e:
            error_message.set(f"❌ Error during processing: {e}")
        finally:
            is_processing.set(False)

    if not is_processing.value:
        solara.HTML(tag="div", unsafe_innerHTML="<div style='height:4px;'></div>")
        solara.Button(
            label="▶  RUN CHANGE DETECTION",
            on_click=run_processing, color="#2e7d32",
            style={"margin-top": "20px", "width": "100%", "font-family": "Arial",
                   "font-size": "11px", "letter-spacing": "0.15em",
                   "font-weight": "700", "padding": "14px", "border-radius": "4px"})
    # else:
    #     with solara.Row():
    #         solara.ProgressLinear(True)
    #         solara.Text("Processing... this may take several minutes.")

    if status_message.value:
        with solara.Row(style={"align-items": "center", "gap": "12px", "margin-top": "6px"}):
            with solara.Column(style={"flex": "1"}):
                solara.ProgressLinear(value=progress_value.value, color="#0E7500")
            solara.Text(f"{progress_value.value}%",
                        style={"min-width": "12px", "color": "#ffffff"})
            solara.Success(status_message.value,
                           style={"color": "white", "background-color": "#0E7500"})

    if results.value is not None:
        with solara.Row():
            with solara.Card(style=_CARD_STYLE):
                solara.Markdown(f"""
**📡 Start Image**\n
Image ID: `{results.value['im1_id']}`\n
Image Date: {results.value['im1_date']}""")
            with solara.Card(style=_CARD_STYLE):
                solara.Markdown(f"""
**📡 End Image**\n
Image ID: `{results.value['im2_id']}`\n
Image Date: {results.value['im2_date']}""")

    if error_message.value:
        solara.Error(error_message.value)

    if map_object.value is not None:
        solara.display(map_object.value)
        r = results.value
        DownloadZipButton(
            im1=r["im1"], im2=r["im2"],
            im1_metadata=r["im1_metadata"], im2_metadata=r["im2_metadata"],
            imad=r["imad"], gdf_raw=r["gdf_raw"],
            project_id=r["project_id"], project_code=r["project_code"],
            im1_date=r["im1_date"], im2_date=r["im2_date"],
            gdf_clipped=r.get("gdf_clipped"),
            multi_scene=multi_scene,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Mode selector component  ← UPDATED: dropdown replaces card grid
# ─────────────────────────────────────────────────────────────────────────────

@solara.component
def ModeSelectorAndPanel():
    selected_mode = solara.use_reactive(None)

    with solara.Column(style={"padding": "24px 32px 20px", "background": "#080c10", "gap": "0px"}):
        solara.HTML(tag="div", unsafe_innerHTML=_SECTION_LABEL("Select Processing Mode"))
        solara.HTML(tag="div", unsafe_innerHTML="""
            <div style="font-family:Arial;font-size:11px;color:#4a6a5a;
                        margin-bottom:14px;line-height:1.6;">
              Choose whether to use a single tile or multi-scene mode,
              and whether segmentation is already done (Post RP1) - disturbances will be intersected with CEA. 
            </div>""")
        solara.Select(
            label="Processing Mode",
            values=MODES,
            value=selected_mode.value,
            on_value=selected_mode.set,
        )
        if selected_mode.value:
            mode_label = selected_mode.value
            solara.HTML(tag="div", unsafe_innerHTML=f"""
                <div style="margin-top:12px;padding:10px 16px;background:#091a10;
                            border-left:3px solid #4ade80;border-radius:0 4px 4px 0;
                            font-family:Arial;font-size:11px;color:#4ade80;
                            letter-spacing:0.1em;">
                  ▶ &nbsp; <strong>{mode_label}</strong>
                </div>""")

    if selected_mode.value is not None:
        ProcessingPanel(mode=selected_mode.value)


# ─────────────────────────────────────────────────────────────────────────────
#  Auth pages
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_DOMAINS = ["equilibriumearth.com", "greencollargroup.com.au"]


@solara.component
def LoginPage():
    solara.HTML(tag="div", unsafe_innerHTML="""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:80vh;font-family:Arial;background:#0a0f16;">
          <img src="https://raw.githubusercontent.com/nikhil-eq/EQ-GC-/refs/heads/main/Logo_white.png"
               style="width:200px;margin-bottom:32px;" />
          <div style="color:#e8f4ff;font-size:22px;font-weight:300;margin-bottom:8px;">
            <strong>iMAD</strong> Change Detection
          </div>
          <div style="color:#6a8aaa;font-size:13px;margin-bottom:32px;">
            Please log in to access the application.
          </div>
        </div>""")
    with solara.Row(justify="space-around"):
        solara.Button("Login", icon_name="mdi-login", href=_safe_login_url(),
                      color="#000000",
                      style={"padding": "15px 40px", "font-size": "12px", "border-radius": "4px"})


@solara.component
def UnauthorizedPage():
    solara.HTML(tag="div", unsafe_innerHTML="""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:80vh;font-family:Arial;background:#0a0f16;">
          <img src="https://raw.githubusercontent.com/nikhil-eq/EQ-GC-/refs/heads/main/Logo_white.png"
               style="width:200px;margin-bottom:32px;" />
          <div style="color:#e8f4ff;font-size:22px;font-weight:300;margin-bottom:8px;">
            Access Denied
          </div>
          <div style="color:#6a8aaa;font-size:13px;margin-bottom:32px;">
            Only <strong style="color:#e57373;">@equilibriumearth.com or
            @greencollargroup.com.au</strong> accounts are permitted.
          </div>
        </div>""")
    solara.Button("Logout", icon_name="mdi-logout", href=_safe_logout_url(),
                  color="#000000",
                  style={"padding": "12px 32px", "font-size": "12px", "border-radius": "4px"})


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

@solara.component
def Page():
    solara.lab.theme.dark = True
    solara.Title("iMAD Change Detection — Equilibrium")
    solara.lab.theme.themes.dark.primary    = "#0a1628"
    solara.lab.theme.themes.dark.navigation = "#0a1628"

    if not auth.user.value:
        LoginPage()
    else:
        email  = auth.user.value.get("userinfo", {}).get("email", "")
        domain = email.lower().split("@")[-1]
        if domain not in ALLOWED_DOMAINS:
            UnauthorizedPage()
        else:
            app_bar()
            title()
            ModeSelectorAndPanel()


Page()