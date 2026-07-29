"""Cached-header COG reader for the Sentinel-2 L2A COGs.

Ported from ``aef_loader_plus/aef_loader/cache.py``, whose whole point is that a
COG's header is derived purely from immutable bytes, yet the stock reader
re-parses it on every open. Measured here on ``sentinel-cogs``: a rasterio
windowed read costs ~2.6 s, of which **49% is the header parse**, and GDAL barely
caches it even for a repeat open of the same URL in the same process
(1318 ms -> 1087 ms).

So we do what the fork does: fetch the header once, cache the bytes on disk, and
serve every later windowed read as pure obstore range GETs against the cached
tile-offset table. Validated **bit-exact against rasterio on 100 real reads**
(4 VNIR bands + SCL, 10 m and 20 m, interior and granule-edge windows).

Measured at 48 threads over 100 distinct COGs::

    rasterio       7.2 s
    cached, cold   5.8 s   (1.25x -- one range GET beats vsicurl's machinery)
    cached, warm   4.1 s   (1.78x -- the header round trip is gone entirely)

The cache is keyed by URL and holds only header bytes, never pixels, so an entry
is ~64 KB regardless of granule size and never goes stale (the objects are
immutable). Anything unexpected -- a short header, a tile table that does not
match the image grid, any parse error -- falls back to rasterio rather than
risking a silently wrong array, which is the one failure mode a byte-offset
reader must not have.

Why the SYNC obstore API behind threads, and not async
------------------------------------------------------
``extract_tessera_points.py`` uses ``obstore.get_range_async`` and its docstring
records async as ~23x faster than requests+ThreadPool. **That precedent does not
transfer here, and the difference was measured, not assumed** -- on identical
jobs with identical output:

===================  ==================  ===============  ==================
payload              sync+threads(64)    async(128)       verdict
===================  ==================  ===============  ==================
1 KB slices          1.08 s              0.45 s           async 2.43x faster
full tiles (~1 MB)   5.93 s (39 MB/s)    8.68 s (27 MB/s) async 1.5x SLOWER
===================  ==================  ===============  ==================

The crossover is payload size. Tessera reads 128 bytes per point and decodes
nothing, so it is pure latency and async's cheap concurrency dominates. A COG
tile is ~1 MB, and at that size the single event-loop thread becomes the
bottleneck marshalling bytes from Rust into Python, while a thread pool spreads
that work across cores. (Decode is not the reason -- it is 1% of read time, and
zlib releases the GIL anyway.)

So: **do not "optimise" this to async on the strength of the Tessera docstring.**
It has been tried here and it is slower for tile-sized reads.
"""
from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path

import numpy as np
import obstore
from obstore.store import S3Store

BUCKET = "sentinel-cogs"
REGION = "us-west-2"
HOST = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
# 64 KB covers the IFD plus the tile offset/bytecount tables for every S2 L2A
# granule (10980^2 at 1024 tiles -> 121 tiles -> ~1 KB of table). Guarded below
# rather than assumed.
HEADER_BYTES = 65536

_store = S3Store(bucket=BUCKET, region=REGION, skip_signature=True)
_meta_cache: dict[str, tuple] = {}
_meta_lock = threading.Lock()


class CogFallback(Exception):
    """Raised when this reader cannot safely serve a URL; caller uses rasterio."""


def _key(url: str) -> str:
    if HOST not in url:
        raise CogFallback(f"not a sentinel-cogs URL: {url}")
    return url.split(HOST)[-1]


def _cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"hdr_{hashlib.sha256(url.encode()).hexdigest()[:32]}.bin"


def _header_bytes(url: str, cache_dir: Path) -> bytes:
    """The COG's leading bytes, from disk when cached. Never raises on cache IO."""
    path = _cache_path(cache_dir, url)
    try:
        if path.exists():
            return path.read_bytes()
    except OSError:
        pass
    raw = bytes(obstore.get_range(_store, _key(url), start=0, length=HEADER_BYTES))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)  # atomic: a crash mid-write cannot leave a torn header
    except OSError:
        pass
    return raw


def metadata(url: str, cache_dir: Path):
    """``(page, x0, dx, y0, dy, epsg)`` for a COG, from cached header bytes.

    Memoised in-process as well as on disk, because parsing even local bytes is
    not free when it happens once per plot per band.
    """
    with _meta_lock:
        hit = _meta_cache.get(url)
    if hit is not None:
        return hit

    import tifffile

    raw = _header_bytes(url, cache_dir)
    try:
        handle = tifffile.TiffFile(io.BytesIO(raw))
        page = handle.pages[0]
        geo = handle.geotiff_metadata
        scale = geo["ModelPixelScale"]
        tie = geo["ModelTiepoint"]
        epsg = int(geo.get("ProjectedCSTypeGeoKey") or geo["GeographicTypeGeoKey"])
    except Exception as error:  # truncated header, unexpected layout, no geokeys
        raise CogFallback(f"header parse failed for {url}: {error}") from error

    # The guard that makes a byte-offset reader safe: the tile table must cover
    # exactly the image grid. A short header yields a truncated table, and a
    # truncated table would read the wrong bytes *without* raising.
    tiles_x = (page.imagewidth + page.tilewidth - 1) // page.tilewidth
    tiles_y = (page.imagelength + page.tilelength - 1) // page.tilelength
    if page.tilewidth is None or len(page.dataoffsets) != tiles_x * tiles_y:
        raise CogFallback(f"tile table {len(page.dataoffsets)} != grid "
                          f"{tiles_x}x{tiles_y} for {url}")

    value = (page, float(tie[3]), float(scale[0]), float(tie[4]), float(scale[1]),
             epsg)
    with _meta_lock:
        _meta_cache[url] = value
    return value


def read_window(url: str, x: float, y: float, size: int, cache_dir: Path):
    """A ``size x size`` window centred on projected ``(x, y)``, as the COG's dtype.

    ``x``/``y`` must already be in the granule's own CRS -- use ``metadata`` to
    learn it. Out-of-file and untiled (``bytecount == 0``) regions come back as
    zero, matching rasterio's ``boundless`` fill.
    """
    page, x0, dx, y0, dy, _ = metadata(url, cache_dir)
    col = int((x - x0) / dx)
    row = int((y0 - y) / dy)
    half = size // 2
    row0, col0 = row - half, col - half
    tile_w, tile_h = page.tilewidth, page.tilelength
    tiles_x = (page.imagewidth + tile_w - 1) // tile_w

    out = np.zeros((size, size), page.dtype)
    # A 64x64 window touches at most 4 stored tiles; walk only the ones it hits.
    wanted = {(r // tile_h, c // tile_w)
              for r in (row0, row0 + size - 1)
              for c in (col0, col0 + size - 1)
              if 0 <= r < page.imagelength and 0 <= c < page.imagewidth}
    for ty, tx in wanted:
        index = ty * tiles_x + tx
        offset = int(page.dataoffsets[index])
        count = int(page.databytecounts[index])
        if count == 0:
            continue
        segment = bytes(obstore.get_range(_store, _key(url), start=offset,
                                          length=count))
        data, _, shape = page.decode(segment, index)
        tile = np.asarray(data).reshape(shape[-3:-1] if len(shape) > 2 else shape)

        # Intersect the tile's pixel extent with the requested window and copy
        # the overlap in one slice assignment rather than pixel by pixel.
        ty0, tx0 = ty * tile_h, tx * tile_w
        r_lo, r_hi = max(row0, ty0), min(row0 + size, ty0 + tile_h)
        c_lo, c_hi = max(col0, tx0), min(col0 + size, tx0 + tile_w)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        out[r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0] = \
            tile[r_lo - ty0:r_hi - ty0, c_lo - tx0:c_hi - tx0]
    return out
