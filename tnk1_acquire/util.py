"""Shared plumbing: HTTP with retries, caching, checksums, manifest, logging."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlparse

import requests

LOG = logging.getLogger("tnk1.acquire")

# Generic client name only -- never leak user/project identity to external hosts.
USER_AGENT = "tnk1-data-acquire/1.0 (python-requests)"

CHUNK = 1 << 20  # 1 MiB


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layout:
    """Canonical on-disk layout. Every tier writes only inside these dirs."""

    root: Path

    @property
    def raw(self) -> Path:          return self.root / "raw"
    @property
    def interim(self) -> Path:      return self.root / "interim"
    @property
    def processed(self) -> Path:    return self.root / "processed"
    @property
    def features(self) -> Path:     return self.root / "features"
    @property
    def external(self) -> Path:     return self.root / "external"   # untrusted downloads
    @property
    def structures(self) -> Path:   return self.root / "structures"
    @property
    def models(self) -> Path:       return self.root / "models"
    @property
    def logs(self) -> Path:         return self.root / "logs"
    @property
    def manifest_path(self) -> Path: return self.root / "MANIFEST.json"

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.features,
                  self.external, self.structures, self.models, self.logs):
            p.mkdir(parents=True, exist_ok=True)

    def download_dir(self, source_id: str) -> Path:
        """Each external source gets its OWN empty dir under external/.

        Downloaded archives are untrusted data. Keeping them isolated from the
        code tree means a planted `json.py` or `sitecustomize.py` can never land
        next to a script we execute.
        """
        d = self.external / source_id
        d.mkdir(parents=True, exist_ok=True)
        return d


def setup_logging(layout: Layout, verbose: bool = True) -> None:
    layout.logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(layout.logs / "acquire.log"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
class Http:
    """requests.Session with polite retry/backoff and streamed downloads."""

    def __init__(self, timeout: int = 120, max_retries: int = 5,
                 min_interval: float = 0.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last = 0.0

    def _throttle(self) -> None:
        if self.min_interval:
            dt = time.monotonic() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.monotonic()

    def get(self, url: str, *, params: dict | None = None,
            stream: bool = False) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout,
                               stream=stream)
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(60, 2 ** attempt) + 0.5 * attempt
                    LOG.warning("HTTP %s on %s -- retry in %.1fs",
                                r.status_code, url, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:          # noqa: PERF203
                last_exc = exc
                wait = min(60, 2 ** attempt)
                LOG.warning("%s on %s -- retry in %ds", type(exc).__name__, url, wait)
                time.sleep(wait)
        raise RuntimeError(f"GET failed after {self.max_retries} tries: {url}") from last_exc

    def json(self, url: str, *, params: dict | None = None) -> Any:
        return self.get(url, params=params).json()

    def download(self, url: str, dest: Path, *, force: bool = False,
                 expect_sha256: str | None = None) -> Path:
        """Stream a URL to `dest`, resumable-skip if already correct."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            if expect_sha256 is None:
                LOG.info("skip (exists): %s", dest.name)
                return dest
            if sha256_file(dest) == expect_sha256.lower():
                LOG.info("skip (sha256 ok): %s", dest.name)
                return dest
            LOG.warning("checksum mismatch, re-downloading %s", dest.name)

        tmp = dest.with_suffix(dest.suffix + ".part")
        LOG.info("GET %s -> %s", url, dest)
        with self.get(url, stream=True) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            t0 = time.monotonic()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done % (64 * CHUNK) < CHUNK:
                        pct = 100 * done / total
                        rate = done / max(1e-6, time.monotonic() - t0) / 1e6
                        LOG.info("  %s  %5.1f%%  %6.1f MB/s", dest.name, pct, rate)
        tmp.replace(dest)
        got = sha256_file(dest)
        if expect_sha256 and got != expect_sha256.lower():
            raise RuntimeError(
                f"sha256 mismatch for {dest}: got {got}, expected {expect_sha256}")
        LOG.info("done %s (%s, sha256=%s)", dest.name, human(dest.stat().st_size), got[:16])
        return dest


# --------------------------------------------------------------------------- #
# hashing / manifest
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


class Manifest:
    """Append-only record of every artefact, with sha256 + provenance.

    The acceptance test from the provenance doc ("rebuilt files must reproduce
    the manifest's SHA-256 values") runs off this file -- see `verify()`.
    """

    def __init__(self, layout: Layout):
        self.layout = layout
        self.path = layout.manifest_path
        self.data: dict[str, Any] = {"schema": 1, "entries": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                LOG.warning("manifest unreadable; starting fresh")

    def record(self, path: Path, *, tier: str, source: str, url: str | None = None,
               rows: int | None = None, notes: str | None = None,
               extra: dict | None = None) -> None:
        if not path.exists():
            LOG.warning("manifest: %s does not exist, not recorded", path)
            return
        key = str(path.relative_to(self.layout.root)) if self._under_root(path) else str(path)
        self.data["entries"][key] = {
            "tier": tier,
            "source": source,
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": rows,
            "notes": notes,
            "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(extra or {}),
        }
        self.flush()

    def _under_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.layout.root)
            return True
        except ValueError:
            return False

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))

    def verify(self) -> tuple[list[str], list[str], list[str]]:
        """Return (ok, mismatched, missing) relative paths."""
        ok, bad, missing = [], [], []
        for key, meta in sorted(self.data["entries"].items()):
            p = self.layout.root / key
            if not p.exists():
                missing.append(key)
            elif sha256_file(p) == meta["sha256"]:
                ok.append(key)
            else:
                bad.append(key)
        return ok, bad, missing


# --------------------------------------------------------------------------- #
# archives
# --------------------------------------------------------------------------- #
def _safe_members(names: Iterable[str], dest: Path) -> None:
    """Reject absolute paths / traversal before extracting untrusted archives."""
    dest = dest.resolve()
    for name in names:
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"unsafe archive member: {name!r}")


def extract(archive: Path, dest: Path) -> Path:
    """Extract zip/tar/tar.gz/gz into `dest` (created if needed), safely."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _safe_members(zf.namelist(), dest)
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive) as tf:
            _safe_members([m.name for m in tf.getmembers()], dest)
            tf.extractall(dest)                                    # noqa: S202
    elif name.endswith(".gz"):
        out = dest / archive.name[:-3]
        with gzip.open(archive, "rb") as fi, out.open("wb") as fo:
            shutil.copyfileobj(fi, fo, CHUNK)
        return out
    else:
        raise ValueError(f"unknown archive type: {archive}")
    LOG.info("extracted %s -> %s", archive.name, dest)
    return dest


def gz_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:               # type: ignore[operator]
        yield from fh


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def try_import(mod: str) -> Any | None:
    try:
        return __import__(mod)
    except Exception:                                              # noqa: BLE001
        return None


def require(mod: str, pip_name: str | None = None) -> Any:
    m = try_import(mod)
    if m is None:
        raise SystemExit(
            f"missing dependency `{mod}` -- install with: pip install {pip_name or mod}")
    return m


def write_parquet(df, path: Path, manifest: Manifest | None = None, **meta) -> Path:
    """Parquet if pyarrow is around, else CSV.gz; always record in manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if try_import("pyarrow") is not None:
        df.to_parquet(path, index=False, compression="zstd"
                      if _zstd_ok() else "snappy")
        out = path
    else:
        out = path.with_suffix(".csv.gz")
        LOG.warning("pyarrow missing -- writing %s instead of parquet", out.name)
        df.to_csv(out, index=False, compression="gzip")
    LOG.info("wrote %s rows=%d (%s)", out.name, len(df), human(out.stat().st_size))
    if manifest is not None:
        manifest.record(out, rows=len(df), **meta)
    return out


def _zstd_ok() -> bool:
    pa = try_import("pyarrow")
    if pa is None:
        return False
    try:
        import pyarrow.lib as lib  # noqa: PLC0415
        return "zstd" in getattr(lib, "Codec", type("x", (), {})).__dict__ or True
    except Exception:                                              # noqa: BLE001
        return True


def host_of(url: str) -> str:
    return urlparse(url).netloc
