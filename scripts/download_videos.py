#!/usr/bin/env python3
"""
Multi-backend video downloader driven by data/manifest.csv.

Backends:
  direct        HTTP GET of `video_url` (most permissive, use when you have a
                direct .mp4/.mov URL).
  pmc_supp      Scrape PMC article `page_url`, follow links to its /bin/
                supplementary files, download videos or zips of videos.
  ilae_cookie   Fetch EpilepsyDiagnosis.org `page_url` with a cookies.txt
                session, extract HTML5 <video>/<source> URLs, download.
  zenodo_zip    Download `video_url` (a Zenodo archive), extract mp4/avi/mov.

Output layout matches pipeline/flow_extract.py expectations:
  data/videos/{label}/{id}__{n}.mp4

The manifest is updated in place: `status` transitions
proposed -> approved -> downloaded|failed|rejected, and `local_path` is
filled once a file lands. `approved` is user-gated; the script only
downloads rows where status is `approved` (or `proposed` if --include-proposed).

Usage:
  python scripts/download_videos.py                          # approved only
  python scripts/download_videos.py --include-proposed       # also proposed
  python scripts/download_videos.py --only-label seizure
  python scripts/download_videos.py --only-id ilae_seizure_02
  python scripts/download_videos.py --cookies data/ilae_cookies.txt
  python scripts/download_videos.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_VIDEO_ROOT = REPO_ROOT / "data" / "videos"
DEFAULT_COOKIES = REPO_ROOT / "data" / "ilae_cookies.txt"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


@dataclass
class Row:
    id: str
    source: str
    page_url: str
    video_url: str
    tier: str
    label: str
    subtype: str
    backend: str
    status: str
    local_path: str
    notes: str
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Row":
        known = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**{k: d.get(k, "") for k in known}, extra=extra)

    def to_dict(self) -> dict[str, str]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
        d.update(self.extra)
        return d


def load_manifest(path: Path) -> tuple[list[Row], list[str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [Row.from_dict(r) for r in reader]
    return rows, fieldnames


def save_manifest(path: Path, rows: list[Row], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())
    tmp.replace(path)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def make_session(cookies_path: Path | None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    if cookies_path and cookies_path.exists():
        jar = http.cookiejar.MozillaCookieJar(str(cookies_path))
        # ignore_discard/ignore_expires=True so short-lived session cookies load
        jar.load(ignore_discard=True, ignore_expires=True)
        s.cookies = jar  # type: ignore[assignment]
    return s


class InterstitialError(RuntimeError):
    """Raised when a response returns a JS challenge / HTML interstitial
    instead of the expected binary (e.g. PMC cloudpmc-viewer-pow).
    """


def stream_download(
    session: requests.Session, url: str, dest: Path, *, referer: str | None = None
) -> None:
    headers = {}
    if referer:
        headers["Referer"] = referer
    with session.get(url, stream=True, headers=headers, timeout=60) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        first = True
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                if first:
                    # Reject HTML interstitials masquerading as binary (PMC PoW,
                    # Cloudflare, etc.). Real zips/mp4s start with a magic byte.
                    head = chunk[:512].lstrip().lower()
                    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
                        f.close()
                        dest.unlink(missing_ok=True)
                        raise InterstitialError(
                            f"got HTML interstitial from {url} "
                            "(likely JS challenge / anti-bot)"
                        )
                    first = False
                f.write(chunk)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def backend_direct(
    row: Row, session: requests.Session, video_root: Path
) -> list[Path]:
    if not row.video_url:
        raise ValueError("direct backend requires video_url")
    dest = video_root / row.label / f"{row.id}.mp4"
    stream_download(session, row.video_url, dest, referer=row.page_url or None)
    return [dest]


_PMC_BIN_RE = re.compile(r'href="([^"]*?/bin/[^"]+)"')


def backend_pmc_supp(
    row: Row, session: requests.Session, video_root: Path
) -> list[Path]:
    """Find /bin/ supplementary files on a PMC article page, download videos.

    PMC stores supplementary material under `/articles/instance/<pmcid>/bin/<file>`.
    Download every video-like file, plus every zip (zips commonly bundle videos).
    """
    resp = session.get(row.page_url, timeout=60)
    resp.raise_for_status()
    html = resp.text
    rel_links = set(_PMC_BIN_RE.findall(html))
    if not rel_links:
        raise RuntimeError(f"no /bin/ links found on {row.page_url}")

    saved: list[Path] = []
    label_dir = video_root / row.label
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, rel in enumerate(sorted(rel_links)):
        abs_url = urllib.parse.urljoin(row.page_url, rel)
        name = urllib.parse.unquote(Path(urllib.parse.urlparse(abs_url).path).name)
        ext = Path(name).suffix.lower()
        is_video = ext in VIDEO_EXTS
        is_zip = ext == ".zip"
        if not (is_video or is_zip):
            continue
        staged = label_dir / f"{row.id}__{i:02d}__{name}"
        stream_download(session, abs_url, staged, referer=row.page_url)
        if is_video:
            saved.append(staged)
        else:  # zip — extract any videos inside
            saved.extend(_extract_videos_from_zip(staged, label_dir, row.id, i))
            staged.unlink()
    if not saved:
        raise RuntimeError(f"no video-like files in PMC supp for {row.id}")
    return saved


def _extract_videos_from_zip(
    zpath: Path, dest_dir: Path, row_id: str, idx: int
) -> list[Path]:
    out: list[Path] = []
    with zipfile.ZipFile(zpath) as zf:
        for j, info in enumerate(zf.infolist()):
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in VIDEO_EXTS:
                continue
            target = dest_dir / f"{row_id}__{idx:02d}__{j:02d}{ext}"
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


_VIDEO_SRC_PATTERNS = [
    # HTML5 <video src="..."> or <source src="...">
    re.compile(r'<(?:video|source)[^>]+src="([^"]+\.(?:mp4|m4v|webm|mov))"', re.I),
    # JS-inlined data — look for any http(s) URL ending in video extension
    re.compile(r'https?://[^"\'\s<>]+\.(?:mp4|m4v|webm|mov)', re.I),
    # HLS playlists
    re.compile(r'https?://[^"\'\s<>]+\.m3u8[^"\'\s<>]*', re.I),
]


def backend_ilae_cookie(
    row: Row, session: requests.Session, video_root: Path
) -> list[Path]:
    """Fetch an EpilepsyDiagnosis.org page as a logged-in user, pull videos."""
    resp = session.get(row.page_url, timeout=60)
    resp.raise_for_status()
    html = resp.text
    if "Log In For Videos" in html or "/login.html" in html and "Log In" in html:
        # Not definitive — the logged-out page shows this banner. Warn but continue;
        # if we still find URLs below, the cookies worked partially.
        pass

    urls: list[str] = []
    for pat in _VIDEO_SRC_PATTERNS:
        for m in pat.findall(html):
            u = urllib.parse.urljoin(row.page_url, m)
            if u not in urls:
                urls.append(u)

    if not urls:
        raise RuntimeError(
            f"no video URLs found on {row.page_url} — "
            "check cookies.txt is valid and page is gated on login"
        )

    saved: list[Path] = []
    label_dir = video_root / row.label
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, u in enumerate(urls):
        ext = Path(urllib.parse.urlparse(u).path).suffix.lower()
        if ext == ".m3u8":
            dest = label_dir / f"{row.id}__{i:02d}.mp4"
            _ffmpeg_hls_to_mp4(u, dest, referer=row.page_url)
        else:
            dest = label_dir / f"{row.id}__{i:02d}{ext or '.mp4'}"
            stream_download(session, u, dest, referer=row.page_url)
        saved.append(dest)
    return saved


def _ffmpeg_hls_to_mp4(m3u8_url: str, dest: Path, *, referer: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-headers", f"Referer: {referer}\r\nUser-Agent: {USER_AGENT}\r\n",
        "-i", m3u8_url,
        "-c", "copy",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def backend_zenodo_zip(
    row: Row, session: requests.Session, video_root: Path
) -> list[Path]:
    if not row.video_url:
        raise ValueError("zenodo_zip backend requires video_url (direct zip link)")
    label_dir = video_root / row.label
    label_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        stream_download(session, row.video_url, tmp_path)
        return _extract_videos_from_zip(tmp_path, label_dir, row.id, 0)
    finally:
        tmp_path.unlink(missing_ok=True)


BACKENDS = {
    "direct": backend_direct,
    "pmc_supp": backend_pmc_supp,
    "ilae_cookie": backend_ilae_cookie,
    "zenodo_zip": backend_zenodo_zip,
}


# ---------------------------------------------------------------------------
# ffprobe metadata
# ---------------------------------------------------------------------------


def ffprobe_meta(path: Path) -> dict[str, str]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    data = json.loads(out)
    vstream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    meta = {}
    if vstream:
        meta["resolution"] = f"{vstream.get('width')}x{vstream.get('height')}"
    fmt = data.get("format", {})
    if "duration" in fmt:
        meta["duration_s"] = f"{float(fmt['duration']):.2f}"
    return meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def select_rows(
    rows: Iterable[Row],
    *,
    include_proposed: bool,
    only_label: str | None,
    only_id: str | None,
    only_backend: str | None,
) -> list[Row]:
    want_status = {"approved"} | ({"proposed"} if include_proposed else set())
    out = []
    for r in rows:
        if r.status not in want_status:
            continue
        if only_label and r.label != only_label:
            continue
        if only_id and r.id != only_id:
            continue
        if only_backend and r.backend != only_backend:
            continue
        out.append(r)
    return out


def run(args: argparse.Namespace) -> int:
    manifest_path: Path = args.manifest
    rows, fieldnames = load_manifest(manifest_path)

    # Ensure metadata columns exist so we can write them back.
    for col in ("duration_s", "resolution"):
        if col not in fieldnames:
            fieldnames.append(col)

    selected = select_rows(
        rows,
        include_proposed=args.include_proposed,
        only_label=args.only_label,
        only_id=args.only_id,
        only_backend=args.only_backend,
    )
    print(f"Selected {len(selected)} / {len(rows)} rows")
    if args.dry_run:
        for r in selected:
            print(f"  [{r.status:8s}] {r.id:24s} {r.backend:12s} {r.label:10s} {r.page_url}")
        return 0
    if args.limit:
        selected = selected[: args.limit]

    session = make_session(args.cookies)
    video_root: Path = args.video_root

    for r in selected:
        backend = BACKENDS.get(r.backend)
        if backend is None:
            print(f"[skip] {r.id}: unknown backend {r.backend!r}", file=sys.stderr)
            continue
        print(f"[{r.backend}] {r.id} -> {r.label}/")
        try:
            paths = backend(r, session, video_root)
        except InterstitialError as e:
            r.status = "blocked_interstitial"
            r.extra["error"] = str(e)[:200]
            if "error" not in fieldnames:
                fieldnames.append("error")
            print(f"  BLOCKED: {e}", file=sys.stderr)
            save_manifest(manifest_path, rows, fieldnames)
            continue
        except Exception as e:  # noqa: BLE001 — surface any backend failure
            r.status = "failed"
            r.extra["error"] = str(e)[:200]
            if "error" not in fieldnames:
                fieldnames.append("error")
            print(f"  FAIL: {e}", file=sys.stderr)
            save_manifest(manifest_path, rows, fieldnames)
            continue

        r.local_path = ";".join(str(p.relative_to(REPO_ROOT)) for p in paths)
        r.status = "downloaded"
        if paths:
            meta = ffprobe_meta(paths[0])
            for k, v in meta.items():
                r.extra[k] = v
        print(f"  OK: {len(paths)} file(s)")
        save_manifest(manifest_path, rows, fieldnames)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--cookies", type=Path, default=DEFAULT_COOKIES,
                   help="Netscape-format cookies.txt for ILAE (and any Tier B site)")
    p.add_argument("--include-proposed", action="store_true",
                   help="Also download entries with status=proposed (default: approved only)")
    p.add_argument("--only-label", type=str)
    p.add_argument("--only-id", type=str)
    p.add_argument("--only-backend", type=str)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
