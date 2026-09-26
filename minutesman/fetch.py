# Download a recording from a share link (Google Drive, Dropbox, direct URL)
from __future__ import annotations

import logging
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


# Turn a browser share link into a direct download link
def direct_url(url: str) -> str:
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([\w-]+)", url)
    if m:
        # confirm=t skips the "can't scan this file for viruses" page on large files.
        return f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t"
    if "dropbox.com" in url:
        parts = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        query["dl"] = "1"
        return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
    return url


def _filename(resp, url: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
    name = urllib.parse.unquote(m.group(1)) if m else Path(urllib.parse.urlsplit(url).path).name
    return re.sub(r"[^\w.\-]+", "_", name) or "recording"


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    real = direct_url(url)
    req = urllib.request.Request(real, headers={"User-Agent": "minutesman"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError("The link returned a web page, not audio. Make sure the file is shared "
                               "as 'anyone with the link' and the link points to the file itself.")
        dst = dest_dir / _filename(resp, real)
        if dst.exists() and resp.headers.get("Content-Length") == str(dst.stat().st_size):
            log.info("Already downloaded: %s", dst)
            return dst
        log.info("Downloading %s (%s MB)", dst.name, round(int(resp.headers.get("Content-Length", 0)) / 1e6, 1))
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(dst)
    return dst
