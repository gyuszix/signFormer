"""
scripts/download_wlasl.py

Downloads all available WLASL videos. Run from the project root:
    python3 scripts/download_wlasl.py

Skips already-downloaded files. Dead direct-link domains are blacklisted after
the first connection error so we don't waste time on every URL from a dead host.

Requirements:
    pip3 install requests tqdm yt-dlp
"""

import json
import argparse
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

try:
    import requests
except ImportError:
    raise ImportError("pip3 install requests")


# ── Config ────────────────────────────────────────────────────────────────────

ANNO_PATH   = Path("data/raw/wlasl/WLASL_v0.3.json")
SAVE_DIR    = Path("data/raw/wlasl/videos")
TIMEOUT     = 5
MIN_BYTES   = 5000
MAX_WORKERS = 8
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
VALID_CONTENT_TYPES = ("video/", "application/octet-stream", "binary/octet-stream")

# Shared set of dead domains — populated at runtime, checked before each request
dead_domains: set[str] = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_domain(url: str) -> str:
    return urlparse(url).netloc


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_direct(url: str, save_path: Path) -> tuple[bool, str]:
    domain = get_domain(url)
    if domain in dead_domains:
        return False, f"dead_domain ({domain})"
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS, stream=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        ct = r.headers.get("Content-Type", "")
        if not any(ct.startswith(t) for t in VALID_CONTENT_TYPES):
            return False, f"bad_content_type ({ct.split(';')[0].strip()})"
        data = b"".join(r.iter_content(chunk_size=16384))
        if len(data) < MIN_BYTES:
            return False, f"too_small ({len(data)}B)"
        save_path.write_bytes(data)
        return True, "ok"
    except requests.exceptions.SSLError:
        dead_domains.add(domain)
        return False, f"ssl_error ({domain})"
    except requests.exceptions.ConnectionError:
        dead_domains.add(domain)
        return False, f"connection_error ({domain})"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, type(e).__name__


def download_youtube(url: str, save_path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "--merge-output-format", "mp4",
                "-o", str(save_path),
                "--quiet", "--no-warnings",
                url,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore").strip().splitlines()
            reason = stderr[-1] if stderr else "yt-dlp error"
            return False, reason[:80]
        if not save_path.exists() or save_path.stat().st_size < MIN_BYTES:
            return False, "too_small"
        return True, "ok"
    except FileNotFoundError:
        return False, "yt-dlp not installed"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def process_one(url: str, save_path: Path, skip_youtube: bool) -> tuple[str, str]:
    """Returns (status, reason). Status: 'skip' | 'skip_yt' | 'ok' | 'fail'."""
    if save_path.exists() and save_path.stat().st_size >= MIN_BYTES:
        return "skip", ""

    if is_youtube(url):
        if skip_youtube:
            return "skip_yt", ""
        ok, reason = download_youtube(url, save_path)
    else:
        ok, reason = download_direct(url, save_path)

    if not ok and save_path.exists():
        save_path.unlink()

    return ("ok", "") if ok else ("fail", reason)


def build_queue(data: list) -> list:
    queue = []
    for entry in data:
        for inst in entry["instances"]:
            url    = inst.get("url", "").strip()
            vid_id = inst["video_id"]
            if not url:
                continue
            ext       = "swf" if url.endswith(".swf") else "mp4"
            save_path = SAVE_DIR / f"{vid_id}.{ext}"
            queue.append((url, vid_id, save_path))
    return queue


# ── Main ──────────────────────────────────────────────────────────────────────

def main(skip_youtube: bool = False):
    if not ANNO_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANNO_PATH}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNO_PATH) as f:
        data = json.load(f)

    queue   = build_queue(data)
    already = sum(1 for _, _, p in queue if p.exists() and p.stat().st_size >= MIN_BYTES)

    print(f"Words in dataset:    {len(data)}")
    print(f"Total instances:     {len(queue)}")
    print(f"Already downloaded:  {already}")
    print(f"To attempt:          {len(queue) - already}")
    print(f"YouTube:             {'skipped' if skip_youtube else 'enabled (yt-dlp)'}")
    print(f"Workers:             {MAX_WORKERS}")
    print(f"Save dir:            {SAVE_DIR.resolve()}\n")

    counts: dict      = {"ok": 0, "skip": 0, "skip_yt": 0, "fail": 0}
    fail_reasons: Counter = Counter()
    fail_log = SAVE_DIR.parent / "failed_downloads.jsonl"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_one, url, save_path, skip_youtube): (vid_id, url)
            for url, vid_id, save_path in queue
        }
        with tqdm(total=len(futures), desc="Downloading", unit="video") as bar:
            with open(fail_log, "w") as log:
                for fut in as_completed(futures):
                    vid_id, url = futures[fut]
                    status, reason = fut.result()
                    counts[status] = counts.get(status, 0) + 1
                    if status == "fail":
                        fail_reasons[reason] += 1
                        log.write(json.dumps({"video_id": vid_id, "url": url, "reason": reason}) + "\n")
                    top = fail_reasons.most_common(1)[0][0] if fail_reasons else ""
                    bar.set_postfix(ok=counts["ok"], skip=counts["skip"], fail=counts["fail"], why=top[:30])
                    bar.update(1)

    total_good = counts["ok"] + counts["skip"]
    print(f"\nDone.")
    print(f"  Downloaded now:  {counts['ok']}")
    print(f"  Already had:     {counts['skip']}")
    print(f"  Skipped (YT):    {counts['skip_yt']}")
    print(f"  Failed/bad URL:  {counts['fail']}")
    print(f"  Total usable:    {total_good}")

    if fail_reasons:
        print(f"\nFailure breakdown:")
        for reason, n in fail_reasons.most_common(15):
            print(f"  {n:>5}x  {reason}")
        print(f"\n  Full log: {fail_log.resolve()}")

    if dead_domains:
        print(f"\nDead domains ({len(dead_domains)}):")
        for d in sorted(dead_domains):
            print(f"  {d}")

    print(f"\nFiles saved to: {SAVE_DIR.resolve()}")

    if total_good > 0:
        print(f"\nNext — extract keypoints:")
        print(f"  python3 src/dataloader.py --extract --split train")
        print(f"  python3 src/dataloader.py --extract --split val")
        print(f"  python3 src/dataloader.py --extract --split test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all WLASL videos.")
    parser.add_argument(
        "--skip-youtube", action="store_true",
        help="Skip YouTube videos (faster, but fewer samples)"
    )
    args = parser.parse_args()
    main(skip_youtube=args.skip_youtube)
