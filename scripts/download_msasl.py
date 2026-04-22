"""
scripts/download_msasl.py

Downloads and trims all MS-ASL clips. Run from the project root:
    python3 scripts/download_msasl.py

Strategy:
  - Groups clips by YouTube URL: each unique video is downloaded once, then
    all clips from it are trimmed with ffmpeg. 7,213 downloads → 25,513 clips.
  - Already-trimmed clips are skipped on re-runs.

Output:
    data/raw/msasl/videos/train/00000.mp4, 00001.mp4, ...
    data/raw/msasl/videos/val/...
    data/raw/msasl/videos/test/...

Requirements:
    pip3 install yt-dlp tqdm
    brew install ffmpeg   # or apt install ffmpeg
"""

import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/raw/msasl")
VIDEO_DIR  = DATA_DIR / "videos"
SPLITS     = ["train", "val", "test"]
MIN_BYTES  = 5000
MAX_WORKERS  = 2   # parallel video downloads — keep low to avoid YouTube rate limiting
COOKIES_FILE = Path("data/raw/msasl/cookies.txt")  # export once with: yt-dlp --cookies-from-browser chrome --cookies data/raw/msasl/cookies.txt <any-yt-url>
COOKIE_BROWSER = ""  # disabled — use cookies.txt file instead (see README)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_split(split: str) -> list:
    with open(DATA_DIR / f"MSASL_{split}.json") as f:
        return json.load(f)


def clip_path(split: str, idx: int) -> Path:
    return VIDEO_DIR / split / f"{idx:05d}.mp4"


def already_done(split: str, idx: int) -> bool:
    p = clip_path(split, idx)
    return p.exists() and p.stat().st_size >= MIN_BYTES


def download_video(url: str, dest: Path) -> tuple[bool, str]:
    """Download a full YouTube video to dest. Returns (ok, reason)."""
    try:
        cmd = ["yt-dlp"]
        if COOKIES_FILE.exists():
            cmd += ["--cookies", str(COOKIES_FILE)]
        elif COOKIE_BROWSER:
            cmd += ["--cookies-from-browser", COOKIE_BROWSER]
        cmd += [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/bestvideo/best",
            "--merge-output-format", "mp4",
            "-o", str(dest),
            "--quiet", "--no-warnings",
            url,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore").strip().splitlines()
            return False, (stderr[-1] if stderr else "yt-dlp error")[:80]
        if not dest.exists() or dest.stat().st_size < MIN_BYTES:
            return False, "empty file"
        return True, "ok"
    except FileNotFoundError:
        return False, "yt-dlp not installed"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def trim_clip(src: Path, start: float, end: float, dest: Path) -> tuple[bool, str]:
    """Trim src from start to end seconds, write to dest."""
    duration = end - start
    if duration <= 0:
        return False, f"invalid duration ({duration:.2f}s)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-ss", str(start),
                "-i", str(src),
                "-t", str(duration),
                "-c", "copy",
                "-y",
                str(dest),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size < MIN_BYTES:
            if dest.exists():
                dest.unlink()
            return False, "ffmpeg error"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout"


def process_url(url: str, clips: list, counts: dict, fail_reasons: Counter, fail_log) -> None:
    """Download one video and trim all its clips. Updates counts/fail_reasons in place."""
    pending = [(split, idx, entry) for split, idx, entry in clips if not already_done(split, idx)]
    if not pending:
        counts["skip"] += len(clips)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "video.mp4"
        ok, reason = download_video(url, tmp_video)

        if not ok:
            counts["dl_fail"] += 1
            fail_reasons[reason] += len(pending)
            for split, idx, _ in pending:
                counts["fail"] += 1
                fail_log.write(json.dumps({
                    "split": split, "idx": idx, "url": url, "reason": reason
                }) + "\n")
            counts["skip"] += len(clips) - len(pending)
            return

        counts["dl_ok"] += 1

        for split, idx, entry in pending:
            dest = clip_path(split, idx)
            ok, reason = trim_clip(
                tmp_video,
                entry["start_time"],
                entry["end_time"],
                dest,
            )
            if ok:
                counts["ok"] += 1
            else:
                counts["fail"] += 1
                fail_reasons[reason] += 1
                fail_log.write(json.dumps({
                    "split": split, "idx": idx, "url": url,
                    "start": entry["start_time"], "end": entry["end_time"],
                    "reason": reason,
                }) + "\n")

        counts["skip"] += len(clips) - len(pending)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all splits and group by URL
    # url_clips[url] = [(split, idx, entry), ...]
    url_clips: dict = defaultdict(list)
    total_clips = 0

    for split in SPLITS:
        entries = load_split(split)
        VIDEO_DIR.joinpath(split).mkdir(parents=True, exist_ok=True)
        for idx, entry in enumerate(entries):
            url = entry.get("url", "").strip()
            if not url:
                continue
            url_clips[url].append((split, idx, entry))
            total_clips += 1

    unique_urls  = len(url_clips)
    already_have = sum(
        1 for clips in url_clips.values()
        for split, idx, _ in clips if already_done(split, idx)
    )

    print(f"Total clips:         {total_clips}")
    print(f"Unique YouTube URLs: {unique_urls}")
    print(f"Already trimmed:     {already_have}")
    print(f"Workers:             {MAX_WORKERS}\n")

    counts: dict      = {"ok": 0, "skip": 0, "fail": 0, "dl_ok": 0, "dl_fail": 0}
    fail_reasons: Counter = Counter()
    fail_log_path = DATA_DIR / "failed_downloads.jsonl"

    cookies_src = ("cookies.txt" if COOKIES_FILE.exists()
                   else COOKIE_BROWSER if COOKIE_BROWSER else "none")
    print(f"Cookie source:       {cookies_src}\n")

    with open(fail_log_path, "w") as fail_log:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_url, url, clips, counts, fail_reasons, fail_log): url
                for url, clips in url_clips.items()
            }
            with tqdm(total=unique_urls, desc="Videos", unit="video") as bar:
                for fut in as_completed(futures):
                    fut.result()
                    top = fail_reasons.most_common(1)[0][0][:35] if fail_reasons else ""
                    bar.set_postfix(
                        ok=counts["ok"],
                        fail=counts["fail"],
                        dl_fail=counts["dl_fail"],
                        why=top,
                    )
                    bar.update(1)

    total_good = counts["ok"] + counts["skip"]
    print(f"\nDone.")
    print(f"  Videos downloaded:  {counts['dl_ok']}")
    print(f"  Videos failed:      {counts['dl_fail']}")
    print(f"  Clips trimmed now:  {counts['ok']}")
    print(f"  Clips already had:  {counts['skip']}")
    print(f"  Clips failed:       {counts['fail']}")
    print(f"  Total usable clips: {total_good}")

    if fail_reasons:
        print(f"\nFailure breakdown (by clip count):")
        for reason, n in fail_reasons.most_common(15):
            print(f"  {n:>5}x  {reason}")
        print(f"\n  Full log: {fail_log_path.resolve()}")

    print(f"\nFiles saved to: {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()
