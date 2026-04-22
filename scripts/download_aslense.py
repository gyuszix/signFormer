"""
scripts/download_aslense.py

Download videos from the Aslense ASL dataset on HuggingFace that overlap
with the project vocabulary. Only downloads clips for words already in
data/processed/vocab.json — skips everything else to save time and disk space.

Source: https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset
License: MIT
Stats:  108,618 videos · 2,208 ASL words · 30+ videos per word minimum

Usage:
    python3 scripts/download_aslense.py
    python3 scripts/download_aslense.py --vocab-path data/processed/vocab.json
    python3 scripts/download_aslense.py --save-dir data/raw/aslense --limit 50

Requirements:
    pip3 install huggingface_hub tqdm
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_vocab(vocab_path: str) -> dict:
    """Load vocab and return uppercase word -> (original_word, label_idx) mapping."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {w.upper(): (w, i) for w, i in vocab.items()}


def get_overlap(vocab_upper: dict, limit: int = None) -> list[tuple[str, str]]:
    """
    List all HuggingFace repo files and return (filepath, word) pairs
    that overlap with the project vocabulary.

    Args:
        vocab_upper: {UPPERCASE_WORD: (original_word, label_idx)}
        limit:       max videos per word (None = no limit)

    Returns:
        list of (hf_filepath, uppercase_word) tuples to download
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        raise ImportError("pip3 install huggingface_hub")

    print("Fetching file list from HuggingFace (this takes ~30s)...")
    files = list(list_repo_files(
        "akasheroor/American-Sign-Language-Dataset",
        repo_type="dataset"))

    mp4_files = [f for f in files if f.endswith('.mp4')]
    print(f"Total videos in repo: {len(mp4_files)}")

    # Count per word to apply limit
    word_counts = {}
    to_download = []

    for filepath in mp4_files:
        # Filename format: {id}-{WORD}.mp4
        word_upper = filepath.split('-', 1)[-1].replace('.mp4', '').upper().strip()
        if word_upper not in vocab_upper:
            continue
        count = word_counts.get(word_upper, 0)
        if limit and count >= limit:
            continue
        to_download.append((filepath, word_upper))
        word_counts[word_upper] = count + 1

    print(f"Words overlapping with vocab: {len(word_counts)}")
    print(f"Videos to download: {len(to_download)}")
    return to_download


def download_videos(to_download: list, save_dir: Path) -> int:
    """
    Download videos from HuggingFace, saving flat to save_dir.
    Skips files that already exist. Returns count of newly downloaded files.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip3 install huggingface_hub")

    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0

    for filepath, word_upper in tqdm(to_download, desc="Downloading"):
        dest = save_dir / Path(filepath).name
        if dest.exists() and dest.stat().st_size > 1000:
            continue

        try:
            local = hf_hub_download(
                repo_id="akasheroor/American-Sign-Language-Dataset",
                filename=filepath,
                repo_type="dataset",
                local_dir=str(save_dir),
                local_dir_use_symlinks=False)

            # hf_hub_download may save in a nested subfolder — move to flat
            local_path = Path(local)
            if local_path != dest and local_path.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                local_path.rename(dest)

            downloaded += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\nFailed {Path(filepath).name}: {e}")

    print(f"\nDownloaded: {downloaded}  |  Failed: {failed}  |  "
          f"Total in dir: {len(list(save_dir.glob('*.mp4')))}")
    return downloaded


def build_manifest(save_dir: Path, vocab_upper: dict,
                   proc_dir: Path) -> list[dict]:
    """
    Build a manifest JSON for all downloaded Aslense videos and
    merge with existing WLASL manifest into combined_train_manifest.json.
    """
    mp4s = list(save_dir.glob("*.mp4"))
    manifest = []

    for mp4 in mp4s:
        word_upper = mp4.stem.split('-', 1)[-1].upper().strip()
        if word_upper not in vocab_upper:
            continue
        original_word, label_idx = vocab_upper[word_upper]
        manifest.append({
            "path":      str(mp4),
            "label":     original_word,
            "label_idx": label_idx,
            "source":    "aslense",
        })

    manifest_path = proc_dir / "aslense_train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Aslense manifest: {len(manifest)} samples → {manifest_path}")

    # Merge all manifests into combined
    combined = []
    for name in ["train_manifest.json",
                 "aslcitizen_train_manifest.json",
                 "aslense_train_manifest.json"]:
        p = proc_dir / name
        if p.exists():
            with open(p) as f:
                entries = json.load(f)
            for e in entries:
                e.setdefault("source", name.split("_")[0])
            combined += entries
            print(f"  + {len(entries):5d} from {name}")

    combined_path = proc_dir / "combined_train_manifest.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined train manifest: {len(combined)} samples → {combined_path}")
    return manifest


def main(vocab_path: str, save_dir: str, limit: int, skip_manifest: bool):
    vocab_path_obj = Path(vocab_path)
    save_dir_obj   = Path(save_dir)
    proc_dir       = Path("data/processed")

    if not vocab_path_obj.exists():
        raise FileNotFoundError(
            f"vocab.json not found at {vocab_path_obj}. "
            "Run keypoint extraction first to build the vocabulary.")

    vocab_upper = load_vocab(vocab_path)
    print(f"Vocab size: {len(vocab_upper)} words\n")

    to_download = get_overlap(vocab_upper, limit=limit)

    if not to_download:
        print("Nothing to download — check vocab path or HuggingFace connectivity.")
        return

    # Disk space estimate
    est_gb = len(to_download) * 5 / 1024  # ~5MB per video
    print(f"\nEstimated download size: ~{est_gb:.1f} GB")
    print(f"Save directory: {save_dir_obj.resolve()}\n")

    download_videos(to_download, save_dir_obj)

    if not skip_manifest:
        print("\nBuilding manifest...")
        build_manifest(save_dir_obj, vocab_upper, proc_dir)
        print("\nNext step — extract keypoints:")
        print("  python3 scripts/preprocess_aslense.py")
        print("  (or re-run training with --combined flag)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Aslense ASL videos that overlap with project vocab")
    parser.add_argument("--vocab-path",     default="data/processed/vocab.json",
                        help="Path to vocab.json (default: data/processed/vocab.json)")
    parser.add_argument("--save-dir",       default="data/raw/aslense",
                        help="Where to save downloaded videos (default: data/raw/aslense)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Max videos per word (default: no limit, ~31/word)")
    parser.add_argument("--skip-manifest",  action="store_true",
                        help="Skip manifest generation after download")
    args = parser.parse_args()

    main(
        vocab_path=args.vocab_path,
        save_dir=args.save_dir,
        limit=args.limit,
        skip_manifest=args.skip_manifest)