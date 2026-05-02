"""
Convert the original AVE dataset (videos + split files) into the simplified
multimodal format used by our experiments:

    data/ave/
        images/<sample>.jpg
        audio/<sample>.wav
        texts/<sample>.txt
        index.json

Each entry in index.json contains:
    {
        "id": "...",
        "split": "train|val|test",
        "image": "images/....jpg",
        "audio": "audio/....wav",
        "text": "texts/....txt",
        "label": <int>
    }

Requirements:
    - FFmpeg must be installed and available in PATH.
    - The original dataset should be extracted to:
        data/AVE_raw/AVE_Dataset/
          ├─ AVE/            (contains *.mp4 files)
          ├─ trainSet.txt
          ├─ valSet.txt
          └─ testSet.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

OUTPUT_ROOT = Path("data/ave")
RAW_ROOT = Path("data/AVE_raw/AVE_Dataset")
VIDEO_DIR = RAW_ROOT / "AVE"

SPLIT_FILES = {
    "train": RAW_ROOT / "trainSet.txt",
    "val": RAW_ROOT / "valSet.txt",
    "test": RAW_ROOT / "testSet.txt",
}


@dataclass
class SampleEntry:
    class_name: str
    video_id: str
    split: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.5, self.end - self.start)

    def mid_time(self) -> float:
        return self.start + self.duration / 2.0


def parse_split_file(path: Path, split: str) -> List[SampleEntry]:
    entries: List[SampleEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("&")
        if len(parts) < 5:
            continue
        class_name, video_id, _, start, end = parts[:5]
        try:
            start_f = float(start)
            end_f = float(end)
        except ValueError:
            continue
        if end_f <= start_f:
            end_f = start_f + 2.0  # ensure positive duration

        entries.append(
            SampleEntry(
                class_name=class_name.strip(),
                video_id=video_id.strip(),
                split=split,
                start=start_f,
                end=end_f,
            )
        )
    return entries


def ensure_dirs(root: Path) -> Dict[str, Path]:
    subdirs = {}
    for name in ["images", "audio", "texts"]:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def run_ffmpeg(args: List[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    subprocess.run(cmd, check=True)


def build_index(entries: List[Dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def main(output_root: Path = OUTPUT_ROOT) -> None:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    for split, file in SPLIT_FILES.items():
        if not file.exists():
            raise FileNotFoundError(f"Split file missing: {file}")

    subdirs = ensure_dirs(output_root)
    records: List[Dict] = []

    print("[INFO] Parsing split files...")
    split_entries: Dict[str, List[SampleEntry]] = {}
    class_to_idx: Dict[str, int] = {}

    for split, file in SPLIT_FILES.items():
        entries = parse_split_file(file, split)
        split_entries[split] = entries
        for entry in entries:
            if entry.class_name not in class_to_idx:
                class_to_idx[entry.class_name] = len(class_to_idx)

    print(f"[INFO] Found {len(class_to_idx)} unique classes.")

    total_entries = sum(len(v) for v in split_entries.values())
    print(f"[INFO] Preparing {total_entries} samples...")

    for split, entries in split_entries.items():
        for idx, entry in enumerate(entries, 1):
            video_path = VIDEO_DIR / f"{entry.video_id}.mp4"
            if not video_path.exists():
                print(f"[WARN] Missing video {video_path}, skipping.")
                continue

            sample_id = f"{split}_{idx:05d}"
            image_path = subdirs["images"] / f"{sample_id}.jpg"
            audio_path = subdirs["audio"] / f"{sample_id}.wav"
            text_path = subdirs["texts"] / f"{sample_id}.txt"

            try:
                # Extract representative frame
                run_ffmpeg([
                    "-ss", f"{entry.mid_time():.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    str(image_path),
                ])

                # Extract audio segment
                run_ffmpeg([
                    "-ss", f"{entry.start:.3f}",
                    "-t", f"{entry.duration:.3f}",
                    "-i", str(video_path),
                    "-ac", "1",
                    "-ar", "16000",
                    str(audio_path),
                ])
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] ffmpeg failed for {sample_id}: {exc}")
                for path in [image_path, audio_path]:
                    if path.exists():
                        path.unlink()
                continue

            text_content = (
                f"This {split} sample corresponds to class {entry.class_name} "
                f"from video {entry.video_id}. "
                f"The event occurs between {entry.start:.2f}s and {entry.end:.2f}s."
            )
            text_path.write_text(text_content, encoding="utf-8")

            records.append({
                "id": sample_id,
                "split": split,
                "image": f"images/{image_path.name}",
                "audio": f"audio/{audio_path.name}",
                "text": f"texts/{text_path.name}",
                "label": class_to_idx[entry.class_name],
                "class_name": entry.class_name,
            })

        print(f"[INFO] Processed {len(entries)} entries for split '{split}'.")

    build_index(records, output_root / "index.json")
    print(f"[INFO] Completed. Total usable samples: {len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AVE multimodal dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to store processed samples (default: data/ave)",
    )
    args = parser.parse_args()
    main(args.output_dir)
