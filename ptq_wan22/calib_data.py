from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _progress(iterable, *, desc: str, total: int | None = None, enabled: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        def _logged():
            for idx, item in enumerate(iterable, start=1):
                if idx == 1 or idx % 10 == 0 or (total is not None and idx == total):
                    suffix = f"/{total}" if total is not None else ""
                    print(f"[{desc}] {idx}{suffix}")
                yield item

        return _logged()


@dataclass(frozen=True)
class CalibrationItem:
    image: Path
    prompt: str
    source_video: Path | None = None
    source_path: str | None = None


def _prompt_from_sidecar(image: Path) -> str:
    for suffix in (".txt", ".prompt", ".caption"):
        sidecar = image.with_suffix(suffix)
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text
    return (
        "Generate a high-quality video that preserves the subject and scene "
        "from the reference image with natural motion."
    )


def _candidate_paths(roots: tuple[Path, ...], value: str) -> list[Path]:
    path = Path(value)
    candidates = [path] if path.is_absolute() else []

    relative_candidates = [path] if not path.is_absolute() else []
    parts = path.parts
    for marker in ("OpenS2V-5M", "datasets"):
        if marker in parts:
            idx = parts.index(marker)
            relative_candidates.append(Path(*parts[idx + 1 :]))
    if "total_part2" in parts:
        idx = parts.index("total_part2")
        relative_candidates.append(Path(*parts[idx:]))

    for rel in relative_candidates:
        for root in roots:
            candidates.append(root / rel)
    return candidates


def _resolve_media(roots: tuple[Path, ...], value: Any, suffixes: set[str]) -> Path | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, str):
        return None
    for candidate in _candidate_paths(roots, value):
        if candidate.exists() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
        if candidate.exists() and candidate.suffix.lower() in suffixes:
            return candidate
    return None


def _extract_video_frame(video: Path, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    key = sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:16]
    out = cache_root / f"{video.stem}_{key}.jpg"
    if out.exists():
        return out

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(out), frame)
            if out.exists():
                return out
    except Exception:
        pass

    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(video, index=0)
        from PIL import Image

        Image.fromarray(frame).save(out)
        if out.exists():
            return out
    except Exception:
        pass

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )
    return out


def _item_from_mapping(
    roots: tuple[Path, ...],
    item: dict[str, Any],
    cache_root: Path,
) -> CalibrationItem | None:
    image = None
    for key in ("image", "image_path", "img", "img_path", "img_paths", "images"):
        if key in item:
            image = _resolve_media(roots, item[key], IMAGE_SUFFIXES)
            if image is not None:
                break
    if image is None:
        for key in ("video", "video_path", "path", "mp4"):
            if key in item:
                video = _resolve_media(roots, item[key], VIDEO_SUFFIXES)
                if video is not None and video.suffix.lower() in VIDEO_SUFFIXES:
                    image = _extract_video_frame(video, cache_root)
                    source_video = video
                    source_path = item[key] if isinstance(item[key], str) else str(video)
                    break
        else:
            source_video = None
            source_path = None
    else:
        source_video = None
        source_path = None
    if image is None:
        return None

    prompt = None
    for key in ("prompt", "caption", "cap", "text", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    return CalibrationItem(
        image=image,
        prompt=prompt or _prompt_from_sidecar(image),
        source_video=source_video,
        source_path=source_path,
    )


def _iter_json_items(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    if isinstance(data, list):
        return data
    return []


def load_calibration_items(
    root: str | Path,
    limit: int | None = None,
    show_progress: bool = True,
) -> list[CalibrationItem]:
    """Load I2V calibration image/prompt pairs from a dataset directory.

    Supported layouts:
    - JSON/JSONL files with `image`/`img_path`/`img_paths` and `prompt`/`caption`.
    - Image files with optional same-name `.txt`, `.prompt`, or `.caption` files.
    """

    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Calibration dataset root does not exist: {dataset_root}")

    items: list[CalibrationItem] = []
    if dataset_root.is_file():
        meta_files = [dataset_root]
        search_root = dataset_root.parent
    else:
        meta_files = sorted(dataset_root.rglob("*.json")) + sorted(dataset_root.rglob("*.jsonl"))
        search_root = dataset_root

    json_seen = 0
    json_loaded = 0
    for meta in _progress(meta_files, desc="calib metadata", enabled=show_progress):
        try:
            json_items = _iter_json_items(meta)
            for obj in _progress(
                json_items,
                desc=f"calib items {meta.name}",
                total=len(json_items),
                enabled=show_progress,
            ):
                json_seen += 1
                if isinstance(obj, dict):
                    item = _item_from_mapping(
                        (meta.parent, search_root),
                        obj,
                        search_root / ".calib_frames",
                    )
                    if item is not None:
                        items.append(item)
                        json_loaded += 1
                        if limit is not None and len(items) >= limit:
                            if show_progress:
                                print(
                                    f"Loaded {json_loaded}/{json_seen} calibration items "
                                    "from JSON records before reaching the limit."
                                )
                            return items
        except Exception:
            continue
    if show_progress and json_seen:
        print(
            f"Loaded {json_loaded}/{json_seen} calibration items from JSON records; "
            f"skipped {json_seen - json_loaded} records with missing/unsupported media."
        )

    seen = {item.image.resolve() for item in items}
    if dataset_root.is_file():
        image_files = []
    else:
        image_files = sorted(
            p
            for p in dataset_root.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES and ".calib_frames" not in p.parts
        )
    for image in _progress(
        image_files,
        desc="calib image files",
        total=len(image_files),
        enabled=show_progress,
    ):
        resolved = image.resolve()
        if resolved in seen:
            continue
        items.append(CalibrationItem(image=image, prompt=_prompt_from_sidecar(image)))
        if limit is not None and len(items) >= limit:
            return items

    if not items:
        raise ValueError(
            f"No calibration image/prompt pairs found under {dataset_root}. "
            "Expected JSON/JSONL image entries or image files."
        )
    return items
