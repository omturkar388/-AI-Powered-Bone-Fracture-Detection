from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "FracAtlas" / "images"
REPORTS = ROOT / "reports"
CLASSES = ("Fractured", "Non_fractured")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_class(name):
    folder = DATA / name
    if not folder.is_dir():
        raise FileNotFoundError(f"Class folder not found: {folder}")

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    readable = 0
    corrupt = []
    dimensions = Counter()
    formats = Counter()

    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                formats[image.format or "UNKNOWN"] += 1
            readable += 1
        except (OSError, ValueError, UnidentifiedImageError) as error:
            corrupt.append({
                "path": str(path.relative_to(ROOT)),
                "error": str(error),
            })

    return {
        "image_count": len(files),
        "readable_count": readable,
        "corrupt_count": len(corrupt),
        "formats": dict(formats),
        "dimensions": dict(dimensions.most_common()),
        "corrupt_files": corrupt,
    }


def main():
    if not DATA.is_dir():
        raise FileNotFoundError(f"Dataset images folder not found: {DATA}")

    results = {
        "dataset": "FracAtlas",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes": {},
    }

    for name in CLASSES:
        results["classes"][name] = audit_class(name)

    total = sum(v["image_count"] for v in results["classes"].values())
    readable = sum(v["readable_count"] for v in results["classes"].values())
    corrupt = sum(v["corrupt_count"] for v in results["classes"].values())

    results["summary"] = {
        "total_images": total,
        "total_readable": readable,
        "total_corrupt": corrupt,
        "fractured_fraction": (
            results["classes"]["Fractured"]["image_count"] / total
            if total else 0.0
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / "data_audit.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, details in results["classes"].items():
        print(f"\n{name}")
        print(f"  Images: {details['image_count']}")
        print(f"  Readable: {details['readable_count']}")
        print(f"  Corrupt: {details['corrupt_count']}")
        print(f"  Formats: {details['formats']}")

    print("\n=== SUMMARY ===")
    print(f"Total images: {total}")
    print(f"Readable images: {readable}")
    print(f"Corrupt images: {corrupt}")
    print(f"Report saved to: {report}")


if __name__ == "__main__":
    main()
