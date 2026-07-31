"""Download the official OpenAGI multimodal benchmark assets from Google Drive."""

from __future__ import annotations

from pathlib import Path

import gdown


OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/1AjT6y7qLIMxcmHhUBG5IE1_5SnCPR57e"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "OpenAGI" / "official_assets"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(
        url=OFFICIAL_FOLDER,
        output=str(OUTPUT_DIR) + "/",
        quiet=False,
        resume=True,
    )
    if not files:
        raise RuntimeError(
            "OpenAGI download returned no files. Check access to drive.google.com "
            "and rerun this command."
        )
    print(f"Downloaded {len(files)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
