"""Ensure bio_entities.db is present; download from Hugging Face if missing."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_DATASET_ID = "EstevaoNaval/chemtables"
HF_DB_FILENAME = "bio_entities.db"


def ensure_bio_entities_db(dest: Path) -> Path:
    """Return `dest`, downloading from HF when the file is missing.

    First call needs network. Later calls reuse `dest` and stay offline.
    """
    dest = Path(dest)
    if dest.is_file():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=HF_DB_FILENAME,
            repo_type="dataset",
            local_dir=str(dest.parent),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to download {HF_DB_FILENAME} from Hugging Face "
            f"dataset {HF_DATASET_ID} to {dest}"
        ) from exc

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        shutil.copy2(downloaded_path, dest)
    if not dest.is_file():
        raise RuntimeError(
            f"Hugging Face download did not produce catalog file: {dest}"
        )
    return dest
