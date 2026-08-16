"""Unit tests for Hugging Face bio_entities.db ensure/download."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemtables.catalog import HF_DATASET_ID, HF_DB_FILENAME, ensure_bio_entities_db


class TestEnsureBioEntitiesDb(unittest.TestCase):
    def test_existing_file_skips_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / HF_DB_FILENAME
            dest.write_bytes(b"local-db")
            with patch("chemtables.catalog.hf_hub_download") as download:
                result = ensure_bio_entities_db(dest)
            download.assert_not_called()
            self.assertEqual(result, dest)
            self.assertEqual(dest.read_bytes(), b"local-db")

    def test_missing_file_downloads_to_dest(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nested" / HF_DB_FILENAME

            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], HF_DATASET_ID)
                self.assertEqual(kwargs["filename"], HF_DB_FILENAME)
                self.assertEqual(kwargs["repo_type"], "dataset")
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                out = local_dir / HF_DB_FILENAME
                out.write_bytes(b"hf-db")
                return str(out)

            with patch(
                "chemtables.catalog.hf_hub_download", side_effect=fake_download
            ) as download:
                result = ensure_bio_entities_db(dest)
            download.assert_called_once()
            self.assertEqual(result, dest)
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), b"hf-db")

    def test_download_failure_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / HF_DB_FILENAME
            with patch(
                "chemtables.catalog.hf_hub_download",
                side_effect=OSError("offline"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_bio_entities_db(dest)
            self.assertIn(HF_DATASET_ID, str(ctx.exception))
            self.assertFalse(dest.exists())
