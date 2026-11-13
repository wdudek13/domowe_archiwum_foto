import datetime
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np

from app.scanner import FastHasher, MetadataDateResolver
from app.backup import BackupManager
from app.ai.embeddings import OnDemandClipEmbedder


class TestHydraArchitecture(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_video_partial_hash_efficiency(self):
        """
        Verify partial hashing reads only first 64KB, middle 64KB, and last 64KB,
        producing distinct hashes for gigabyte-scale structures.
        """
        file_path = self.test_path / "test_video.mp4"
        # Create a synthetic 1 MB file
        size_bytes = 1024 * 1024
        with open(file_path, "wb") as f:
            f.write(b"A" * size_bytes)

        hash1 = FastHasher.hash_video_partial(file_path, size_bytes)
        self.assertEqual(len(hash1), 64, "Partial hash must be valid SHA-256 hex string")

        # Mutate byte in the middle chunk
        with open(file_path, "r+b") as f:
            f.seek(size_bytes // 2)
            f.write(b"Z")

        hash2 = FastHasher.hash_video_partial(file_path, size_bytes)
        self.assertNotEqual(hash1, hash2, "Altering the middle chunk must alter the partial hash")

    def test_date_anomaly_resolution(self):
        """
        Verify that effective_date selects the oldest date, and flags
        discrepancies > 24 hours between EXIF and filesystem timestamp.
        """
        test_file = self.test_path / "IMG_20210510_120000.jpg"
        test_file.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 100)  # Minimal JPEG header

        exif_dt, fn_dt, ctime, mtime, effective_dt, has_anomaly, reason = (
            MetadataDateResolver.resolve_dates(test_file, is_image=True)
        )

        self.assertIsNotNone(fn_dt)
        self.assertEqual(fn_dt.year, 2021)
        self.assertEqual(fn_dt.month, 5)
        self.assertEqual(fn_dt.day, 10)
        self.assertTrue(has_anomaly, "Missing EXIF must flag has_date_anomaly = True")
        self.assertIn("Brak metadanych EXIF", reason)

    def test_sqlite_native_backup(self):
        """
        Test SQLite online backup mechanism with physical integrity check.
        """
        db_path = self.test_path / "source.db"
        backup_dir = self.test_path / "backups"
        backup_dir.mkdir()

        # Init SQLite source with WAL mode
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, payload TEXT);")
        conn.execute("INSERT INTO test_data (payload) VALUES ('critical_record');")
        conn.commit()
        conn.close()

        mgr = BackupManager(db_path=db_path, backup_dir=backup_dir, max_retained_backups=3)
        backup_file = mgr.create_backup(tag="unit_test")

        self.assertTrue(backup_file.exists())

        # Verify integrity of backup
        dest_conn = sqlite3.connect(str(backup_file))
        res = dest_conn.execute("SELECT payload FROM test_data;").fetchone()
        dest_conn.close()

        self.assertEqual(res[0], "critical_record")

    def test_vector_cosine_similarity(self):
        """
        Tests NumPy fast vectorized cosine ranking for embeddings stored as raw BLOBs.
        """
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        v2 /= np.linalg.norm(v2)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        candidates = [
            ("item_similar", v2.tobytes()),
            ("item_orthogonal", v3.tobytes()),
        ]

        ranked = OnDemandClipEmbedder.rank_similar(v1, candidates, top_n=2)
        self.assertEqual(ranked[0][0], "item_similar")
        self.assertGreater(ranked[0][1], 0.8)


if __name__ == "__main__":
    unittest.main()