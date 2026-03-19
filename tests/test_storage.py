import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.storage import StorageManager


class StorageManagerDiskIoRecoveryTests(unittest.TestCase):
    def test_recovers_by_quarantining_corrupt_sqlite_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / 'knowledge.sqlite3'
            checkpoint_dir = root / 'checkpoints'
            db_path.write_text('broken db')
            for suffix in ('-wal', '-shm', '-journal'):
                Path(f'{db_path}{suffix}').write_text('broken sidecar')

            original_initialize = StorageManager._initialize_schema
            calls = {'count': 0}

            def flaky_initialize(manager: StorageManager) -> None:
                calls['count'] += 1
                if calls['count'] == 1:
                    raise sqlite3.OperationalError('disk I/O error')
                original_initialize(manager)

            with patch.object(StorageManager, '_initialize_schema', autospec=True, side_effect=flaky_initialize):
                storage = StorageManager(db_path, checkpoint_dir)

            self.assertEqual(storage._journal_mode, 'DELETE')
            self.assertEqual(storage._temp_store, 'MEMORY')
            self.assertTrue(any(root.glob('knowledge.sqlite3.corrupt-*')))
            self.assertTrue(any(root.glob('knowledge.sqlite3-wal.corrupt-*')))
            self.assertTrue(any(root.glob('knowledge.sqlite3-shm.corrupt-*')))
            self.assertTrue(any(root.glob('knowledge.sqlite3-journal.corrupt-*')))
            self.assertTrue(db_path.exists())
            self.assertGreater(len(storage.search_knowledge('cohete')), 0)

    def test_non_disk_io_errors_are_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'knowledge.sqlite3'
            checkpoint_dir = Path(tmp) / 'checkpoints'
            with patch.object(StorageManager, '_initialize_schema', autospec=True, side_effect=sqlite3.OperationalError('database is locked')):
                with self.assertRaisesRegex(sqlite3.OperationalError, 'database is locked'):
                    StorageManager(db_path, checkpoint_dir)


if __name__ == '__main__':
    unittest.main()
