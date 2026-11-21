from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
import unittest

from check_sources import load_sources

DATA = Path(__file__).with_name('data')


class ExchangeTests(unittest.TestCase):
    def test_example_is_readable(self):
        tables = load_sources(DATA)
        self.assertTrue(all(tables.values()))
        for obs in tables['observations']:
            self.assertLessEqual(int(obs['entry_age']), int(obs['last_ok']))
            self.assertLessEqual(int(obs['last_ok']), int(obs['last_seen']))
            if obs['first_failed']:
                self.assertLess(int(obs['last_ok']), int(obs['first_failed']))
                self.assertLessEqual(int(obs['first_failed']), int(obs['last_seen']))
        self.assertTrue(any(r['shipped'] == '1' for r in tables['picks']))
        self.assertTrue(any(int(r['consumed']) < int(r['qty']) for r in tables['historical_orders']))

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / 'copies'
            shutil.copytree(DATA, duplicate)
            path = duplicate / 'inventory.csv'
            rows = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(rows + [rows[1]]) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '标识为空或重复'):
                load_sources(duplicate)
