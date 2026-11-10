"""No real population selection in these tests."""
import unittest
import pandas as pd
from aurum.development_data import choose
from aurum.confirmation_data import SOURCES


class SelectionTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([{'source': source, 'cluster_id': f'{source}_{i}', 'item_id': f'{source}_{i}'}
                             for source in SOURCES for i in range(12)])

    def test_balanced_deterministic_metadata_selection(self):
        rows = self.rows()
        chosen = choose(rows, 8, 71, reserve_per_source=8)
        self.assertEqual(chosen.groupby('source').size().to_dict(), {s: 4 for s in SOURCES})
        self.assertEqual(chosen.item_id.tolist(), choose(rows.sample(frac=1), 8, 71, 8).item_id.tolist())

    def test_reserve_protection(self):
        with self.assertRaisesRegex(ValueError, 'confirmation reserve'):
            choose(self.rows(), 10, 71, reserve_per_source=8)


if __name__ == '__main__':
    unittest.main()
