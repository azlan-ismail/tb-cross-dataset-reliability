"""Synthetic software checks only: these are NOT medical experiment results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import prepare as prep
import experiment as exp


class PipelineChecks(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(123)
        rows = []
        for source, n in [('shenzhen', 60), ('montgomery', 20)]:
            for i in range(n):
                key = f'{source}:{i}'
                rows.append(dict(image_id=key, patient_id=key, source=source,
                    label=i % 2, path=key, sha256=key.replace(':', '_'),
                    pixel_sha256=key, dhash=rng.bytes(8).hex()))
        return rows

    def test_grouped_holdout(self):
        rows = self.rows()
        # Simulated duplicate between development and test must be excluded.
        rows[0].update({k: rows[-2][k] for k in ['sha256', 'pixel_sha256', 'dhash']})
        prep.duplicate_groups(rows)
        result = prep.make_split(rows)
        self.assertEqual(result[0]['split'], 'excluded_overlap')
        self.assertTrue(all(r['split'] == 'test' for r in result if r['source'] == 'montgomery'))
        self.assertEqual(result, prep.make_split(rows))
        broken = copy.deepcopy(result)
        train = next(r for r in broken if r['split'] == 'train')
        val = next(r for r in broken if r['split'] == 'validation')
        val['group_id'] = train['group_id']
        with self.assertRaises(ValueError): prep.validate_split(broken)

    def test_similarity_is_not_identity(self):
        rows = self.rows()[:2]
        rows[1]['dhash'] = rows[0]['dhash']
        candidates = prep.duplicate_groups(rows)
        self.assertEqual(len(candidates), 1)
        self.assertNotEqual(rows[0]['group_id'], rows[1]['group_id'])

    def test_conflicting_duplicate_labels(self):
        rows = self.rows()
        rows[1]['pixel_sha256'] = rows[0]['pixel_sha256']
        with self.assertRaises(ValueError): prep.duplicate_groups(rows)

    def test_known_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .4, .35, .8])
        m = exp.metric_values(y, p, .5)
        self.assertAlmostEqual(m['auroc'], .75)
        self.assertEqual(m['sensitivity'], .5)
        self.assertEqual(m['specificity'], 1.)
        self.assertAlmostEqual(exp.select_threshold(y, p), .35)

    def test_training_evaluation_and_lock(self):
        with tempfile.TemporaryDirectory() as d:
            study = Path(d)
            rows = self.rows()
            prep.duplicate_groups(rows)
            rows = prep.make_split(rows)
            prep.write_csv(study/'manifest.csv', rows)
            prep.write_json(study/'protocol.json', {'manifest_sha256': prep.digest(study/'manifest.csv'),
                'test_source': 'montgomery'})
            features = study/'features'
            features.mkdir()
            prep.write_json(features/'metadata.json', {'synthetic_qa_only': True})
            rng = np.random.default_rng(456)
            for row in rows:
                x = rng.normal(size=16).astype(np.float32)
                x[0] += int(row['label']) * 2
                np.save(features/f"{row['sha256']}.npy", x, allow_pickle=False)
            exp.fit(study, seeds=(42,), epochs=2)
            exp.evaluate(study, repeats=100)
            self.assertTrue((study/'results'/'summary.csv').exists())
            with self.assertRaises(FileExistsError): exp.fit(study, seeds=(42,), epochs=2)
            with self.assertRaises(FileExistsError): exp.evaluate(study, repeats=100)
            with open(study/'manifest.csv', 'a') as f: f.write('\n')
            with self.assertRaises(ValueError): exp.locked_manifest(study)


if __name__ == '__main__':
    unittest.main()
