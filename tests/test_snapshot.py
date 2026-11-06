import json
import unittest
import numpy as np
from retrieval import ROOT, metrics, rank

class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=np.load(ROOT/'data/features.npz')
        cls.meta=json.loads((ROOT/'data/metadata.json').read_text())

    def test_index_integrity(self):
        self.assertEqual(len(self.meta['records']),852)
        self.assertEqual(self.data['crops'].shape,(852,512))
        self.assertEqual(self.data['images'].shape,(128,512))
        for field in self.data:
            self.assertTrue(np.isfinite(self.data[field]).all())
            np.testing.assert_allclose(np.linalg.norm(self.data[field],axis=1),1,atol=1e-6)

    def test_cached_bear_ranking(self):
        categories=np.array([r['category'] for r in self.meta['records']])
        scores=self.data['crops']@self.data['texts'][21]
        result=metrics(rank(scores),categories==21)
        self.assertAlmostEqual(result['AP@10'],1)
        parents=[r['parent'] for r in self.meta['records']]
        contrast=scores-.5*(self.data['images']@self.data['texts'][21])[parents]
        self.assertEqual(metrics(rank(contrast),categories==21)['AP@10'],0)

    def test_thumbnails_and_boxes_match_index(self):
        for record in self.meta['records']:
            self.assertTrue((ROOT/f'data/thumbnails/{record["id"]}.jpg').exists())
            x1,y1,x2,y2=record['bbox_xyxy']
            self.assertGreaterEqual(x2-x1,8)
            self.assertGreaterEqual(y2-y1,8)

if __name__=='__main__': unittest.main()
