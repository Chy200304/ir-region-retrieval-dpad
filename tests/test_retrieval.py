import unittest
import numpy as np
from retrieval import metrics, rank, score

class RetrievalTests(unittest.TestCase):
    def test_metrics_multiple_relevant(self):
        result=metrics([0,1,2,3],[False,True,False,True],k=3)
        self.assertEqual(result['Hit@1'],0)
        self.assertEqual(result['Recall@10'],0.5)
        self.assertEqual(result['AP@10'],0.25)
        self.assertEqual(result['MRR'],0.5)

    def test_stable_ties(self):
        np.testing.assert_array_equal(rank([0.2,0.8,0.8]),[1,2,0])

    def test_same_image_context_cannot_change_order(self):
        crop=np.array([0.3,0.8,0.1])
        np.testing.assert_array_equal(rank(crop),rank(score(crop,np.full(3,0.6),1)))

    def test_cross_image_context_can_change_order(self):
        np.testing.assert_array_equal(rank(score(np.array([0.8,0.7]),np.array([0.9,0.1]),0.5)),[1,0])

    def test_no_positives_is_invalid(self):
        with self.assertRaises(ValueError): metrics([0],[False])

if __name__=='__main__': unittest.main()
