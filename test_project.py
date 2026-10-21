import unittest
from project import *
class ProjectTests(unittest.TestCase):
    def test_training_reduces_error(self):
        pairs,r=generate();m=fit(pairs[:600],r[:600],epochs=35)
        self.assertLess(float(np.mean((predict(m,pairs[:600])-r[:600])**2)),float(np.mean((r[:600].mean()-r[:600])**2)))
    def test_excludes_seen(self):
        pairs,r=generate();m=fit(pairs[:200],r[:200],epochs=2);seen={0,1,2}
        self.assertFalse(seen & {x['item'] for x in recommend(m,0,seen)})
    def test_no_candidates(self):
        pairs,r=generate();m=fit(pairs[:200],r[:200],epochs=2);self.assertEqual(recommend(m,0,set(range(40))),[])
    def test_unique_pairs(self):
        pairs,_=generate();self.assertEqual(len(set(map(tuple,pairs))),len(pairs))
