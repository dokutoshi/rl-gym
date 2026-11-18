import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from nstep import nstep_target
class NStep(unittest.TestCase):
    def test_one_step(self): self.assertAlmostEqual(nstep_target([1],1,.9,10),10)
    def test_mc_end(self): self.assertAlmostEqual(nstep_target([1,2,3],3,.9,999,True),5.23)
    def test_terminal_tail(self): self.assertEqual(nstep_target([2],5,.9,999,True),2)
    def test_intermediate(self): self.assertAlmostEqual(nstep_target([1,2,3],2,.9,10,True),10.9)
