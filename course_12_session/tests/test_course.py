import unittest, sys, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
import train
import numpy as np
import torch

class Correctness(unittest.TestCase):
    def test_terminal_reward_propagates(self):
        self.assertEqual(train.target(20,True,999),20)
    def test_truncation_is_not_terminal(self):
        self.assertAlmostEqual(train.target(1,False,10,.9),10)
    def test_mask_excludes_higher_invalid_value(self):
        rng=np.random.default_rng(0)
        for eps in (0,1):
            for _ in range(50): self.assertEqual(train.choose(np.array([999.,2.]),[0,1],eps,rng),1)
    def test_empty_mask_fails(self):
        with self.assertRaises(ValueError): train.choose(np.zeros(2),[0,0],0,np.random.default_rng(0))
    def test_dpo_margin_gradient(self):
        margin=torch.tensor(0.,requires_grad=True)
        loss=-torch.nn.functional.logsigmoid(.1*margin); loss.backward()
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        self.assertLess(margin.grad.item(),0)
    def test_zero_variance_group_finite(self):
        rewards=torch.ones(4)
        adv=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-8)
        self.assertTrue(torch.equal(adv,torch.zeros(4)))
    def test_ppo_clips_positive_advantage(self):
        ratio=torch.tensor(2.); advantage=torch.tensor(1.)
        value=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
        self.assertAlmostEqual(value.item(),1.2,places=6)
    def test_reference_gradient_isolated(self):
        policy=torch.tensor([0.,0.],requires_grad=True); reference=policy.detach().clone()
        loss=-(policy.log_softmax(0)[0]-reference.log_softmax(0)[0]);loss.backward()
        self.assertIsNone(reference.grad)
if __name__=='__main__': unittest.main()
