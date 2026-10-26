import unittest
import torch
from run import losses,encode
from common import extract
class DistillTest(unittest.TestCase):
 def test_mask_and_shift(self):
  torch.manual_seed(7)
  s=torch.randn(2,5,9,requires_grad=True);t=torch.randn_like(s);y=torch.tensor([[-100,-100,3,4,-100],[-100,2,-100,1,0]])
  ce,kl,n=losses(s,t,y);self.assertEqual(int(n),5)
  mask=y[:,1:]!=-100
  expected=torch.nn.functional.cross_entropy(s[:,:-1][mask],y[:,1:][mask],reduction='sum')
  torch.testing.assert_close(ce,expected)
  ((ce+kl)/n).backward();self.assertEqual(float(s.grad[:,-1].abs().sum()),0)
  self.assertEqual(float(s.grad[0,0].abs().sum()),0)
 def test_identity_kl(self):
  s=torch.randn(2,3,7);y=torch.tensor([[-100,1,2],[-100,3,4]])
  _,kl,_=losses(s,s,y);self.assertLess(abs(float(kl)),2e-6)
 def test_answer_extraction(self):
  self.assertEqual(extract('reason 3\n#### 1,200'),'1.2E+3')
  self.assertIsNone(extract('answer is 42'))
  self.assertEqual(extract('answer is 42',False),'42')
if __name__=='__main__':unittest.main()
