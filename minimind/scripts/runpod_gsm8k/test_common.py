"""Validate full-answer coverage and scoring edge cases before a paid run."""
import unittest
from common import extract,windows
class MathDataTests(unittest.TestCase):
 def test_each_assistant_token_once_across_windows(self):
  for length in [110,340,341,512,596,700,1200]:
   for assistant_start in [1,100,length-5]:
    tokens=list(range(length));labels=[-100]*assistant_start+tokens[assistant_start:]
    chunks=windows(tokens,labels)
    actual=[y for x,ys in chunks for y in ys[1:] if y!=-100]
    self.assertEqual(actual,tokens[assistant_start:])
    self.assertTrue(all(len(x)==len(y)==340 for x,y in chunks))
 def test_numeric_scoring(self):
  self.assertEqual(extract('work\n#### 1,234.00'),'1234')
  self.assertEqual(extract('#### -2.5'),'-2.5')
  self.assertIsNone(extract('I think it is 56.'))
  self.assertEqual(extract('I think it is 56.',False),'56')
  self.assertEqual(extract('#### 2\n#### 3'),'3')
  self.assertIsNone(extract('#### apple'))
if __name__=='__main__':unittest.main()
