import unittest
from codec import split_snac
class CodecContract(unittest.TestCase):
 def test_temporal_hierarchy(self):
  self.assertEqual(split_snac('# 1 2 3 4 5 6 7 # 8 9 10 11 12 13 14'), [[1,8],[2,5,9,12],[3,4,6,7,10,11,13,14]])
 def test_reject_corrupt_groups(self):
  for x in ['# 1 2 3', '# 4096 2 3 4 5 6 7', '']:
   with self.assertRaises(ValueError):split_snac(x)
if __name__=='__main__':unittest.main()
