import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from package import copy_contents
class GlobalVolumeCopy(unittest.TestCase):
 def test_does_not_require_metadata_operations(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'src';dst=Path(d)/'dst';src.mkdir();(src/'sub').mkdir();(src/'sub/model.bin').write_bytes(b'weight-bytes');(src/'.cache').mkdir();(src/'.cache/temp').write_text('skip')
   with patch('shutil.copystat',side_effect=PermissionError('global volume does not support chmod/utime')):copy_contents(src,dst)
   self.assertEqual((dst/'sub/model.bin').read_bytes(),b'weight-bytes');self.assertFalse((dst/'.cache').exists())
if __name__=='__main__':unittest.main()
