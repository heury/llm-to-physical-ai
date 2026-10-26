"""Boundary and EOS preservation tests without importing the GPU model."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from common import window_bounds,assistant_spans,canonical,prompt_key,read_json
class SFTPreparationTests(unittest.TestCase):
 def test_windows_cover_every_shifted_target_once(self):
  for length in [2,339,340,341,512,596,700,2001]:
   actual=[];covered=1
   for start,end in window_bounds(length):
    begin=max(start+1,covered);actual.extend(range(begin,end));covered=max(covered,end)
   self.assertEqual(actual,list(range(1,length)))
 def test_partial_json_is_retried(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'ready.json';path.write_text('{"complete":')
   with patch('time.sleep',side_effect=lambda _:path.write_text('{"complete":true}')):
    self.assertEqual(read_json(path,retries=2),{'complete':True})
 def test_permanent_invalid_json_is_not_silently_accepted(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'bad.json';path.write_text('{bad')
   import json
   with self.assertRaises(json.JSONDecodeError):read_json(path,retries=1)
 def test_short_final_window_is_kept(self):
  self.assertEqual(list(window_bounds(341)),[(0,340),(256,341)])
 def test_prompt_normalization(self):
  self.assertEqual(canonical(' What  is 2 + 3?\n'),'what is 2 + 3?')
  self.assertEqual(prompt_key([{'role':'user','content':'Hello!'}]),prompt_key([{'role':'user','content':' hello! '}]))
if __name__=='__main__':unittest.main()
