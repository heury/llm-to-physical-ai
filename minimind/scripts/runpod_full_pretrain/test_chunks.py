import unittest
from common import full_chunks,chunk_count,split_for
class ChunksTests(unittest.TestCase):
 def test_drop_tail_and_preserve_transitions(self):
  for length in [0,1,339,340,341,678,679,680,2001]:
   ids=list(range(length));chunks=list(full_chunks(ids));self.assertEqual(len(chunks),chunk_count(length))
   self.assertTrue(all(len(x)==340 for x in chunks))
   actual=[(c[i-1],c[i]) for c in chunks for i in range(1,len(c))]
   expected=list(zip(ids,ids[1:]))[:len(chunks)*339]
   self.assertEqual(actual,expected)
 def test_duplicate_text_same_split(self):
  for text in ['Hello','안녕하세요','', 'long '*100]:self.assertEqual(split_for(text),split_for(text))
if __name__=='__main__':unittest.main()
