"""Real-tokenizer tests: full answer coverage, EOS and image-preserving windows."""
import os, unittest
from transformers import AutoTokenizer
from common import windows, encode, messages, LENGTH

class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=AutoTokenizer.from_pretrained(os.environ['LLAVA_TEST_TOKENIZER'],local_files_only=True)
    def check_row(self,row):
        tok=self.tok;msgs=messages(row);expected=[]
        for i,m in enumerate(msgs):
            if m['role']=='assistant':expected.extend(encode(tok,msgs[:i+1])[len(encode(tok,msgs[:i],True)):])
        result=list(windows(tok,row));actual=[]
        for x,y in result:
            self.assertEqual(len(x),LENGTH);self.assertEqual(len(y),LENGTH);self.assertEqual(x.count(12),64)
            self.assertTrue(all(t==-100 for t,v in zip(y,x) if v==12))
            actual.extend(t for t in y if t!=-100)
        self.assertEqual(actual,expected)
        self.assertGreaterEqual(actual.count(tok.eos_token_id),sum(m['role']=='assistant' for m in msgs))
        return result
    def test_short_answer_keeps_eos(self):
        result=self.check_row({'conversations':[{'from':'human','value':'<image>\nWhat color?'},{'from':'gpt','value':'Red.'}]})
        self.assertEqual(len(result),1)
    def test_long_multiturn_every_target_exactly_once(self):
        row={'conversations':[{'from':'human','value':'Describe this.\n<image>'},{'from':'gpt','value':'A red bird is on a tree. '*300},{'from':'human','value':'What is behind it?'},{'from':'gpt','value':'The blue sky. '*100}]}
        self.assertGreater(len(self.check_row(row)),2)
    def test_missing_image_rejected(self):
        with self.assertRaises(AssertionError):
            list(windows(self.tok,{'conversations':[{'from':'human','value':'Hello'},{'from':'gpt','value':'Hi'}]}))

if __name__=='__main__':unittest.main()
