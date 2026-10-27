import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from data import normalize,examples
from transformers import AutoTokenizer

class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=AutoTokenizer.from_pretrained('/tmp/agent-base-model')
        cls.tok.chat_template="{% filter replace('<think>\\n\\n</think>\\n\\n', '') %}"+cls.tok.chat_template+'{% endfilter %}'
    def row(self):
        return {'id':'test','tools':json.dumps([{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}]),'conversations':[
            {'from':'system','value':'ignored duplicate schema'}, {'from':'human','value':'Calculate.'},
            {'from':'gpt','value':'<tool_call>{"name":"f","arguments":{}}</tool_call>'},
            {'from':'tool','value':'<tool_response>{"result":703}</tool_response>'},
            {'from':'gpt','value':'The answer is 703.'}]}
    def test_both_assistant_targets_and_tool_input_mask(self):
        r=normalize(self.row());xs,n=examples(r,self.tok)
        self.assertEqual((len(xs),n),(2,0));self.assertTrue(xs[1]['post_tool'])
        self.assertIn('703',self.tok.decode(xs[1]['prompt']))
        for x in xs:
            self.assertTrue(all(v==-100 for v in x['labels'][:len(x['prompt'])]))
            self.assertIn(self.tok.eos_token_id,x['labels'][len(x['prompt']):])
        self.assertNotIn('<tool_response>',r['messages'][3]['content'])
    def test_overlength_target_not_silently_truncated(self):
        xs,n=examples(normalize(self.row()),self.tok,max_length=2)
        self.assertEqual(xs,[]);self.assertEqual(n,2)
    def test_same_question_same_split(self):
        a=self.row();b=self.row();b['id']='other';b['conversations'][-1]['value']='Different answer.'
        self.assertEqual(normalize(a)['validation'],normalize(b)['validation'])
    def test_invalid_json_is_not_guessed(self):
        r=self.row();r['conversations'][2]['value']="<tool_call>{'name': 'f'}</tool_call>"
        with self.assertRaises(ValueError):normalize(r)

if __name__=='__main__':unittest.main()
