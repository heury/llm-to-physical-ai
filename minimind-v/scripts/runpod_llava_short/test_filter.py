import unittest
from filter_data import simple_pair,sentences

class FilteringTests(unittest.TestCase):
    def test_identification(self):
        self.assertIsNone(simple_pair('What color is the bus in this image?','The bus is red.'))
        self.assertIsNone(simple_pair('How many cats are in the image?','There are two cats. They are sitting together.'))
    def test_complex_reasoning(self):
        for q in ['Why is the man holding a bat?','What might happen next?','What is the purpose of the device?']:
            self.assertIsNotNone(simple_pair(q,'It is used for playing.'))
    def test_long_answers(self):
        self.assertIsNotNone(simple_pair('What color is the bus?','It is red. It is big. It is old.'))
        self.assertIsNotNone(simple_pair('What color is the bus?','red '*41))
    def test_context_dependency(self):
        self.assertIsNotNone(simple_pair('What color is it?','Red.'))
        self.assertIsNone(simple_pair('What color is the bus?','Red.'))
    def test_uncertain_answer(self):
        self.assertIsNotNone(simple_pair('How many people are in the image?', 'I cannot determine the exact number.'))

    def test_speculation(self):
        self.assertIsNotNone(simple_pair('What is the man wearing?','He might be wearing a coat.'))

if __name__=='__main__':unittest.main()
