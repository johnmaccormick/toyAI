import unittest
import corpus


class TestCorpus(unittest.TestCase):

    # @unittest.skip("temp")
    def test_Statquest(self):
        c = corpus.Corpus(corpus.Statquest_inputs.token_to_id,
                          corpus.Statquest_inputs.input_strings)
        self.assertEqual(c.vocab_size, 11)

    def test_token_dict_from_tokens(self):
        tokens = ['a', 'b', 'c']
        token_to_id = corpus.Corpus.token_dict_from_tokens(tokens)
        self.assertEqual(len(token_to_id), 5)
        self.assertIn('<PAD>', token_to_id)
        input_strings = ['a b c', 'a', 'b c']
        c = corpus.Corpus(token_to_id, input_strings)
        self.assertEqual(c.vocab_size, 5)

    def test_token_dict_from_string(self):
        token_str = '''
        
          a b  c  
        d elephant fox   
             golf h i j
                     '''
        token_to_id = corpus.Corpus.token_dict_from_string(token_str)
        self.assertEqual(len(token_to_id), 12)
        self.assertIn('<PAD>', token_to_id)
        input_strings = ['a b c', 'a', 'b c golf']
        c = corpus.Corpus(token_to_id, input_strings)
        self.assertEqual(c.vocab_size, 12)


if __name__ == '__main__':
    unittest.main()
