import unittest
import corpus


class TestCorpus(unittest.TestCase):

    # @unittest.skip("temp")
    def test_Statquest(self):
        c = corpus.Corpus(corpus.Statquest_inputs.token_to_id,
                          corpus.Statquest_inputs.input_strings)
        self.assertEqual(c.vocab_size, 11)

    def test_token_dict_from_tokens(self):
        tokens = ['a', 'b', 'c', 'a']
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
                     
                     a b c'''
        token_to_id = corpus.Corpus.token_dict_from_string(token_str)
        self.assertEqual(len(token_to_id), 12)
        self.assertIn('<PAD>', token_to_id)
        input_strings = ['a b c', 'a', 'b c golf']
        c = corpus.Corpus(input_strings=input_strings,
                          token_str=token_str)
        self.assertEqual(c.vocab_size, 12)

    def test_token_dict_from_inputs(self):
        input_strings = ['a b c', 'a', 'b c golf']
        token_to_id = corpus.Corpus.token_dict_from_inputs(input_strings)
        self.assertEqual(len(token_to_id), 6)
        self.assertIn('<PAD>', token_to_id)
        c = corpus.Corpus(input_strings=input_strings)
        self.assertEqual(c.vocab_size, 6)

    def test_Synonym_inputs(self):
        si = corpus.Synonym_inputs
        inputs, labels = si.make_inputs(num_inputs=4, seed=54545)
        for input, label in zip(inputs, labels):
            words = input.split()
            self.assertIn(label, words[:-1])
            self.assertTrue(si.are_synonyms(label, words[-1]))


if __name__ == '__main__':
    unittest.main()
