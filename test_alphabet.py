import unittest
import basic_transformer as bt
import corpus
import alphabet


class TestAlphabet(unittest.TestCase):

    def test_learn_alphabet_var_embedding(self):
        corp = alphabet.get_Alphabet_corpus()
        btp = bt.BasicTransformerParams()
        btp.batch_size = 1
        btp.num_attn_heads = 1
        btp.attn_head_config = 'single'
        vocab_size = len(corp.id_to_token)
        btp.d_model = vocab_size
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        self.assertEqual(response_errs, 0)
        bt.print_some_predictions(corp, model)
