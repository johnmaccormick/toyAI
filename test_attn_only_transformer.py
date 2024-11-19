import unittest
import basic_transformer as bt
import corpus
import alphabet
import attn_only_transformer as aot


class TestAttnOnlyTransformer(unittest.TestCase):

    def test_learn_attn_only(self):
        corp = alphabet.get_Alphabet_corpus()
        btp = bt.BasicTransformerParams()
        btp.batch_size = 1
        vocab_size = len(corp.id_to_token)
        btp.d_model = vocab_size
        btp.d_qk = vocab_size
        btp.d_vo = vocab_size
        model, optimizer, dataloader = bt.create_model(
            btp, corp, aot.AttnOnlyTransformer)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(avg_loss, 1.44293, places=5)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        self.assertEqual(response_errs, 3)
        print(f'avg_loss {avg_loss}, response_errs {response_errs}')

    def test_learn_attnUnembed(self):
        corp = alphabet.get_Alphabet_corpus()
        btp = bt.BasicTransformerParams()
        btp.batch_size = 1
        vocab_size = len(corp.id_to_token)
        btp.d_model = vocab_size
        btp.d_qk = 3
        btp.d_vo = vocab_size
        btp.use_attn_layers = True
        btp.num_attn_layers = 3
        model, optimizer, dataloader = bt.create_model(
            btp, corp, aot.AttnAndUnembedTransformer)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        print(f'avg_loss {avg_loss}, response_errs {response_errs}')
        self.assertEqual(response_errs, 0)


if __name__ == '__main__':
    unittest.main()
