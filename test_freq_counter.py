import unittest
import basic_transformer as bt
import corpus
import freq_counter


class TestFreqCounter(unittest.TestCase):

    def test_freq_model(self):
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 200
        btp.d_model = 16
        btp.num_attn_heads = 2
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = True
        btp.d_qk = btp.d_model
        btp.d_vo = btp.d_model
        btp.d_ffn = 32
        btp.only_final_input_loss = True

        btp.use_attn_layers = False
        btp.batch_size = 20
        btp.learning_rate = 0.0002
        btp.loss_print_freq = 50

        btp.seed = 12321

        num_in_strs = 500
        min_len = 1
        max_len = 10

        btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer

        sf = freq_counter.Simple_Freq_inputs(seed=btp.seed)
        inputs, labels = sf.make_inputs(num_in_strs, min_len, max_len)
        corp = corpus.Corpus(input_strings=inputs,
                             queries_only=btp.only_final_input_loss)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        # self.assertAlmostEqual(
        #     avg_loss, 0.05972, places=5)
        response_errs = bt.count_last_tok_errors(model, corp)
        self.assertEqual(response_errs, 18)
        print(f'response_errs: {response_errs}')


if __name__ == '__main__':
    unittest.main()
