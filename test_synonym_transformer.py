import unittest
import basic_transformer as bt
import corpus


class TestSynonymTransformer(unittest.TestCase):

    def test_make_and_train_quick(self):
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 3  # 150
        btp.d_model = 4
        btp.num_attn_heads = 3
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = True
        num_in_strs = 100

        num_syn_lists = 2
        exp_avg_loss = 1.02341
        exp_response_errs = 60
        si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                                   fixed_order=True)
        inputs, labels = si.make_inputs(num_inputs=num_in_strs)
        corp = corpus.Corpus(input_strings=inputs)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(
            avg_loss, exp_avg_loss, places=5)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        self.assertEqual(response_errs, exp_response_errs)

    @unittest.skip("temp")
    def test_make_and_train_no_errs(self):
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 500
        btp.d_model = 4
        btp.num_attn_heads = 2
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = True
        btp.d_qk = btp.d_model
        btp.d_vo = btp.d_model
        btp.d_ffn = 20

        btp.use_attn_layers = False
        btp.batch_size = 15
        btp.learning_rate = 0.01

        num_in_strs = 100
        num_syn_lists = 2
        exp_avg_loss = 0.50418
        exp_response_errs = 0

        si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                                   fixed_order=True)
        inputs, labels = si.make_inputs(num_inputs=num_in_strs)
        corp = corpus.Corpus(input_strings=inputs)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(
            avg_loss, exp_avg_loss, places=5)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        self.assertEqual(response_errs, exp_response_errs)

    @unittest.skip("temp")
    def test_learn_syms_fixed_order(self):
        # Succeeded learning synonyms for fixed order of inputs of the categories,
        # 3 syn lists.
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 200
        btp.d_model = 16
        btp.num_attn_heads = 8
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = True
        btp.d_qk = btp.d_model
        btp.d_vo = btp.d_model
        btp.d_ffn = 32

        btp.use_attn_layers = False
        btp.batch_size = 10
        btp.learning_rate = 0.002
        btp.loss_print_freq = 50

        num_syn_lists = 3
        num_in_strs = 300
        fixed_order = True

        si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                                   fixed_order=fixed_order)
        inputs, labels = si.make_inputs(num_inputs=num_in_strs)
        corp = corpus.Corpus(input_strings=inputs)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        start_seed = 123
        response_err_vals = []
        for i in range(3):
            btp.seed = start_seed + i
            model, optimizer, dataloader = bt.create_model(btp, corp)
            avg_loss = bt.do_epochs(model, optimizer, dataloader)
            response_errs = bt.count_errors(
                model, dataloader, response_errs_only=True, corp=corp)
            response_err_vals.append(response_errs)
            self.assertEqual(response_errs, 0)
        print(f'response_err_vals: {response_err_vals}')

    @unittest.skip("temp")
    def test_learn_syms_random_order(self):
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 500
        btp.d_model = 16
        btp.num_attn_heads = 16
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = True
        btp.d_qk = btp.d_model
        btp.d_vo = btp.d_model
        btp.d_ffn = 32

        btp.use_attn_layers = False
        btp.batch_size = 10
        btp.learning_rate = 0.002
        btp.loss_print_freq = 100

        num_syn_lists = 3
        num_in_strs = 300
        fixed_order = False

        si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                                   fixed_order=fixed_order)
        inputs, labels = si.make_inputs(num_inputs=num_in_strs)
        corp = corpus.Corpus(input_strings=inputs)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        start_seed = 123
        response_err_vals = []
        for i in range(3):
            btp.seed = start_seed + i
            model, optimizer, dataloader = bt.create_model(btp, corp)
            avg_loss = bt.do_epochs(model, optimizer, dataloader)
            response_errs = bt.count_errors(
                model, dataloader, response_errs_only=True, corp=corp)
            self.assertEqual(response_errs, 0)
            response_err_vals.append(response_errs)
        print(f'response_err_vals: {response_err_vals}')

    @unittest.skip("temp")
    def test_syn_model_final_only_loss(self):
        btp = bt.BasicTransformerParams()
        btp.num_epochs = 200
        btp.d_model = 16
        btp.num_attn_heads = 2
        btp.attn_head_config = 'multicompact'
        btp.use_2ffn = False
        btp.d_qk = 8
        btp.d_vo = 8
        btp.d_ffn = 16

        btp.use_attn_layers = False
        btp.batch_size = 10
        btp.learning_rate = 0.002
        btp.loss_print_freq = 50

        btp.only_final_input_loss = True

        num_syn_lists = 3
        num_in_strs = 300
        fixed_order = False
        queries_only = True

        si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                                   fixed_order=fixed_order)
        inputs, labels = si.make_inputs(num_inputs=num_in_strs)
        corp = corpus.Corpus(input_strings=inputs, queries_only=queries_only)
        print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
        start_seed = 123
        btp.seed = start_seed
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(avg_loss, 0.00023, places=5)
        errs = bt.count_last_tok_errors(model, corp)
        print(f'errs: {errs}')
        self.assertEqual(errs, 0)


if __name__ == '__main__':
    unittest.main()
