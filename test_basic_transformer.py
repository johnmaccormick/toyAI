import unittest
import basic_transformer as bt


class TestBasicTransformer(unittest.TestCase):

    # @unittest.skip("temp")
    def test_single_head(self):
        btp = bt.BasicTransformerParams()
        btp.num_attn_heads = 1
        btp.d_qk = 1
        btp.d_vo = 1
        btp.attn_head_config = 'single'
        btp.use_2ffn = False
        btp.d_ffn = 20
        btp.batch_size = 1

        exp_avg_losses = {('single', 1): 0.1263359, ('single', 5): 0.13378,
                          ('multi', 1): 0.1263359, ('multi', 5): 0.13378,
                          ('multicompact', 1): 0.1263359, ('multicompact', 5): 0.13378,
                          }

        for btp.attn_head_config in ['single', 'multi', 'multicompact']:
            for btp.batch_size in [1, 5]:
                with self.subTest(batch_size=btp.batch_size, attn_head_config=btp.attn_head_config):
                    model, optimizer, dataloader = bt.create_model(btp)
                    avg_loss = bt.do_epochs(model, optimizer, dataloader)
                    self.assertAlmostEqual(
                        avg_loss, exp_avg_losses[(btp.attn_head_config, btp.batch_size)], places=5)
                    response_errs = bt.count_errors(
                        model, dataloader, response_errs_only=True)
                    self.assertEqual(response_errs, 0)

    def test_bigger_dims(self):
        btp = bt.BasicTransformerParams()
        btp.d_model = 6
        btp.use_attn_layers = False
        btp.d_qk = 4
        btp.d_vo = 3
        btp.attn_head_config = 'single'
        btp.use_2ffn = True
        btp.batch_size = 5

        exp_avg_loss = 0.10733

        model, optimizer, dataloader = bt.create_model(btp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(
            avg_loss, exp_avg_loss, places=5)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True)
        self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multilayer(self):
        btp = bt.BasicTransformerParams()
        btp.use_attn_layers = True
        btp.num_attn_layers = 3
        btp.d_qk = 1
        btp.d_vo = 1
        btp.use_2ffn = True
        btp.batch_size = 5
        btp.num_epochs = 1000
        btp.loss_print_freq = 100
        btp.learning_rate = 0.005

        exp_avg_losses = {'single': 0.108839326,
                          'multi': 0.108839326,
                          'multicompact': 0.108839326,
                          }

        for btp.attn_head_config in ['single', 'multi', 'multicompact']:
            with self.subTest(attn_head_config=btp.attn_head_config):
                model, optimizer, dataloader = bt.create_model(btp)
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[btp.attn_head_config], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multihead(self):
        btp = bt.BasicTransformerParams()
        btp.d_model = 4
        btp.num_attn_heads = 3
        btp.d_qk = 3
        btp.d_vo = 2
        btp.batch_size = 5
        btp.num_epochs = 500

        exp_avg_losses = {
            'multi': 0.10699,
            'multicompact': 0.10699,
        }

        models_etc = dict()
        for btp.attn_head_config in ['multi', 'multicompact']:
            model, optimizer, dataloader = bt.create_model(btp)
            models_etc[btp.attn_head_config] = (
                model, optimizer, dataloader)
        bt.copy_attn_multi_to_compact(btp,
                                      models_etc['multi'][0],
                                      models_etc['multicompact'][0])
        for btp.attn_head_config in ['multi', 'multicompact']:
            with self.subTest(attn_head_config=btp.attn_head_config):
                model, optimizer, dataloader = models_etc[btp.attn_head_config]
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[btp.attn_head_config], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multilayer_multihead(self):
        btp = bt.BasicTransformerParams()
        btp.num_attn_heads = 2
        btp.use_attn_layers = True
        btp.num_attn_layers = 3
        btp.d_qk = 1
        btp.d_vo = 1
        btp.use_2ffn = True
        btp.batch_size = 5
        btp.num_epochs = 1000
        btp.loss_print_freq = 100
        btp.learning_rate = 0.005

        exp_avg_losses = {'multi': 0.113159,
                          'multicompact': 0.1074407,
                          }

        for btp.attn_head_config in ['multi', 'multicompact']:
            with self.subTest(attn_head_config=btp.attn_head_config):
                model, optimizer, dataloader = bt.create_model(btp)
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[btp.attn_head_config], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    def test_quick_all_configs(self):
        btp = bt.BasicTransformerParams()
        btp.d_model = 8
        btp.d_qk = 4
        btp.d_vo = 3
        btp.use_2ffn = True
        btp.batch_size = 5

        for btp.attn_head_config in ['single', 'multi', 'multicompact']:
            for btp.batch_size in [1, 5]:
                for btp.use_attn_layers in [False, True]:
                    for btp.use_2ffn in [False, True]:
                        with self.subTest(batch_size=btp.batch_size, attn_head_config=btp.attn_head_config,
                                          use_attn_layers=btp.use_attn_layers,
                                          use_2ffn=btp.use_2ffn):
                            max_steps = 3
                            model, optimizer, dataloader = bt.create_model(btp)
                            for step, (X, y) in enumerate(dataloader):
                                # print(f"starting step {step}, params:")
                                # print_params(model)
                                # print_gradients(model)
                                if step >= max_steps:
                                    break
                                # print(f'training step {step}...')
                                bt.do_training_step(model, optimizer,
                                                    X, y, print_loss=False)


if __name__ == '__main__':
    unittest.main()
