import unittest
import torch
import basic_transformer_v6 as bt


class TestBasicTransformer(unittest.TestCase):

    # @unittest.skip("temp")
    def test_single_head(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 2
        bt.D_QK = 1
        bt.D_VO = 1
        bt.NUM_ATTN_HEADS = 1
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = False
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 300
        bt.LOSS_PRINT_FREQ = 50
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        # relevant params
        bt.BATCH_SIZE = 1
        bt.NUM_ATTN_HEADS = 1

        exp_avg_losses = {('single', 1): 0.1263359, ('single', 5): 0.13378,
                          ('multi', 1): 0.1263359, ('multi', 5): 0.13378,
                          ('multicompact', 1): 0.1263359, ('multicompact', 5): 0.13378,
                          }

        for bt.ATTN_HEAD_CONFIG in ['single', 'multi', 'multicompact']:
            for bt.BATCH_SIZE in [1, 5]:
                with self.subTest(batch_size=bt.BATCH_SIZE, attn_head_config=bt.ATTN_HEAD_CONFIG):
                    model, optimizer, dataloader = bt.create_model(
                        batch_size=bt.BATCH_SIZE)
                    avg_loss = bt.do_epochs(model, optimizer, dataloader)
                    self.assertAlmostEqual(
                        avg_loss, exp_avg_losses[(bt.ATTN_HEAD_CONFIG, bt.BATCH_SIZE)], places=5)
                    response_errs = bt.count_errors(
                        model, dataloader, response_errs_only=True)
                    self.assertEqual(response_errs, 0)

    def test_bigger_dims(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 2
        bt.D_QK = 1
        bt.D_VO = 1
        bt.NUM_ATTN_HEADS = 1
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = False
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 300
        bt.LOSS_PRINT_FREQ = 50
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        # relevant params
        bt.BATCH_SIZE = 5
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.D_MODEL = 6
        bt.D_QK = 4
        bt.D_VO = 3
        bt.USE_2FFN = True

        exp_avg_loss = 0.10733

        model, optimizer, dataloader = bt.create_model(
            batch_size=bt.BATCH_SIZE)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        self.assertAlmostEqual(
            avg_loss, exp_avg_loss, places=5)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True)
        self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multilayer(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 2
        bt.D_QK = 1
        bt.D_VO = 1
        bt.NUM_ATTN_HEADS = 1
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = True
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 1000
        bt.LOSS_PRINT_FREQ = 100
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        # relevant params
        bt.USE_ATTN_LAYERS = True
        bt.NUM_ATTN_LAYERS = 3
        bt.LEARNING_RATE = 0.005

        exp_avg_losses = {'single': 0.108839326,
                          'multi': 0.108839326,
                          'multicompact': 0.108839326,
                          }

        for bt.ATTN_HEAD_CONFIG in ['single', 'multi', 'multicompact']:
            # for bt.ATTN_HEAD_CONFIG in ['single']:
            with self.subTest(attn_head_config=bt.ATTN_HEAD_CONFIG):
                model, optimizer, dataloader = bt.create_model(
                    batch_size=bt.BATCH_SIZE)
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[bt.ATTN_HEAD_CONFIG], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multihead(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 4
        bt.D_QK = 3
        bt.D_VO = 2
        bt.NUM_ATTN_HEADS = 1
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = False
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 300
        bt.LOSS_PRINT_FREQ = 50
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        # relevant params
        bt.NUM_ATTN_HEADS = 3
        bt.BATCH_SIZE = 5
        bt.NUM_EPOCHS = 500

        exp_avg_losses = {
            'multi': 0.10699,
            'multicompact': 0.10699,
        }

        models_etc = dict()
        for bt.ATTN_HEAD_CONFIG in ['multi', 'multicompact']:
            model, optimizer, dataloader = bt.create_model(
                batch_size=bt.BATCH_SIZE)
            models_etc[bt.ATTN_HEAD_CONFIG] = (
                model, optimizer, dataloader)
        bt.copy_attn_multi_to_compact(models_etc['multi'][0],
                                      models_etc['multicompact'][0],
                                      bt.NUM_ATTN_HEADS, bt.D_MODEL, bt.D_QK, bt.D_VO)
        for bt.ATTN_HEAD_CONFIG in ['multi', 'multicompact']:
            with self.subTest(attn_head_config=bt.ATTN_HEAD_CONFIG):
                model, optimizer, dataloader = models_etc[bt.ATTN_HEAD_CONFIG]
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[bt.ATTN_HEAD_CONFIG], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    # @unittest.skip("temp")
    def test_multilayer_multihead(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 2
        bt.D_QK = 1
        bt.D_VO = 1
        bt.NUM_ATTN_HEADS = 2
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = True
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 1000
        bt.LOSS_PRINT_FREQ = 100
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        # relevant params
        bt.USE_ATTN_LAYERS = True
        bt.NUM_ATTN_LAYERS = 3
        bt.LEARNING_RATE = 0.005

        exp_avg_losses = {'multi': 0.113159,
                          'multicompact': 0.1074407,
                          }

        for bt.ATTN_HEAD_CONFIG in ['multi', 'multicompact']:
            with self.subTest(attn_head_config=bt.ATTN_HEAD_CONFIG):
                model, optimizer, dataloader = bt.create_model(
                    batch_size=bt.BATCH_SIZE)
                avg_loss = bt.do_epochs(model, optimizer, dataloader)
                self.assertAlmostEqual(
                    avg_loss, exp_avg_losses[bt.ATTN_HEAD_CONFIG], places=5)
                response_errs = bt.count_errors(
                    model, dataloader, response_errs_only=True)
                self.assertEqual(response_errs, 0)

    def test_quick_all_configs(self):
        bt.ATTN_HEAD_CONFIG = 'single'
        bt.USE_ATTN_LAYERS = False

        bt.MAX_INPUT_TOKENS = 6
        bt.D_MODEL = 8
        bt.D_QK = 4
        bt.D_VO = 3
        bt.NUM_ATTN_HEADS = 1
        bt.NUM_ATTN_LAYERS = 3
        bt.USE_2FFN = True
        bt.D_FFN = 20

        bt.LEARNING_RATE = 0.02
        bt.NUM_EPOCHS = 300
        bt.LOSS_PRINT_FREQ = 50
        bt.BATCH_SIZE = 5
        bt.THE_SEED = 42

        for bt.ATTN_HEAD_CONFIG in ['single', 'multi', 'multicompact']:
            for bt.BATCH_SIZE in [1, 5]:
                for bt.USE_ATTN_LAYERS in [False, True]:
                    for bt.USE_2FFN in [False, True]:
                        with self.subTest(batch_size=bt.BATCH_SIZE, attn_head_config=bt.ATTN_HEAD_CONFIG,
                                          use_attn_layers=bt.USE_ATTN_LAYERS,
                                          use_2ffn=bt.USE_2FFN):
                            max_steps = 3
                            model, optimizer, dataloader = bt.create_model(
                                batch_size=bt.BATCH_SIZE)
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
