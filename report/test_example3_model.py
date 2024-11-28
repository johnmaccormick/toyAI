import unittest

import torch
import example3_model as ex3


class TestExample3(unittest.TestCase):
    def test_a_in_pos0(self):
        token_to_id = {'a': 0,
                       'b': 1,
                       'y': 2,
                       'n': 3,
                       'E': 4,
                       }
        id_to_token = dict(map(reversed, token_to_id.items()))
        vocab = sorted(token_to_id, key=token_to_id.get)
        v = len(vocab)
        # pad_idx = token_to_id['P']

        strs, labels = zip(('abE', 'y'),
                           ('baE', 'n'),
                           ('aaE', 'y'),
                           ('bbE', 'n'),
                           )
        # seqs = [torch.tensor([token_to_id[c] for c in s]) for s in strs]
        # labels = [torch.tensor(token_to_id[c]) for c in labels]
        seqs, labels = ex3.convert_to_IDs(token_to_id, strs, labels)
        print(f'sequences {seqs}')
        print(f'labels {labels}')
        dataset = ex3.Data(seqs, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        ctx_window = len(seqs[0])
        for s in seqs:
            assert len(s) == ctx_window

        num_layers = 1
        use_mask = False
        use_pos_enc = True
        model = ex3.Example3(vocab_size=v, ctx_window=ctx_window, num_layers=num_layers,
                             inverse_class_probs=None, init_with_zeros=False,
                             use_mask=use_mask, use_pos_enc=use_pos_enc)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # max_steps = 30
        num_epochs = 3000
        print_freq = 1000
        # print(f"initial params:")
        # print_params(model, precision=2)

        # x, y = dataset[0]
        # evaluate_gradient(model, x, y)
        # return

        ex3.do_training(dataset, dataloader, model,
                        optimizer, num_epochs, print_freq)

        # print(f"final params:")
        # print_params(model, precision=1)
        accuracy = ex3.validate(
            model, dataset, id_to_token=id_to_token, print_probs=True)
        self.assertAlmostEqual(accuracy, 1.0, places=4)
        # model.verbose = True
        loss = ex3.evaluate_input(model, 'abE', 'y', token_to_id)
        self.assertAlmostEqual(loss, 0.000243514, delta=1e-6)
        loss = ex3.evaluate_input(model, 'baE', 'n', token_to_id)
        self.assertAlmostEqual(loss, 0.0004744596, delta=1e-6)
        # model.visualize(vocab)


if __name__ == '__main__':
    unittest.main()
