from collections import Counter
import numpy as np
import torch
import attn_only_transformer as aot
import basic_transformer as bt
import corpus


class B_before_C_inputs:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed=seed)

    def make_input(self, input_len):
        background = ['u', 'v', 'w', 'x']
        chars = [self.rng.choice(background) for _ in range(input_len)]
        b_loc, c_loc = self.rng.choice(
            np.arange(input_len), size=2, replace=False)
        chars[b_loc] = 'b'
        chars[c_loc] = 'c'
        if b_loc < c_loc:
            response = 'y'
        else:
            response = 'n'
        input_str = ' '.join(chars) + ' <EOS> ' + response
        return input_str, response

    def make_inputs(self, num_inputs, min_len, max_len):
        inputs = []
        labels = []
        assert min_len >= 2  # We need space for a 'b' and a 'c'
        for _ in range(num_inputs):
            input_len = self.rng.integers(low=min_len, high=max_len+1)
            input_str, label = self.make_input(input_len)
            inputs.append(input_str)
            labels.append(label)
        return inputs, labels


def create_and_save_b_before_c_model():
    seed = 23433
    num_in_strs = 50000
    min_len = 10
    max_len = min_len
    queries_only = True
    bc = B_before_C_inputs(seed=seed)
    inputs, labels = bc.make_inputs(num_in_strs, min_len, max_len)
    corp = corpus.Corpus(input_strings=inputs,
                         queries_only=queries_only)

    btp = bt.BasicTransformerParams()
    btp.d_model = corp.vocab_size
    btp.only_final_input_loss = queries_only
    btp.seed = seed
    btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer
    btp.use_position_encoding = False

    btp.use_attn_layers = True
    btp.num_attn_layers = 2

    btp.num_epochs = 30
    btp.only_final_input_loss = True

    btp.batch_size = 20
    btp.learning_rate = 0.0002
    btp.loss_print_freq = 1

    btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer

    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    # model, optimizer, dataloader = bt.create_model(btp, corp)
    model, optimizer, dataloader = bt.create_model(
        btp, corp, aot.AttnAndUnencodeTransformer)
    assert isinstance(model, aot.AttnAndUnencodeTransformer)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_last_tok_errors(model, corp)
    print(f'response_errs: {response_errs}')
    bt.print_some_query_answers(corp, model)
    torch.save(model.state_dict(), 'b-before-c-model.pth')


def validate_model():
    seed = 12121
    num_in_strs = 1000
    min_len = 10
    max_len = min_len
    queries_only = True
    bc = B_before_C_inputs(seed=seed)
    inputs, labels = bc.make_inputs(num_in_strs, min_len, max_len)
    corp = corpus.Corpus(input_strings=inputs,
                         queries_only=queries_only)

    btp = bt.BasicTransformerParams()
    btp.d_model = corp.vocab_size
    btp.only_final_input_loss = queries_only
    btp.seed = seed
    btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer
    btp.use_position_encoding = False

    btp.use_attn_layers = True
    btp.num_attn_layers = 2

    btp.num_epochs = 5000
    btp.only_final_input_loss = True

    btp.batch_size = 20
    btp.learning_rate = 0.0002
    btp.loss_print_freq = 50

    btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer

    model, optimizer, dataloader = bt.create_model(
        btp, corp, aot.AttnAndUnencodeTransformer)
    assert isinstance(model, aot.AttnAndUnencodeTransformer)

    model.load_state_dict(torch.load(
        'b-before-c-model-11-19-2024b.pth', weights_only=True))
    model.eval()

    response_errs = bt.count_last_tok_errors(model, corp)
    print(f'response_errs: {response_errs}')
    bt.print_some_query_answers(corp, model)


# def learn_freq_model(model: aot.AttnOnlyTransformer, optimizer, dataloader):
#     btp = model.btp
#     btp.num_epochs = 200
#     btp.only_final_input_loss = True
#     btp.batch_size = 20
#     btp.learning_rate = 0.0002
#     btp.loss_print_freq = 20

#     avg_loss = bt.do_epochs(model, optimizer, dataloader)
#     return avg_loss


# def expt_manual_freq_model():
#     seed = 23432
#     num_in_strs = 500
#     min_len = 1
#     max_len = 10
#     queries_only = True
#     sf = Simple_Freq_inputs(seed=seed)
#     inputs, labels = sf.make_inputs(num_in_strs, min_len, max_len)
#     corp = corpus.Corpus(input_strings=inputs,
#                          queries_only=queries_only)

#     btp = bt.BasicTransformerParams()
#     btp.d_model = corp.vocab_size
#     btp.only_final_input_loss = queries_only
#     btp.seed = seed
#     btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer
#     btp.use_position_encoding = False

#     model, optimizer, dataloader = aot.AttnOnlyTransformer.create_model(
#         btp, corp)
#     assert isinstance(model, aot.AttnOnlyTransformer)
#     assert isinstance(model.head, aot.MinimalAttnHead)
#     model.head.zero_out_W_bil()
#     model.head.mask_pad_token(corp)

#     learn_freq_model(model, optimizer, dataloader)

#     bt.print_params(model)

#     response_errs = bt.count_last_tok_errors(model, corp)
#     print(f'response_errs: {response_errs}')
#     bt.print_some_query_answers(corp, model)


def main1():
    sf = B_before_C_inputs(345)
    for i in range(2, 20):
        print(f'{i}: {sf.make_input(i)}')
    inputs, labels = sf.make_inputs(10, 2, 10)
    for input, label in zip(inputs, labels):
        print(input, label)


def main():
    # create_and_save_b_before_c_model()
    validate_model()


# torch.save(model.state_dict(), 'syn-model.pth')

if __name__ == "__main__":
    main()
