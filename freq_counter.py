from collections import Counter
import numpy as np
import torch
import basic_transformer as bt
import corpus


class Simple_Freq_inputs:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.chars = ['a', 'b', 'c']
        self.probs = [0.45, 0.35, 0.2]
        self.num_chars = len(self.chars)

    def make_input(self, input_len):
        chosen_chars = [self.rng.choice(
            self.chars, p=self.probs) for _ in range(input_len)]
        counter = Counter(chosen_chars)
        most_freq = counter.most_common(1)[0][0]
        input_str = ' '.join(chosen_chars) + ' <EOS> ' + most_freq
        return input_str, most_freq

    def make_inputs(self, num_inputs, min_len, max_len):
        inputs = []
        labels = []
        for _ in range(num_inputs):
            input_len = self.rng.integers(low=min_len, high=max_len+1)
            input_str, label = self.make_input(input_len)
            inputs.append(input_str)
            labels.append(label)
        return inputs, labels


def create_and_save_freq_model():
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 1000
    btp.d_model = 16
    btp.num_attn_heads = 4
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True
    btp.d_qk = btp.d_model
    btp.d_vo = btp.d_model
    btp.d_ffn = 32
    btp.only_final_input_loss = True

    btp.use_attn_layers = False
    btp.batch_size = 20
    btp.learning_rate = 0.0001
    btp.loss_print_freq = 50

    btp.seed = 12321

    num_in_strs = 300
    min_len = 1
    max_len = 10

    btp.max_input_tokens = max_len + 2  # +2 is due to <EOS> and query answer

    sf = Simple_Freq_inputs(seed=btp.seed)
    inputs, labels = sf.make_inputs(num_in_strs, min_len, max_len)
    corp = corpus.Corpus(input_strings=inputs,
                         queries_only=btp.only_final_input_loss)
    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    model, optimizer, dataloader = bt.create_model(btp, corp)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_last_tok_errors(model, corp)
    print(f'response_errs: {response_errs}')
    bt.print_some_query_answers(corp, model)
    torch.save(model.state_dict(), 'freq-model.pth')


def main1():
    sf = Simple_Freq_inputs(345)
    for i in range(1, 20):
        print(f'{i}: {sf.make_input(i)}')


def main():
    create_and_save_freq_model()


# torch.save(model.state_dict(), 'syn-model.pth')

if __name__ == "__main__":
    main()
