import collections
from typing import override
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc

INF = 1e9
SEED = 12321


class TransformerParams:
    pass


def visualize_matrix(matrix: torch.Tensor, row_labels: list[str],
                     col_labels: list[str], title: str,
                     log_scale: bool, ax):
    # based on version written by Colab AI

    data = matrix.detach().numpy()
    # fig, ax = plt.subplots()
    if log_scale:
        img_data = np.exp(data)
    else:
        img_data = data

    im = ax.imshow(img_data, cmap='viridis')  # Choose a colormap

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, round(data[i, j], 1),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    # fig.tight_layout()
    # plt.show()


def make_mask(n_val):
    lower_tri = torch.tril(torch.ones(n_val, n_val))
    upper_tri_bool = (lower_tri == 0)
    zeros = torch.zeros(n_val, n_val)
    mask = zeros.masked_fill(upper_tri_bool, -INF)
    return mask


def convert_to_IDs(token_to_id, seqs, labels):
    seqs = [torch.tensor([token_to_id[c] for c in s]) for s in seqs]
    labels = [torch.tensor(token_to_id[c]) for c in labels]
    return seqs, labels


def print_params(model: nn.Module, precision=4):
    # torch.set_printoptions(precision=precision)
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data.numpy().round(precision)}")


def ids_to_string(id_to_token, ids):
    # ids is a 1D  tensor containing token IDs
    assert ids.ndim == 1
    tokens = [id_to_token[ids[i].item()] for i in range(len(ids))]
    return ''.join(tokens)


class Data(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class RelationAttn(nn.Module):

    def __init__(self, d: int, init_with_zeros, use_mask):
        super().__init__()
        self.d = d
        self.use_mask = use_mask
        if not init_with_zeros:
            self.R = nn.Parameter(torch.randn(d, d) / d)
        else:
            self.R = nn.Parameter(torch.zeros(d, d))

    def forward(self, encoding: torch.Tensor):
        assert encoding.ndim == 3
        self.this_batch_size = encoding.shape[0]
        b = self.this_batch_size
        self.n = encoding.shape[1]
        n = self.n
        d = self.d

        assert encoding.shape == (b, n, d)
        assert self.R.shape == (d, d)

        encR = torch.matmul(encoding, self.R)
        assert encR.shape == (b, n, d)

        A_unsc = torch.matmul(encR, encoding.transpose(dim0=1, dim1=2))
        assert A_unsc.shape == (b, n, n)

        if self.use_mask:
            M_mask = make_mask(n)
        else:
            M_mask = torch.zeros(n, n)
        assert M_mask.shape == (n, n)

        A_mskd = A_unsc + M_mask
        assert A_mskd.shape == (b, n, n)
        A = torchfunc.softmax(A_mskd, dim=2)
        assert A.shape == (b, n, n)

        if self.use_mask:
            topleft = A[0, 0, 0].item()
            assert (abs(topleft-1.0) < 1e-5)

        enc_prime = torch.matmul(A, encoding)
        assert enc_prime.shape == (b, n, d)

        return enc_prime


class TrivEmbed(nn.Module):
    def __init__(self, vocab_size: int, ctx_window: int, use_pos_enc: bool):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_window = ctx_window
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.d = vocab_size + ctx_window
        else:
            self.d = vocab_size

    def forward(self, token_ids):
        assert token_ids.ndim == 2
        self.this_batch_size = token_ids.shape[0]
        b = self.this_batch_size
        n = token_ids.shape[1]
        assert n == self.ctx_window
        v = self.vocab_size
        d = self.d

        token_enc = torchfunc.one_hot(token_ids, num_classes=v).float()
        assert token_enc.shape == (b, n, v)
        if self.use_pos_enc:
            pos_enc = torch.eye(n).expand(b, -1, -1)
            assert pos_enc.shape == (b, n, n)
            embedding = torch.cat([token_enc, pos_enc], dim=2)
            assert embedding.shape == (b, n, d)
            return embedding
        else:
            return token_enc


class Example1(nn.Module):
    def __init__(self, vocab_size: int, ctx_window: int, inverse_class_probs=None, init_with_zeros=False):
        torch.manual_seed(SEED)
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_window = ctx_window
        self.d = vocab_size + ctx_window
        self.embed = TrivEmbed(vocab_size, ctx_window, use_pos_enc=True)
        self.reln_attn = RelationAttn(self.d,
                                      init_with_zeros=init_with_zeros,
                                      use_mask=True)
        self.loss = nn.CrossEntropyLoss(weight=inverse_class_probs)

    def forward(self, token_ids):
        assert token_ids.ndim == 2
        self.this_batch_size = token_ids.shape[0]
        b = self.this_batch_size
        n = self.ctx_window
        v = self.vocab_size
        d = self.d

        embedding = self.embed(token_ids)
        assert embedding.shape == (b, n, d)

        X_prime = self.reln_attn(embedding)
        assert X_prime.shape == (b, n, d)

        X_prime_last_row = X_prime[:, -1, :]
        assert X_prime_last_row.shape == (b, d)

        logits = X_prime_last_row

        return logits


class RelationAttnLayers(nn.Module):

    def __init__(self, d: int, num_layers: int,
                 init_with_zeros: bool, use_mask: bool):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            attention_layer = RelationAttn(
                d, init_with_zeros=init_with_zeros, use_mask=use_mask)
            self.attn_layers.append(attention_layer)

    def forward(self, X):
        assert X.ndim == 3
        self.this_batch_size = X.shape[0]
        b = self.this_batch_size
        self.n = X.shape[1]
        n = self.n
        v = self.d
        assert X.shape == (b, n, v)

        for attn in self.attn_layers:
            X = attn(X)
        X_prime = X
        assert X_prime.shape == (b, n, v)

        X_prime_last_row = X_prime[:, -1, :]
        assert X_prime_last_row.shape == (b, v)

        return X_prime_last_row


class Example2(nn.Module):
    def __init__(self, v, num_layers: int, inverse_class_probs=None, init_with_zeros=False):
        torch.manual_seed(SEED)
        super().__init__()
        self.v = v
        self.attn_layers = RelationAttnLayers(
            v, num_layers, init_with_zeros, use_mask=True)
        self.unembed = nn.Linear(in_features=v, out_features=v, bias=False)
        self.loss = nn.CrossEntropyLoss(weight=inverse_class_probs)

    def forward(self, token_ids):
        assert token_ids.ndim == 2
        self.this_batch_size = token_ids.shape[0]
        b = self.this_batch_size
        self.n = token_ids.shape[1]
        n = self.n
        v = self.v

        X = torchfunc.one_hot(token_ids, num_classes=v).float()
        assert X.shape == (b, n, v)

        attn_outputs = self.attn_layers(X)
        assert attn_outputs.shape == (b, v)

        logits = self.unembed(attn_outputs)
        assert logits.shape == (b, v)

        return logits


class Example3(nn.Module):
    def __init__(self, vocab_size: int, ctx_window: int,
                 num_layers: int, inverse_class_probs,
                 init_with_zeros: bool, use_mask: bool,
                 use_pos_enc: bool):
        torch.manual_seed(SEED)
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_window = ctx_window
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.d = vocab_size + ctx_window
        else:
            self.d = vocab_size
        self.embed = TrivEmbed(vocab_size, ctx_window, use_pos_enc)
        self.attn_layers = RelationAttnLayers(
            self.d, num_layers, init_with_zeros, use_mask)
        self.unembed = nn.Linear(
            in_features=self.d, out_features=self.vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss(weight=inverse_class_probs)
        self.verbose = False

    def forward(self, token_ids):
        assert token_ids.ndim == 2
        self.this_batch_size = token_ids.shape[0]
        b = self.this_batch_size
        n = self.ctx_window
        v = self.vocab_size
        d = self.d

        embedding = self.embed(token_ids)
        assert embedding.shape == (b, n, d)
        self.maybe_print_tensor('embedding', embedding)

        attn_outputs = self.attn_layers(embedding)
        assert attn_outputs.shape == (b, d)
        self.maybe_print_tensor('attn_outputs', attn_outputs)

        logits = self.unembed(attn_outputs)
        assert logits.shape == (b, v)
        self.maybe_print_tensor('logits', logits)

        return logits

    def visualize(self, vocab):
        if self.use_pos_enc:
            mat_labels = matrix_labels(vocab, self.ctx_window)
        else:
            mat_labels = vocab

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        R_matrix = self.attn_layers.attn_layers[0].R.data
        title = 'R'
        visualize_matrix(R_matrix, mat_labels, mat_labels,
                         title, log_scale=True, ax=axes[0])

        U_matrix = self.unembed.weight.data.transpose(dim0=0, dim1=1)
        title = 'U'
        visualize_matrix(U_matrix, mat_labels, vocab,
                         title, log_scale=False, ax=axes[1])
        fig.tight_layout()
        plt.show()

    def maybe_print_tensor(self, name: str, t: torch.Tensor):
        if self.verbose:
            print(f'{name}:\n{t.detach().numpy().round(1)}')


def add_padding(ids_list, pad_idx, padded_len=None, on_right=True):
    # ids_list is list of tensors
    max_len = max(map(len, ids_list))
    if padded_len is None:
        padded_len = max_len
    assert padded_len >= max_len
    padded_IDs = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        padding = torch.full((pad_len,), pad_idx)
        if on_right:
            padded_tensor = torch.cat((ids, padding))
        else:
            padded_tensor = torch.cat((padding, ids))
        padded_IDs.append(padded_tensor)
    return padded_IDs


def validate(model, dataset,
             print_errs=False, print_probs=False,
             num_instances=None, id_to_token=None):
    v = model.vocab_size
    num_errs = 0
    tot_loss = 0.0
    for i, (X, y) in enumerate(dataset):
        if num_instances is not None and i >= num_instances:
            break
        y_pred = model(X.unsqueeze(0))
        assert y_pred.shape == (1, v)

        pred_ID = torch.argmax(y_pred, dim=1)
        assert pred_ID.shape == (1,)

        if pred_ID.item() != y.item():
            num_errs += 1
            if print_errs:
                print(f'error on X: {X}, ' +
                      f'y_true: {y.item()}, ' +
                      f'pred_ID {pred_ID.item()}')

        if print_probs:
            do_print_probs(model, id_to_token, v, X, y, y_pred)

        y_true = y.unsqueeze(0)
        assert y_true.shape == (1,)

        loss = model.loss(y_pred, y_true)
        tot_loss += loss.item()

        # print(f'X: {X}, y_true: {y.item()}, pred_ID {pred_ID.item()}')

    avg_loss = tot_loss / len(dataset)
    accuracy = 1.0 - num_errs / len(dataset)
    print(f'validation accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
    return accuracy


def do_print_probs(model, id_to_token, v, X, y, y_pred):
    torch.set_printoptions(precision=2)
    probs = torchfunc.softmax(y_pred, dim=1)
    assert probs.shape == (model.this_batch_size, v)
    x_str = ids_to_string(id_to_token, X)
    y_str = ids_to_string(id_to_token, y.unsqueeze(0))
    print(f'X {x_str}, y {y_str}, probs {probs}')


def do_epoch(dataloader, model, optimizer):
    v = model.vocab_size
    epoch_loss = 0.0
    num_errs = 0
    for step, (X, y) in enumerate(dataloader):
        # print(f"starting step {step}, params:")
        # print_params(model)
        # if step >= max_steps:
        #     break
        y_pred = model(X)
        assert y_pred.shape == (model.this_batch_size, v)

        # probs = torchfunc.softmax(y_pred, dim=1)
        # assert probs.shape == (model.this_batch_size, v)

        y_true = y
        assert y_true.shape == (model.this_batch_size,)

        loss = model.loss(y_pred, y_true)
        epoch_loss += loss.item() * model.this_batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_ids = torch.argmax(y_pred, dim=-1)
        mispredictions = (y - predicted_ids).bool().squeeze(0)
        this_num_errs = torch.count_nonzero(mispredictions).item()
        num_errs += this_num_errs

        # print(f'y_pred: {y_pred}')
        # print(f'probs: {probs}')
        # print(f'y_true: {y_true}')
        # print(f'loss: {loss}')
    avg_loss = epoch_loss / len(dataloader.dataset)
    accuracy = 1 - num_errs / len(dataloader.dataset)
    return avg_loss, accuracy


class Inputs:
    def __init__(self, chars, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.chars = chars
        self.num_chars = len(self.chars)
        self.probs = np.ones(self.num_chars) / self.num_chars
        print(f'self.probs {self.probs}')

    def make_input(self, input_len: int) -> tuple[list[str], str]:
        assert False, 'Abstract method; not implemented'
        # return input_seq, label

    def make_inputs(self, num_inputs, min_len, max_len) -> tuple[list[list[str]], list[str]]:
        inputs = []
        labels = []
        for _ in range(num_inputs):
            input_len = self.rng.integers(low=min_len, high=max_len+1)
            input_seq, label = self.make_input(input_len)
            inputs.append(input_seq)
            labels.append(label)
        return inputs, labels

    def make_dataset(self, token_to_id, num_inputs, min_len, max_len, batch_size):
        seqs, labels = self.make_inputs(num_inputs, min_len, max_len)
        seqs, labels = convert_to_IDs(token_to_id, seqs, labels)
        seqs = add_padding(
            seqs, pad_idx=token_to_id['P'], padded_len=max_len, on_right=False)
        dataset = Data(seqs, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)
        return dataset, dataloader


class Simple_Freq_inputs(Inputs):
    def __init__(self, chars, seed, yesno=False):
        super().__init__(chars, seed)
        self.yesno = yesno

    @override
    def make_input(self, input_len):
        done = False  # disallow ties for most frequent
        while not done:
            # use input_len-1 because E is appended later
            input_seq = [self.rng.choice(self.chars, p=self.probs)
                         for _ in range(input_len-1)]
            if len(input_seq) < 2:
                done = True
            else:
                counter = collections.Counter(input_seq)
                top_two = counter.most_common(2)
                # no tie, so done
                if len(top_two) < 2 or top_two[0][1] != top_two[1][1]:
                    most_freq = top_two[0][0]
                    done = True
        input_seq.append('E')  # seq always ends in E
        if self.yesno:
            if most_freq == 'a':
                label = 'y'
            else:
                label = 'n'
        else:
            label = most_freq
        return input_seq, label


class A_Before_B_inputs(Inputs):
    def __init__(self, chars, seed):
        super().__init__(chars, seed)

    @override
    def make_input(self, input_len):
        input_seq = [self.rng.choice(self.chars, p=self.probs)
                     for _ in range(input_len-1)]
        a_loc, b_loc = self.rng.choice(
            np.arange(input_len-1), size=2, replace=False)
        input_seq[a_loc] = 'a'
        input_seq[b_loc] = 'b'
        label = 'y' if a_loc < b_loc else 'n'
        input_seq.append('E')  # seq always ends in E
        return input_seq, label


def calc_inverse_class_probs(class_ids, num_classes):
    """Computes the inverses of the probabilities of class IDs in a tensor.
    Based on code produced by Colab AI.

    Args:
      class_ids: A 1D PyTorch tensor containing class IDs.

    Returns:
      A PyTorch tensor containing the inverses of the probabilities of each class ID.
    """
    class_counts = torch.bincount(class_ids, minlength=num_classes)
    total_count = torch.sum(class_counts)
    class_probs = class_counts.float() / total_count
    # Prevent division by zero
    class_probs[class_probs == 0] = torch.min(class_probs[class_probs != 0])
    inverse_probs = 1.0 / class_probs
    return inverse_probs


def evaluate_input(model: Example3, x: str, y: str,
                   token_to_id: dict[str, int]):
    print(f'** evaluating {x} **')
    model.eval()
    x = torch.tensor([token_to_id[c] for c in x])
    y = torch.tensor(token_to_id[y])

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    y_pred = model(x)
    loss = model.loss(y_pred, y)
    print(f'x {x}')
    print(f'y {y}')
    print(f'logits {y_pred}')
    print(f'outprobs {torchfunc.softmax(y_pred, dim=1)}')
    print(f'loss {loss}')
    return loss


def evaluate_gradient(model: Example1, x: torch.Tensor, y: torch.Tensor):
    model.train()
    model.zero_grad()
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    y_pred = model(x)
    loss = model.loss(y_pred, y)
    print(f'x {x}')
    print(f'y {y}')
    print(f'logits {y_pred}')
    print(f'outprobs {torchfunc.softmax(y_pred, dim=1)}')

    print(f'loss {loss}')
    loss.backward()
    print_gradients(model)


def print_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}:\n{param.grad}")
        else:
            print(f"Gradient of {name}:\nNot available")


def example2a():
    token_to_id = {'a': 0,
                   'b': 1,
                   'E': 2,
                   'P': 3,
                   }
    id_to_token = dict(map(reversed, token_to_id.items()))
    v = len(token_to_id)
    pad_idx = token_to_id['P']

    strs, labels = zip(('abE', 'a'),
                       ('baE', 'a'),
                       ('aaE', 'a'),
                       ('bbE', 'b'),
                       )
    # seqs = [torch.tensor([token_to_id[c] for c in s]) for s in strs]
    # labels = [torch.tensor(token_to_id[c]) for c in labels]
    seqs, labels = convert_to_IDs(token_to_id, strs, labels)
    print(f'sequences {seqs}')
    print(f'labels {labels}')
    dataset = Data(seqs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    model = Example1(v, init_with_zeros=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # max_steps = 30
    num_epochs = 10000
    print_freq = 2000
    print(f"initial params:")
    print_params(model)

    # x, y = dataset[0]
    # evaluate_gradient(model, x, y)
    # return

    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(dataloader, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            validate(model, dataset)

    print(f"final params:")
    print_params(model)
    validate(model, dataset, print_probs=True)


def example3a():
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
    seqs, labels = convert_to_IDs(token_to_id, strs, labels)
    print(f'sequences {seqs}')
    print(f'labels {labels}')
    dataset = Data(seqs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    ctx_window = len(seqs[0])
    for s in seqs:
        assert len(s) == ctx_window

    num_layers = 1
    use_mask = False
    use_pos_enc = True
    model = Example3(vocab_size=v, ctx_window=ctx_window, num_layers=num_layers,
                     inverse_class_probs=None, init_with_zeros=False,
                     use_mask=use_mask, use_pos_enc=use_pos_enc)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # max_steps = 30
    num_epochs = 3000
    print_freq = 1000
    print(f"initial params:")
    print_params(model, precision=2)

    # x, y = dataset[0]
    # evaluate_gradient(model, x, y)
    # return

    do_training(dataset, dataloader, model, optimizer, num_epochs, print_freq)

    print(f"final params:")
    print_params(model, precision=1)
    validate(model, dataset, id_to_token=id_to_token, print_probs=True)
    model.verbose = True
    evaluate_input(model, 'abE', 'y', token_to_id)
    evaluate_input(model, 'baE', 'n', token_to_id)


def do_training(dataset, dataloader, model, optimizer, num_epochs, print_freq):
    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(dataloader, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            validate(model, dataset)
    # model.visualize(vocab)


def example2b():
    num_layers = 2
    vocab = ['a', 'b',
             'c', 'd',
             #  'e', 'f',
             'y', 'n',
             #  'g', 'h', 'i', 'j',
             'E', 'P']
    token_to_id = dict()
    for i, token in enumerate(vocab):
        token_to_id[token] = i
    id_to_token = dict(map(reversed, token_to_id.items()))
    v = len(token_to_id)
    pad_idx = token_to_id['P']
    chars = sorted(list(set(vocab) - {'E', 'P', 'y', 'n', 'a', 'b'}))

    num_inputs_train = 10000
    num_inputs_validate = 1000
    min_len = 4
    max_len = 12
    batch_size = 20
    learning_rate = 0.01
    num_epochs = 15
    print_freq = 3

    abb = A_Before_B_inputs(chars, SEED)
    dataset_train, dataloader_train = abb.make_dataset(
        token_to_id, num_inputs_train, min_len, max_len, batch_size)
    dataset_validate, _ = abb.make_dataset(
        token_to_id, num_inputs_validate, min_len, max_len, batch_size)
    labels = torch.tensor([y.item() for y in dataset_train.Y])
    inverse_class_probs = calc_inverse_class_probs(
        labels, num_classes=v)
    print(f'inverse_class_probs {inverse_class_probs}')

    print('First few training instances:')
    for i in range(10):
        x, y = dataset_train[i]
        x = ids_to_string(id_to_token, x)
        y = ids_to_string(id_to_token, y.unsqueeze(0))
        print(x, y)

    model = Example2(v, num_layers, inverse_class_probs)
    # model = Example1c(v)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(f"initial params:")
    # print_params(model)
    validate(model, dataset_validate, print_errs=False)
    # x, y = dataset_validate[0]
    # evaluate_gradient(model, x, y)
    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(
            dataloader_train, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            print_errs = epoch == num_epochs-1
            validate(model, dataset_validate, print_errs=print_errs)

    # torch.save(model.state_dict(), 'example1c.pth')
    print(f"final params:")
    print_params(model, precision=1)
    num_instances = 5
    print(f'probs for first {num_instances} validation instances:')
    validate(model, dataset_validate,
             print_probs=True, num_instances=num_instances,
             id_to_token=id_to_token)


def matrix_labels(vocab: list[str], ctx_window: int) -> list:
    pos_strs = [str(i) for i in range(ctx_window)]
    return vocab + pos_strs


def example1a():
    num_layers = 1
    token_to_id = {'a': 0,
                   'b': 1,
                   'E': 2,
                   'P': 3,
                   }
    id_to_token = dict(map(reversed, token_to_id.items()))
    vocab = sorted(token_to_id, key=token_to_id.get)
    vocab_size = len(vocab)
    pad_idx = token_to_id['P']

    strs, labels = zip(('abE', 'a'),
                       ('baE', 'a'),
                       ('aaE', 'a'),
                       ('bbE', 'b'),
                       )
    # seqs = [torch.tensor([token_to_id[c] for c in s]) for s in strs]
    # labels = [torch.tensor(token_to_id[c]) for c in labels]
    seqs, labels = convert_to_IDs(token_to_id, strs, labels)
    print(f'sequences {seqs}')
    print(f'labels {labels}')
    dataset = Data(seqs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    ctx_window = len(seqs[0])
    for s in seqs:
        assert len(s) == ctx_window
    model = Example1(vocab_size, ctx_window, init_with_zeros=False)
    # model = Example2(vocab_size, ctx_window, num_layers, init_with_zeros=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # max_steps = 30
    num_epochs = 6000
    print_freq = 2000
    print(f"initial params:")
    print_params(model, precision=1)

    # x, y = dataset[0]
    # evaluate_gradient(model, x, y)
    # return

    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(dataloader, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            validate(model, dataset)

    print(f"final params:")
    print_params(model, precision=1)
    validate(model, dataset, id_to_token=id_to_token, print_probs=True)
    R_matrix = model.reln_attn.R.data
    mat_labels = matrix_labels(vocab, ctx_window)
    title = 'R'
    visualize_matrix(R_matrix, mat_labels, title)


def example1b():
    num_layers = 2
    vocab = ['a', 'b', 'c', 'd', 'e', 'f',
             #  'g', 'h', 'i', 'j',
             'E', 'P']
    token_to_id = dict()
    for i, token in enumerate(vocab):
        token_to_id[token] = i
    id_to_token = dict(map(reversed, token_to_id.items()))
    v = len(token_to_id)
    pad_idx = token_to_id['P']
    chars = sorted(list(set(vocab) - {'E', 'P'}))

    num_inputs_train = 10000
    num_inputs_validate = 1000
    min_len = 4
    max_len = 10
    batch_size = 20
    learning_rate = 0.01
    num_epochs = 50
    print_freq = 5

    sfi = Simple_Freq_inputs(chars, SEED)
    dataset_train, dataloader_train = sfi.make_dataset(
        token_to_id, num_inputs_train, min_len, max_len, batch_size)
    dataset_validate, _ = sfi.make_dataset(
        token_to_id, num_inputs_validate, min_len, max_len, batch_size)
    labels = torch.tensor([y.item() for y in dataset_train.Y])
    inverse_class_probs = calc_inverse_class_probs(
        labels, num_classes=v)
    print(f'inverse_class_probs {inverse_class_probs}')

    print('First four training instances:')
    for i in range(4):
        x, y = dataset_train[i]
        x = ids_to_string(id_to_token, x)
        y = ids_to_string(id_to_token, y.unsqueeze(0))
        print(x, y)

    model = Example2(v, num_layers, inverse_class_probs)
    # model = Example1(v)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(f"initial params:")
    # print_params(model)
    validate(model, dataset_validate, print_errs=False)
    # x, y = dataset_validate[0]
    # evaluate_gradient(model, x, y)
    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(
            dataloader_train, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            validate(model, dataset_validate, print_errs=True)

    print(f"final params:")
    print_params(model)
    validate(model, dataset_validate,
             print_probs=True, num_instances=4,
             id_to_token=id_to_token)


def example1c():
    num_layers = 2
    vocab = ['a', 'b',
             'c', 'd',
             #  'e', 'f',
             'y', 'n',
             #  'g', 'h', 'i', 'j',
             'E', 'P']
    token_to_id = dict()
    for i, token in enumerate(vocab):
        token_to_id[token] = i
    id_to_token = dict(map(reversed, token_to_id.items()))
    v = len(token_to_id)
    pad_idx = token_to_id['P']
    chars = sorted(list(set(vocab) - {'E', 'P', 'y', 'n'}))

    num_inputs_train = 10000
    num_inputs_validate = 1000
    min_len = 4
    max_len = 10
    batch_size = 20
    learning_rate = 0.01
    num_epochs = 30
    print_freq = 5

    sfi = Simple_Freq_inputs(chars, SEED, yesno=True)
    dataset_train, dataloader_train = sfi.make_dataset(
        token_to_id, num_inputs_train, min_len, max_len, batch_size)
    dataset_validate, _ = sfi.make_dataset(
        token_to_id, num_inputs_validate, min_len, max_len, batch_size)
    labels = torch.tensor([y.item() for y in dataset_train.Y])
    inverse_class_probs = calc_inverse_class_probs(
        labels, num_classes=v)
    print(f'inverse_class_probs {inverse_class_probs}')

    print('First few training instances:')
    for i in range(4):
        x, y = dataset_train[i]
        x = ids_to_string(id_to_token, x)
        y = ids_to_string(id_to_token, y.unsqueeze(0))
        print(x, y)

    model = Example2(v, num_layers, inverse_class_probs)
    # model = Example1c(v)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(f"initial params:")
    # print_params(model)
    validate(model, dataset_validate, print_errs=False)
    # x, y = dataset_validate[0]
    # evaluate_gradient(model, x, y)
    for epoch in range(num_epochs):
        # print(f"starting epoch {epoch}, params:")
        # print_params(model)
        model.train()
        avg_loss, accuracy = do_epoch(
            dataloader_train, model, optimizer)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            print_errs = epoch == num_epochs-1
            validate(model, dataset_validate, print_errs=print_errs)

    # torch.save(model.state_dict(), 'example1c.pth')
    print(f"final params:")
    print_params(model, precision=3)
    num_instances = 10
    print(f'probs for first {num_instances} validation instances:')
    validate(model, dataset_validate,
             print_probs=True, num_instances=num_instances,
             id_to_token=id_to_token)


def main():
    example3a()
    # visualize_matrix(torch.rand(3, 3), ['a', 'b', 'c'], 'abc')


if __name__ == "__main__":
    main()
