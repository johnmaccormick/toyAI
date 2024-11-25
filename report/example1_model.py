import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc

INF = 1e9
SEED = 12321


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
    torch.set_printoptions(precision=precision)
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data}")


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


class Example1(nn.Module):

    def __init__(self, v, inverse_class_probs=None, init_with_zeros=False):
        torch.manual_seed(SEED)
        super().__init__()
        self.v = v
        if not init_with_zeros:
            self.R = nn.Parameter(torch.randn(v, v) / v)
        else:
            self.R = nn.Parameter(torch.zeros(v, v))
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

        assert self.R.shape == (v, v)

        XR = torch.matmul(X, self.R)
        assert XR.shape == (b, n, v)

        A_unsc = torch.matmul(XR, X.transpose(dim0=1, dim1=2))
        assert A_unsc.shape == (b, n, n)

        M_mask = make_mask(n)
        assert M_mask.shape == (n, n)

        A_mskd = A_unsc + M_mask
        assert A_mskd.shape == (b, n, n)
        A = torchfunc.softmax(A_mskd, dim=2)
        assert A.shape == (b, n, n)
        topleft = A[0, 0, 0].item()
        assert (abs(topleft-1.0) < 1e-5)

        X_prime = torch.matmul(A, X)
        assert X_prime.shape == (b, n, v)

        X_prime_last_row = X_prime[:, -1, :]
        assert X_prime_last_row.shape == (b, v)

        logits = X_prime_last_row

        return logits


class Example1c(nn.Module):
    def __init__(self, v, inverse_class_probs=None, init_with_zeros=False):
        torch.manual_seed(SEED)
        super().__init__()
        self.v = v
        self.example1 = Example1(v, inverse_class_probs, init_with_zeros)
        self.unembed = nn.Linear(in_features=v, out_features=v, bias=False)
        self.loss = nn.CrossEntropyLoss(weight=inverse_class_probs)

    def forward(self, token_ids):
        assert token_ids.ndim == 2
        self.this_batch_size = token_ids.shape[0]
        b = self.this_batch_size
        self.n = token_ids.shape[1]
        n = self.n
        v = self.v

        attn_outputs = self.example1(token_ids)
        assert attn_outputs.shape == (b, v)

        logits = self.unembed(attn_outputs)
        assert logits.shape == (b, v)

        return logits


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
    v = model.v
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
            torch.set_printoptions(precision=2)
            probs = torchfunc.softmax(y_pred, dim=1)
            assert probs.shape == (model.this_batch_size, v)
            x_str = ids_to_string(id_to_token, X)
            y_str = ids_to_string(id_to_token, y.unsqueeze(0))
            print(f'X {x_str}, y {y_str}, probs {probs}')

        y_true = y.unsqueeze(0)
        assert y_true.shape == (1,)

        loss = model.loss(y_pred, y_true)
        tot_loss += loss.item()

        # print(f'X: {X}, y_true: {y.item()}, pred_ID {pred_ID.item()}')

    avg_loss = tot_loss / len(dataset)
    accuracy = 1.0 - num_errs / len(dataset)
    print(f'validation accuracy {accuracy:.5f}, loss {avg_loss:.5f}')


def do_epoch(dataloader, model, optimizer, pad_idx):
    v = model.v
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


class Simple_Freq_inputs:
    def __init__(self, chars, seed, yesno=False):
        self.rng = np.random.default_rng(seed=seed)
        self.chars = chars
        self.num_chars = len(self.chars)
        # Respond with 'y' or 'n' according as 'a' is most freq
        self.yesno = yesno
        # floor_prob = 0.3  # prevent extremely unlikely tokens
        # self.probs = np.array([floor_prob + self.rng.random()
        #                       for _ in range(self.num_chars)])
        # self.probs /= np.sum(self.probs)
        self.probs = np.ones(self.num_chars) / self.num_chars
        print(f'self.probs {self.probs}')

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

    def make_inputs(self, num_inputs, min_len, max_len):
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


def example1a():
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
        avg_loss, accuracy = do_epoch(dataloader, model, optimizer, pad_idx)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            validate(model, dataset)

    print(f"final params:")
    print_params(model)
    validate(model, dataset, print_probs=True)


def example1b():
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

    model = Example1(v, inverse_class_probs)
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
            dataloader_train, model, optimizer, pad_idx)
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
    num_epochs = 200
    print_freq = 10

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

    model = Example1c(v, inverse_class_probs)
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
            dataloader_train, model, optimizer, pad_idx)
        if epoch % print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f'epoch {epoch}: ' +
                  f'accuracy {accuracy:.5f}, loss {avg_loss:.5f}')
            print_errs = epoch == num_epochs-1
            validate(model, dataset_validate, print_errs=print_errs)

    torch.save(model.state_dict(), 'example1c.pth')
    print(f"final params:")
    print_params(model, precision=3)
    num_instances = 10
    print(f'probs for first {num_instances} validation instances:')
    validate(model, dataset_validate,
             print_probs=True, num_instances=num_instances,
             id_to_token=id_to_token)


def example1():
    token_to_id = {'a': 0,
                   'b': 1,
                   'E': 2,
                   'P': 3,
                   }
    # id_to_token = dict(map(reversed, token_to_id.items()))
    v = len(token_to_id)
    pad_idx = token_to_id['P']

    strs, labels = zip(('baE', 'b'),
                       )
    # seqs = [torch.tensor([token_to_id[c] for c in s]) for s in strs]
    # labels = [torch.tensor(token_to_id[c]) for c in labels]
    seqs, labels = convert_to_IDs(token_to_id, strs, labels)
    print(f'sequences {seqs}')
    print(f'labels {labels}')
    dataset = Data(seqs, labels)
    model = Example1(v, init_with_zeros=True)
    with torch.no_grad():
        model.R.data = torch.tensor([[0, 2, 0, -INF],
                                     [0.5, 1.5, 0, -INF],
                                     [0.2, 0.1, 0, -INF],
                                     [-INF, -INF, -INF, 0]])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # max_steps = 30
    print(f"initial params:")
    print_params(model)

    x, y = dataset[0]
    evaluate_gradient(model, x, y)
    return


def main():
    example1c()


if __name__ == "__main__":
    main()
