import torch
import torch.nn.functional as torchfunc

torch.set_printoptions(precision=3)

token_to_id = {'a': 0,
               'b': 1,
               'E': 2,
               'P': 3,
               }
id_to_token = dict(map(reversed, token_to_id.items()))
print(f'token IDs: {token_to_id}')

v = len(token_to_id)
print(f'v: {v}')

s = ['b', 'a', 'E']
# s = ['b', 'b', 'b', 'b', 'a', 'a', 'E']
print(f's: {s}')

n = len(s)
print(f'n: {n}')

s_ids = torch.tensor([token_to_id[t] for t in s])
assert s_ids.shape == (n,)
print(f's_ids: {s_ids}')

X = torchfunc.one_hot(s_ids, num_classes=v).float()
assert X.shape == (n, v)
print(f'X: {X}')

INF = 1e9

R = torch.tensor([[0, 2, 0, -INF],
                  [0.5, 1.5, 0, -INF],
                  [0.2, 0.1, 0, -INF],
                  [-INF, -INF, -INF, 0]])
assert R.shape == (v, v)
print(f'R: {R}')

A_pre = X @ R @ X.transpose(dim0=0, dim1=1)
assert A_pre.shape == (n, n)
print(f'A_pre: {A_pre}')


def make_mask(n_val):
    lower_tri = torch.tril(torch.ones(n_val, n_val))
    upper_tri_bool = (lower_tri == 0)
    zeros = torch.zeros(n_val, n_val)
    mask = zeros.masked_fill(upper_tri_bool, -INF)
    return mask


M_mask = make_mask(n)
assert M_mask.shape == (n, n)
print(f'M_mask: {M_mask}')

A_mskd = A_pre + M_mask
assert A_mskd.shape == (n, n)
print(f'A_mskd: {A_mskd}')

A = torchfunc.softmax(A_mskd, dim=1)
assert A.shape == (n, n)
print(f'A: {A}')

X_prime = A @ X
assert X_prime.shape == (n, v)
print(f'X_prime: {X_prime}')

X_prime_last_row = X_prime[-1, :]
assert X_prime_last_row.shape == (v,)
print(f'X_prime_last_row: {X_prime_last_row}')

token_probs = torchfunc.softmax(X_prime_last_row, dim=0)
assert token_probs.shape == (v,)
print(f'token_probs: {token_probs}')

i_next = torch.argmax(token_probs).item()
print(f'i_next: {i_next}')

next_token = id_to_token[i_next]
print(f'next_token: {next_token}')

logits = X_prime_last_row.unsqueeze(0)
assert logits.shape == (1, v)


ground_truth_tok = 'b'
ground_truth_ID = torch.tensor([token_to_id[ground_truth_tok]])
assert ground_truth_ID.shape == (1,)

loss = torchfunc.cross_entropy(logits, ground_truth_ID)
assert loss.shape == ()


print(f'loss assuming ground truth is "{ground_truth_tok}": {loss.item():.4f}')
