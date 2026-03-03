import torch
import torch.nn.functional as torchfunc

torch.set_printoptions(precision=3)

token_to_id = {'a': 0,
               'b': 1,
               'E': 2,
               }
id_to_token = dict(map(reversed, token_to_id.items()))
print(f'token IDs: {token_to_id}')

v = len(token_to_id)
print(f'v: {v}')

s = ['b', 'b', 'a', 'E']
# s = ['b', 'b', 'b', 'b', 'a', 'a', 'E']
print(f's: {s}')

E_loc = s.index('E')
assert E_loc == len(s) - 1

n = len(s)
print(f'n: {n}')

COPY_LOC = 2
print(f'copying from location: {COPY_LOC}')

s_ids = torch.tensor([token_to_id[t] for t in s])
assert s_ids.shape == (n,)
print(f's_ids: {s_ids}')

X = torchfunc.one_hot(s_ids, num_classes=v).float()
assert X.shape == (n, v)
print(f'X without position encoding: \n{X}')

P = torch.eye(n)
assert P.shape == (n, n)

d = v+n
X = torch.cat((X, P), dim=1)
assert X.shape == (n, d)
print(f'X with position encoding: \n{X}')



INF = 1e9
R = torch.full((d, d), -INF)
print(f'about to set R[{d-1}, {v + COPY_LOC}] = {0}')
R[d-1, v + COPY_LOC] = 0
R = R.float()

assert R.shape == (d, d)
print(f'R: {R}')

print(f'XR: {X @ R}')

print(f'X^T: {X.transpose(dim0=0, dim1=1)}')

A_pre = X @ R @ X.transpose(dim0=0, dim1=1)
assert A_pre.shape == (n, n)
print(f'A_pre: {A_pre}')


def make_mask(n_val):
    lower_tri = torch.tril(torch.ones(n_val, n_val))
    upper_tri_bool = (lower_tri == 0)
    zeros = torch.zeros(n_val, n_val)
    mask = zeros.masked_fill(upper_tri_bool, -INF)
    return mask

A = torchfunc.softmax(A_pre, dim=1)
assert A.shape == (n, n)
print(f'A: {A}')

U = torch.vstack( (torch.eye(v), torch.zeros(n,v)) )
U *= 100 # i.e., high temperature softmax
assert U.shape == (d, v)
print(f'U: {U}')

print(f'A @ X: {A @ X}')

X_prime = A @ X @ U
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


ground_truth_tok = s[COPY_LOC]
ground_truth_ID = torch.tensor([token_to_id[ground_truth_tok]])
assert ground_truth_ID.shape == (1,)

loss = torchfunc.cross_entropy(logits, ground_truth_ID)
assert loss.shape == ()


print(f'loss assuming ground truth is "{ground_truth_tok}": {loss.item():.4f}')
