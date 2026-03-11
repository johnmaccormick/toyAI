# Empirical experiment to verify that the R matrix is invariant to a different constant being added to each row, in the sense that the same output (X_prime) is produced.

import torch
import torch.nn.functional as torchfunc

torch.set_printoptions(precision=3)

def make_mask(n_val):
    lower_tri = torch.tril(torch.ones(n_val, n_val))
    upper_tri_bool = (lower_tri == 0)
    zeros = torch.zeros(n_val, n_val)
    mask = zeros.masked_fill(upper_tri_bool, -INF)
    return mask

token_to_id = {'a': 0,
               'b': 1,
               'E': 2,
               'P': 3,
               }
id_to_token = dict(map(reversed, token_to_id.items()))
print(f'token IDs: {token_to_id}')

v = len(token_to_id)
print(f'v: {v}')

# s = ['b', 'a', 'E']
s = ['b', 'b', 'b', 'b', 'a', 'a', 'E']
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

def compute_X_prime(R, X, use_mask):
    A_pre = X @ R @ X.transpose(dim0=0, dim1=1)
    assert A_pre.shape == (n, n)
    # print(f'A_pre: {A_pre}')

    M_mask = make_mask(n) if use_mask else torch.zeros(n, n)
    assert M_mask.shape == (n, n)
    # print(f'M_mask: {M_mask}')

    A_mskd = A_pre + M_mask
    assert A_mskd.shape == (n, n)
    # print(f'A_mskd: {A_mskd}')

    A = torchfunc.softmax(A_mskd, dim=1)
    assert A.shape == (n, n)
    # print(f'A: {A}')

    X_prime = A @ X
    assert X_prime.shape == (n, v)
    # print(f'X_prime: {X_prime}')

    return X_prime

def do_invariance_experiment():
    use_mask = False
    # R = torch.tensor([[0, 2, 0, -INF],
    #                 [0.5, 1.5, 0, -INF],
    #                 [0.2, 0.1, 0, -INF],
    #                 [-INF, -INF, -INF, 0]])

    
    # make R A random v by v tensor with values sampled from a standard normal distribution
    R = torch.randn(v, v)
    assert R.shape == (v, v)
    print(f'R: {R}')
    X_prime = compute_X_prime(R, X, use_mask)
    print(f'X_prime with original R: {X_prime}')

    # Add 10.0 to the second row of R
    # R_perturbed = R.clone()
    # R_perturbed[1, :] += 10.0
    # X_prime_perturbed = compute_X_prime(R_perturbed, X, use_mask)
    # print(f'X_prime with perturbed R: {X_prime_perturbed}')

    # Perturb every row of R by a different constant: specifically, add 10, 20, 30, ... to respective rows
    R_perturbed = R.clone()
    for i in range(v):
        R_perturbed[i, :] += (i + 1) * 10.0
    print(f'R_perturbed: {R_perturbed}')
    X_prime_perturbed = compute_X_prime(R_perturbed, X, use_mask)
    print(f'X_prime with perturbed R: {X_prime_perturbed}')

    # Assert that the two versions of X_prime are equal subject to numerical round off
    assert torch.allclose(X_prime, X_prime_perturbed, atol=1e-6)
    print('X_prime is invariant to the perturbation of R')



def main():
    do_invariance_experiment()

if __name__ == "__main__":
    main()
