import torch
import numpy as np

torch.set_printoptions(precision=10)
torch.use_deterministic_algorithms(True)


class PVS():
    def __init__(self, d):
        # self.p = d['p']
        # self.v = d['v']
        # self.s = d['s']
        self.p = d['p'].squeeze(0)[..., -2:, :]
        self.v = d['v'].squeeze(0)
        self.s = d['s'].squeeze(0)[..., -2:, :]
        # self.p = d['p'].squeeze(0).squeeze(0)
        # self.v = d['v'].squeeze(0).squeeze(0)
        # self.s = d['s'].squeeze(0).squeeze(0)
        self.sm = torch.matmul(self.p, self.v)
        # self.sm = torch.matmul(self.p.squeeze(), self.v.squeeze())
        self.smdiff = self.sm - self.s

    def __str__(self):
        return f'p.shape: {self.p.shape}\n' + f'v.shape: {self.v.shape}\n' + f's.shape: {self.s.shape}\n' + f'sm.shape: {self.sm.shape}\n' \
            + f'smdiff.shape: {self.smdiff.shape}\n' \
            + f'p: {self.p}\n' + f'v: {self.v}\n' + f's: {self.s}\n' + \
            f'sm: {self.sm}\n' + f'smdiff: {self.smdiff}'


def compare_from_disk():
    d = torch.load('results/single.pt', weights_only=True)
    # for key, val in d.items():
    #     print(key, val)
    single = PVS(d)

    d = torch.load('results/compact.pt', weights_only=True)
    # for key, val in d.items():
    #     print(key, val)
    compact = PVS(d)

    print(f'single:\n{single}')
    print(f'compact:\n{compact}')

    p_diff = single.p - compact.p
    print(f'p_diff:\n{p_diff}')
    if torch.all(p_diff == 0.0):
        print(' -- all zeros')
    else:
        print(' -- NONZERO')

    v_diff = single.v - compact.v
    print(f'v_diff:\n{v_diff}')
    if torch.all(v_diff == 0.0):
        print(' -- all zeros')
    else:
        print(' -- NONZERO')

    s_diff = single.s - compact.s
    print(f's_diff:\n{s_diff}')
    if torch.all(s_diff == 0.0):
        print(' -- all zeros')
    else:
        print(' -- NONZERO')

    sm_diff = single.sm - compact.sm
    print(f'sm_diff:\n{sm_diff}')
    if torch.all(sm_diff == 0.0):
        print(' -- all zeros')
    else:
        print(' -- NONZERO')


def compare_squeeze_from_literals():
    p = torch.Tensor([[[0.2019751817, 0.2742579281, 0.2727965415, 0.2509703636, 0.0000000000],
                       [0.2173996717, 0.1838891804, 0.1844275892, 0.1930384338, 0.2212450802]]])
    v = torch.Tensor([[[2.1437067986],
                       [-0.4696947336],
                       [-0.5696251988],
                       [-0.9105849266],
                       [0.6614450216]]])
    s1 = torch.matmul(p, v)
    s2 = torch.matmul(p, v.squeeze(0))
    diff = s1 - s2
    print(f's1: {s1}')
    print(f's2: {s2}')
    print(f'diff: {diff}')
    print(f'nonzero?: {torch.any(diff != 0.0)}')


def compare_unsqueeze_from_literals():
    p = torch.Tensor([[0.2019751817, 0.2742579281, 0.2727965415, 0.2509703636, 0.0000000000],
                      [0.2173996717, 0.1838891804, 0.1844275892, 0.1930384338, 0.2212450802]])
    v = torch.Tensor([[2.1437067986],
                      [-0.4696947336],
                      [-0.5696251988],
                      [-0.9105849266],
                      [0.6614450216]])
    s1 = torch.matmul(p, v)
    s2 = torch.matmul(p, v.unsqueeze(0))
    diff = s1 - s2
    print(f's1: {s1}')
    print(f's2: {s2}')
    print(f'diff: {diff}')
    print(f'nonzero?: {torch.any(diff != 0.0)}')


def float32_as_exact(x: torch.Tensor):
    """Return a string describing the exact value of a PyTorch tensor containing 
    a single float32, assumed positive."""
    n = x.view(torch.int32).item()
    exponent = (n >> 23) - 127
    hidden_bit = 1 << 23
    significand_mask = (1 << 24) - 1
    significand = (n & significand_mask) | hidden_bit
    significand_as_float = float(significand) / hidden_bit
    return f'{significand_as_float} x 2^{exponent}'


def compare_many_unsqueezes():
    n1 = 1
    n2 = 4
    n3 = 1
    dtype = torch.float32
    iters = 100000
    num_diffs = 0
    max_diff = torch.tensor(0.0, dtype=dtype)
    for _ in range(iters):
        matrix1 = torch.randn(n1, n2, dtype=dtype)
        matrix2 = torch.randn(n2, n3, dtype=dtype)
        matrix2_unsqueezed = matrix2.unsqueeze(0)
        result = torch.matmul(matrix1, matrix2)
        result_unsqueezed = torch.matmul(matrix1, matrix2_unsqueezed)
        diff = result - result_unsqueezed
        if torch.any(diff != 0.0):
            num_diffs += 1
            max_diff = torch.maximum(max_diff, diff.abs().max())
    print(f'{num_diffs} diffs out of {iters} iters; '
          + f'max diff is exactly {float32_as_exact(max_diff)}')


def my_decompose(x: torch.Tensor):
    v = x.item()
    n = x.view(torch.int32).item()
    print(f'v: {v}')
    print(f'v.as_integer_ratio(): {v.as_integer_ratio()}')
    print(f'v.hex(): {v.hex()}')
    print(f'n: {n}')
    print(f'hex(n): {hex(n)}')
    print(f'hex(n>>23): {hex(n >> 23)}')
    print(f'n>>23: {n >> 23}')
    exponent = (n >> 23) - 127  # drop significand, correct exponent offset
    # 23 and 127 are specific to float32
    print(f'exponent: {exponent}')

    significand = n & (2**23 - 1)  # second factor provides mask
    print(f'significand: {significand}')
    print(f'hex(significand): {hex(significand)}')
    sig_with_hidden = (1 << 23) | significand
    print(f'sig_with_hidden: {sig_with_hidden}')
    print(f'hex(sig_with_hidden): {hex(sig_with_hidden)}')
    significand_float = float(sig_with_hidden) / (1 << 23)
    # to extract significand
    return exponent, significand_float


def main():
    # compare_from_disk()
    # compare_squeeze_from_literals()
    # compare_unsqueeze_from_literals()
    compare_many_unsqueezes()
    x = torch.tensor(5.960464478e-08, dtype=torch.float32)
    # exponent, significand = my_decompose2(x)
    # print(f'exponent {exponent}')
    # print(f'significand {significand}')
    print(float32_as_exact(x))


if __name__ == "__main__":
    main()
