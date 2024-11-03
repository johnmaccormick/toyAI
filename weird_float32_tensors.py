import torch

torch.set_printoptions(precision=12)


class PVS():
    def __init__(self, d):
        # self.p = d['p']
        # self.v = d['v']
        # self.s = d['s']
        self.p = d['p'].squeeze(0)
        self.v = d['v'].squeeze(0)
        self.s = d['s'].squeeze(0)
        # self.p = d['p'].squeeze(0).squeeze(0)
        # self.v = d['v'].squeeze(0).squeeze(0)
        # self.s = d['s'].squeeze(0).squeeze(0)
        self.sm = torch.matmul(self.p, self.v)
        # self.sm = torch.matmul(self.p.squeeze(), self.v.squeeze())
        self.smdiff = self.sm - self.s

    def __str__(self):
        return f'p.shape: {self.p.shape}\n' + f'v.shape: {self.v.shape}\n' + f's.shape: {self.s.shape}\n' + f'sm.shape: {self.sm.shape}\n' + f'smdiff.shape: {self.smdiff.shape}' \
            + f'p: {self.p}\n' + f'v: {self.v}\n' + f's: {self.s}\n' + \
            f'sm: {self.sm}\n' + f'smdiff: {self.smdiff}'


d = torch.load('results/single.pt')
# for key, val in d.items():
#     print(key, val)
single = PVS(d)


d = torch.load('results/compact.pt')
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
