import torch

# Outputs could be different on different architectures.
# Results below are for Intel Core i5-1335U CPU.

a = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # shape [1, 4]
b = torch.tensor([[0.5], [0.6], [0.7], [0.8]])  # shape [4, 1]
b_unsq = b.unsqueeze(0)  # shape [1, 4, 1]

# product of 1x4 and 4x1 tensors, result is 1x1
prod = torch.matmul(a, b)  # yields tensor([[0.699999988]])

# product of 1x4 and 1x4x1 tensors, result is 1x1x1
# yields different result, tensor([[[0.700000048]]])
prod_unsq = torch.matmul(a, b_unsq)

torch.set_printoptions(precision=9)
print(f'prod:\n{prod}')
print(f'prod_unsq:\n{prod_unsq}')
# prints 'tensor([[[-5.960464478e-08]]])'
print(f'discrepancy:\n{prod-prod_unsq}')

# The discrepancy of 5.960464478e-08 is exactly 2^(-24).
# The binary float32 representations of the two products differ
# only their least significant bit:
# 0x3f333333
print(f'prod in hex:      {hex(prod.view(torch.int32).item())}')
print(f'prod_unsq in hex: {
      hex(prod_unsq.view(torch.int32).item())}')  # 0x3f333334
