# `matmul` gives different result on unsqueezed input

`torch.matmul()` often produces tiny discrepancies when computing the matrix product of tensors `a` and `b`, depending on whether `b` has a leading singleton dimension. That is, for many `a` and `b`, we find `torch.matmul(a, b)` is not equal to `torch.matmul(a, b.unsqueeze(0))`. This isn't a bug; it's a known numerical issue relating to floating-point computation in general and not PyTorch. The issue is related to [#34060](https://github.com/pytorch/pytorch/issues/34060) and perhaps [#15359](https://github.com/pytorch/pytorch/issues/15359), but I'm suggesting a different angle here. My question for the community is, could/should we add a small amount of extra documentation in matmul, something like "Warning: Singleton dimensions in the inputs can lead to small differences in the output, compared to the same input with singleton dimensions removed via `squeeze()`". Thanks for any feedback.

## Details

An example in code is given below, but the issue will be explained in a prose example first.
Consider `a = [[0.1, 0.2, 0.3, 0.4]]` (shape 1x4) and `b=[[0.5], [0.6], [0.7], [0.8]]` (shape 4x1). We find `torch.matmul(a,b)` is `[[0.699999988]]` (shape 1x1). 

Now let `b_unsq = b.unsqueeze(0)`, so we have `b_unsq` is `[[[0.5], [0.6], [0.7], [0.8]]]` (shape 1x4x1). The extra leading singleton dimension of `b` (dimension 0) might be expected to have no meaningful effect on the matrix product. If this dimension were not a singleton, then tensor `a` would be broadcast over dimension 0 of `b`. But no broadcast is needed for a singleton dimension. So we might expect the matrix product to produce the same numerical answer as before. But in fact, the result can be very slightly different. On my CPU, I find that `torch.matmul(a, b_unsq)` is `[[[0.700000048]]]`. The amount of the discrepancy is -5.960464478e-08, or _exactly_ 2^(-24). The binary float32 representations of the two results differ only their least significant bit.

It's not a bug. Presumably, the matrix multiplication code uses a different algorithm for inputs `b` that are 2-dimensional (which definitely require no broadcast), compared to inputs that have 3 or more dimensions (which may require broadcast). The different algorithms may cause the 4 terms produced by the matrix multiplication to be added in a different order, and it is well known that this can lead to numerical differences (search the web for "floating point addition not associative").

Nevertheless, this can be a serious gotcha. For example, after a large number of training iterations, the minuscule discrepancy described above can become arbitrarily large. So, a developer attempting to debug two variants of code that are believed to do the same thing (except for some singleton dimensions that are wrongly assumed to have no effect) can be left with inexplicably different outcomes.

My question for the community is: would it be worth adding a warning about this to the to the documentation of matmul? Something like, "warning: Singleton dimensions in the inputs can lead to small differences in the output, compared to the same input with singleton dimensions removed via squeeze()". Thanks for any suggestions or feedback.

## Example code

```
import torch

# Outputs could be different on different architectures. 
# Results below are for Intel Core i5-1335U CPU.

a = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # shape [1, 4]
b = torch.tensor([[0.5], [0.6], [0.7], [0.8]])  # shape [4, 1]
b_unsq = b.unsqueeze(0)  # shape [1, 4, 1]

# product of 1x4 and 4x1 tensors, result is 1x1
prod = torch.matmul(a, b) # yields tensor([[0.699999988]])

# product of 1x4 and 1x4x1 tensors, result is 1x1x1
prod_unsq = torch.matmul(a, b_unsq) # yields different result, tensor([[[0.700000048]]])

torch.set_printoptions(precision=9)
print(f'prod:\n{prod}')
print(f'prod_unsq:\n{prod_unsq}')
print(f'discrepancy:\n{prod-prod_unsq}') # prints 'tensor([[[-5.960464478e-08]]])'

# The discrepancy of 5.960464478e-08 is exactly 2^(-24).
# The binary float32 representations of the two products differ
# only their least significant bit:
print(f'prod in hex:      {hex(prod.view(torch.int32).item())}')      # 0x3f333333
print(f'prod_unsq in hex: {hex(prod_unsq.view(torch.int32).item())}') # 0x3f333334
```