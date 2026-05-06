import torch

# 1. Creating tensors
print("=== Tensor Creation ===")
t_from_list = torch.tensor([1.0, 2.0, 3.0])
t_random = torch.randn(3, 3)
print(f"From list: {t_from_list}")
print(f"Random 3x3:\n{t_random}")

# 2. Basic operations
print("\n=== Basic Operations ===")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"Element-wise addition: {a + b}")

m1 = torch.randn(2, 3)
m2 = torch.randn(3, 2)
print(f"Matrix multiplication (2x3 & 3x2):\n{torch.mm(m1, m2)}")

# 3. Autograd
print("\n=== Autograd ===")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x^2 + 2x + 1
y.backward()               # dy/dx = 2x + 2
print(f"x = {x.item()}, dy/dx = {x.grad.item()}")  # expected: x = 3.0, dy/dx = 8.0
