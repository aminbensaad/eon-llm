import torch

# Test if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Create a tensor and move it to the GPU
x = torch.rand(10000, 10000).cuda()
y = torch.rand(10000, 10000).cuda()

# Perform a matrix multiplication
z = torch.matmul(x, y)
print("Matrix multiplication result shape:", z.shape)
