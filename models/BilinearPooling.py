import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearPooling(nn.Module):
    def __init__(self, input_size):
        super(BilinearPooling, self).__init__()
        self.fc = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x1, x2):
        # Reshape input tensors to have shape (batch_size, -1)
        # x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)

        # Apply fully connected layer to x2
        x2 = self.fc(x2)

        # Compute outer product
        outer_product = torch.matmul(x1.unsqueeze(2), x2.unsqueeze(1))

        # Sum along dimensions 1 and 2 (element-wise addition)
        bilinear_pool = torch.sum(outer_product, dim=(1, 2))

        # L2 normalization
        bilinear_pool = F.normalize(bilinear_pool, p=2, dim=1)
        print("21222")
        return bilinear_pool

# Example usage
input_size = 100  # Adjust the input size according to your specific use case
bilinear_pooling = BilinearPooling(input_size)

# Generate example input tensors
x1 = torch.randn(32, input_size)  # Batch size of 32
x2 = torch.randn(32, input_size)

# Forward pass
output = bilinear_pooling(x1, x2)
print(output)