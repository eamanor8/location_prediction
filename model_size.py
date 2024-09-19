import torch
import os
from preprocess import STRNNModule  # Make sure to import your model class from the existing file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = STRNNModule().to(device)

# Calculate the total number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")

# Calculate the total size in bytes
param_size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
size_in_megabytes = param_size_in_bytes / (1024 ** 2)
print(f"Model size: {size_in_megabytes:.2f} MB")

# Save the model to a file
torch.save(model.state_dict(), 'model.pth')

# Check the size of the saved model on disk
file_size = os.path.getsize('model.pth') / (1024 ** 2)
print(f"Saved model size: {file_size:.2f} MB")
