import torch
import os

# Make sure to import your model class
from train_torch import STRNNCell  # Assuming STRNNCell is defined in train_torch.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your model architecture
dim = 13  # Set the appropriate dimension here based on your model
strnn_model = STRNNCell(dim).to(device)

# Define the path to the saved model
model_file_path = './saved_model/strnn_model.pth'

# Load the model's state dictionary
if os.path.exists(model_file_path):
    strnn_model.load_state_dict(torch.load(model_file_path))
    strnn_model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_file_path}")
else:
    print(f"Model file not found at {model_file_path}")

# Add any testing or inference code you need here, such as evaluating the model on new data
