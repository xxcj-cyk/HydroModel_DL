import torch
import torch.nn as nn
import torch.nn.functional as F

def test_lstm():
    # Define model parameters
    input_size = 10         # Number of input features
    output_size = 1         # Number of output features
    hidden_sizes = [32, 64] # Hidden layer sizes
    seq_len = 5             # Sequence length
    batch_size = 8          # Batch size
    dropout_rate = 0.1      # Dropout rate

    # Create the model
    model = LSTM(input_size, output_size, hidden_sizes, dr=dropout_rate)

    # Print the model architecture
    print(model)

    # Generate random input data
    x = torch.randn(seq_len, batch_size, input_size)  # (seq_len, batch_size, input_size)

    # Perform a forward pass
    output = model(x)

    # Print the output shape
    print(f"Output shape: {output.shape}")

# Run the test function
test_lstm()
