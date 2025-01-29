import torch
import torch.nn as nn

class PixelLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PixelLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM with multiple layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to produce a single float output (logit for BCEWithLogitsLoss)
        self.fc = nn.Linear(hidden_dim, 1)
        
               
    def forward(self, x, hidden):
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply the fully connected layer to each timestep output
        output = self.fc(lstm_out)
        
        # Apply sigmoid to get output in range [0, 1]
        output = torch.sigmoid(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))