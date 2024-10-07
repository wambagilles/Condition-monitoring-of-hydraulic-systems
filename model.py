import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_var_in):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(n_var_in, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 100, batch_first=True)
        
    def forward(self, x):
        # Input shape: (batch_size, n_var_in, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # LSTM expects (batch_size, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        encoder_last = h_n[-1]  # Get the last hidden state
        return lstm_out, encoder_last

class Decoder(nn.Module):
    def __init__(self, n_var_a_priori, n_neurons_out):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(n_var_a_priori, 100, batch_first=True)
        self.attention = nn.Linear(100, 100)
        self.fc1 = nn.Linear(200, 64)  # Z = concatenate(context, decoder)
        self.fc2 = nn.Linear(64, n_neurons_out)

    def forward(self, decoder_input, encoder_outputs, encoder_last):
        # Initial state from encoder's last hidden state
        lstm_out, _ = self.lstm(decoder_input, (encoder_last.unsqueeze(0), encoder_last.unsqueeze(0)))

        # Compute attention
        attention_scores = torch.bmm(lstm_out, encoder_outputs.permute(0, 2, 1))  # Dot product
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector
        context = torch.bmm(attention_weights, encoder_outputs)
        
        # Combine context with decoder output (concatenate)
        decoder_combined_context = torch.cat((context, lstm_out), dim=-1)
        
        # Final dense layers
        output = F.relu(self.fc1(decoder_combined_context))
        output = self.fc2(output)  # Removed extra ReLU here for regression output
        return output

class AutoLagNet(nn.Module):
    def __init__(self, n_var_in, n_neurons_out, n_var_a_priori):
        super(AutoLagNet, self).__init__()
        self.encoder = Encoder(n_var_in)
        self.decoder = Decoder(n_var_a_priori, n_neurons_out)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, encoder_input, decoder_input):
        encoder_outputs, encoder_last = self.encoder(encoder_input)
        output = self.decoder(decoder_input, encoder_outputs, encoder_last)
        output = self.softmax(output)
        return output
