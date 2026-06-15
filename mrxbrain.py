import torch
import torch.nn as nn





#Policy network
class MrXPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, double_output_dim=2):
        super().__init__()

  
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc_double = nn.Linear(hidden_dim, double_output_dim)

    def forward(self, x):
   
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        action_logits = self.fc2(x)
        double_logits = self.fc_double(x)
        return action_logits, double_logits
    
#Policy network
class MrXValue(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

  
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
   
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
