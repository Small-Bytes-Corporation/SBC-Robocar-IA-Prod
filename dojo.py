# **************************************************************************** #
#                                                                              #
#                        Robocar - Model Trainer                               #
#                                                                              #
# **************************************************************************** #

from torch.utils.data import Dataset, DataLoader, random_split
import csv
import torch
import sys
from torch import nn
from agent import NRAgent
from settings import (INPUT_SIZE, EPOCHS, LEARNING_RATE, BATCH_SIZE)

# **************************************************************************** #
#                                                                              #
#                            DATASET DEFINITION                                #
#                                                                              #
# **************************************************************************** #

class NRAgentDataset(Dataset):
    def __init__(self, data_file):
        self.states, self.actions = self.load_data(data_file)

    def load_data(self, csv_path):
        states, actions = [], []
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader):
                try:
                    state = eval(row[0])
                    action = eval(row[1])
                    states.append(state)
                    actions.append(action)
                except Exception as e:
                    print(f"Line error {i}: {e} | Data: {row}")
        print(f"Loaded {len(states)} lines on {i+1}")
        return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# **************************************************************************** #
#                                                                              #
#                          EVALUATION FUNCTION                                 #
#                                                                              #
# **************************************************************************** #

def evaluate(agent, test_loader):
    agent.eval()
    loss_function = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for states, actions in test_loader:
            predictions = agent(states)
            loss = loss_function(predictions, actions)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test MSE Loss: {test_loss:.4f}")

# **************************************************************************** #
#                                                                              #
#                              MAIN EXECUTION                                  #
#                                                                              #
# **************************************************************************** #

csv_path = sys.argv[1]
dataset = NRAgentDataset(csv_path)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset : {len(dataset)} samples")
print(f"Train : {len(train_dataset)}, Validation : {len(val_dataset)}")

agent = NRAgent(input_size=INPUT_SIZE)

model_path = None
agent.train_model(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, model_path=model_path)
agent.save_model()
