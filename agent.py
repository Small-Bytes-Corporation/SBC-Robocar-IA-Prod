# **************************************************************************** #
#                                                                              #
#                               Robocar - AGENT                                #
#                                                                              #
# **************************************************************************** #

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from settings import INPUT_SIZE, MODEL_SAVE_PATH, NUM_THREADS

# **************************************************************************** #
#                                                                              #
#                                MODEL CLASS                                   #
#                                                                              #
# **************************************************************************** #

class NRAgent(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super(NRAgent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor)
        return action.squeeze(0).numpy()

    def save_model(self, path=MODEL_SAVE_PATH):
        torch.save(self.state_dict(), path)
        print(f"Model saved as {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Model loaded from {path}")

# **************************************************************************** #
#                                                                              #
#                                TRAINING LOOP                                 #
#                                                                              #
# **************************************************************************** #

    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001, model_path=None):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        loss_function = nn.SmoothL1Loss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        train_losses, val_losses = [], []
        torch.set_num_threads(NUM_THREADS)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for states, actions in train_loader:
                optimizer.zero_grad()
                predicted_actions = self(states)
                loss = loss_function(predicted_actions, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss * 1000 / len(train_loader))

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for states, actions in val_loader:
                    val_pred = self(states)
                    loss = loss_function(val_pred, actions)
                    val_loss += loss.item()
            val_losses.append(val_loss * 1000 / len(val_loader))

            scheduler.step()

            if epoch % 10 == 0:
                print(f"🔹 Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Smooth L1)")
        plt.title("Learning evolution")
        plt.legend()
        plt.show()
        print("Training completed!")
