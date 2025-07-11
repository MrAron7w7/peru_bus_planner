# agent/dqn_agent.py (actualizado)
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).float().to(self.device)  # Añade .float() aquí
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.action_counts = [0] * action_size

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # Convierte a float32 explícitamente
            state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
            action = torch.argmax(self.model(state), dim=1).item()

        self.action_counts[action] += 1
        return action

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        
        # Prepara los tensores con el dtype correcto
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).float().to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).float().to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).float().to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).float().to(self.device)

        # Calcula los Q-valores actuales
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Calcula los Q-valores objetivo
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calcula la pérdida
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()