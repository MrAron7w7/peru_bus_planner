# environment/peru_bus_env.py (actualizado)
import gym
import numpy as np
from gym import spaces

class PeruBusEnv(gym.Env):
    def __init__(self):
        super(PeruBusEnv, self).__init__()

        # Especifica dtype=np.float32 para el espacio de observación
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([100, 1, 23], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.state = None
        self.current_hour = None
        self.max_hours = 10
        self.counter = 0

        self.reset()

    def reset(self):
        self.counter = 0
        self.current_hour = np.random.choice([6, 9, 12, 15, 18, 21])
        pasajeros, trafico = self.simular_condiciones(self.current_hour)
        # Asegura que el estado sea float32
        self.state = np.array([pasajeros, trafico, self.current_hour], dtype=np.float32)
        return self.state

    def step(self, action):
        pasajeros, trafico, hora = self.state

        reward = 0.0  # Asegura que reward sea float
        if action == 0:  # Avanzar
            reward = float(pasajeros * (0.5 if trafico == 0 else 0.2))
        else:  # Esperar
            new_passengers = np.random.randint(5, 20)
            pasajeros = min(100, pasajeros + new_passengers)
            reward = float(new_passengers * 0.3)

        self.current_hour = (self.current_hour + 1) % 24
        self.counter += 1

        pasajeros, trafico = self.simular_condiciones(self.current_hour)
        # Asegura que el nuevo estado sea float32
        self.state = np.array([pasajeros, trafico, self.current_hour], dtype=np.float32)

        done = self.counter >= self.max_hours
        return self.state, reward, done, {}

    def simular_condiciones(self, hora):
        # Los valores devueltos serán convertidos a float32 en reset()/step()
        if 6 <= hora <= 8:
            return np.random.randint(60, 100), np.random.choice([0, 1], p=[0.3, 0.7])
        elif 9 <= hora <= 11:
            return np.random.randint(20, 50), np.random.choice([0, 1], p=[0.7, 0.3])
        elif 12 <= hora <= 14:
            return np.random.randint(40, 70), np.random.choice([0, 1], p=[0.5, 0.5])
        elif 15 <= hora <= 17:
            return np.random.randint(10, 30), np.random.choice([0, 1], p=[0.8, 0.2])
        elif 18 <= hora <= 20:
            return np.random.randint(70, 100), np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            return np.random.randint(5, 20), np.random.choice([0, 1], p=[0.9, 0.1])