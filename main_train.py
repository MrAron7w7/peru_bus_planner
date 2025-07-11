# main_train.py (actualizado)
from environment.peru_bus_env import PeruBusEnv
from agent.dqn_agent import DQNAgent
from utils.visualizer import BusSystemVisualizer
import numpy as np
import os
from typing import List

def train() -> None:
    # Configuración inicial
    env = PeruBusEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    visualizer = BusSystemVisualizer()

    # Parámetros de entrenamiento
    episodes = 50
    max_steps = 10
    batch_size = 32

    # Métricas a registrar
    rewards_per_episode: List[float] = []
    epsilons: List[float] = []
    losses: List[float] = []

    # Ciclo de entrenamiento
    for e in range(episodes):
        state = env.reset()
        total_reward = 0.0
        episode_losses: List[float] = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            
            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            if done:
                break

        # Registrar métricas del episodio
        rewards_per_episode.append(total_reward)
        epsilons.append(agent.epsilon)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        print(f"🎯 Episodio {e+1}/{episodes} | Recompensa={total_reward:.2f} | ε={agent.epsilon:.3f}")

    # Guardar modelo
    if not os.path.exists("models"):
        os.makedirs("models")
    agent.save("models/best_model.pth")
    print("✅ Modelo guardado en 'models/best_model.pth'")

    # Visualización de resultados
    visualizer.plot_training_progress(rewards_per_episode, epsilons, losses)
    visualizer.plot_action_distribution(agent.action_counts, ['Avanzar', 'Esperar'])
    visualizer.save_visualizations("training_results")

if __name__ == "__main__":
    train()