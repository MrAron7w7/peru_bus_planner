# main_test.py (actualizado)
from environment.peru_bus_env import PeruBusEnv
from agent.dqn_agent import DQNAgent
from utils.visualizer import BusSystemVisualizer
import torch
import pandas as pd
import os
from typing import List, Dict

def test() -> None:
    # Configuración inicial
    env = PeruBusEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    visualizer = BusSystemVisualizer()
    
    # Cargar modelo entrenado
    if not os.path.exists("models/best_model.pth"):
        raise FileNotFoundError("No se encontró el modelo entrenado. Ejecuta main_train.py primero.")
    
    agent.load("models/best_model.pth")
    print("✅ Modelo cargado desde 'models/best_model.pth'")

    # Configuración de prueba
    acciones = {0: "Avanzar", 1: "Esperar"}
    resultados: List[Dict] = []
    episodes = 5

    # Ejecutar pruebas
    for e in range(episodes):
        state = env.reset()
        total_reward = 0.0
        print(f"\n🚍 Episodio {e+1}/{episodes}")

        for step in range(10):  # 10 pasos por episodio
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action = torch.argmax(agent.model(state_tensor), dim=1).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            pasajeros, trafico, hora = state
            trafico_estado = "Fluido" if trafico == 0 else "Congestionado"

            print(f"Hora {int(hora)}:00 | Pasajeros={int(pasajeros)} | "
                  f"Tráfico={trafico_estado} | Acción={acciones[action]} | "
                  f"Recompensa={reward:.2f}")
            
            resultados.append({
                "episodio": e+1,
                "hora": int(hora),
                "pasajeros": int(pasajeros),
                "trafico": trafico_estado,
                "accion": acciones[action],
                "recompensa": reward
            })

            state = next_state
            if done:
                break

        print(f"🏁 Recompensa total: {total_reward:.2f}")

    # Guardar resultados
    if not os.path.exists("data"):
        os.makedirs("data")
    
    df = pd.DataFrame(resultados)
    df.to_csv("data/test_summary.csv", index=False)
    print("\n📄 Resultados guardados en 'data/test_summary.csv'")

    # Visualización
    visualizer.plot_episode_analysis(resultados)
    
    # Dashboard interactivo (si Plotly está disponible)
    figs = visualizer.create_interactive_dashboard(resultados)
    if figs and PLOTLY_AVAILABLE:
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        
        for i, fig in enumerate(figs, start=1):
            fig.write_html(f"visualizations/dashboard_{i}.html")
        print("📊 Dashboards interactivos guardados en 'visualizations/'")

if __name__ == "__main__":
    test()