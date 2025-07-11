# 🚍 Proyecto: Planificación Dinámica de Paraderos y Horarios en Perú Bus

Este proyecto implementa una **demostración básica** de un **Sistema Inteligente basado en Aprendizaje Reforzado** para optimizar **paraderos** y **horarios** de la empresa **Perú Bus**.

## 📚 Contexto
Actualmente, Perú Bus enfrenta problemas con una planificación estática de rutas y horarios, lo cual impacta negativamente en la eficiencia del servicio y la experiencia del usuario.  
Este sistema busca resolver eso, utilizando técnicas de **Inteligencia Artificial**.

## 🎯 Objetivo
Diseñar y demostrar un entorno simulado donde un **agente inteligente** pueda:
- Decidir si avanzar o esperar más pasajeros.
- Adaptarse dinámicamente a las condiciones de tráfico.
- Mejorar la eficiencia operativa.

## 🏗️ Estructura del proyecto
peru_bus_planner/ │ ├── main.py ├── environment/peru_bus_env.py ├── agent/dqn_agent.py ├── data/simulated_data.csv ├── utils/visualization.py ├── models/ └── README.md

## 🛠️ Tecnologías utilizadas
- Python
- Gym (OpenAI)
- NumPy
- Pandas
- Matplotlib

## 🚀 Cómo ejecutar
1. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
2. Corre el programa principal:
    ```bash
    python main.py
    ```

## 🤖 Próximos pasos
- Entrenar un agente inteligente usando DQN (Deep Q-Learning).
- Mejorar la simulación añadiendo múltiples paraderos y rutas.
- Incorporar predicciones de tráfico basadas en datos reales.

---
Desarrollado por estudiantes de **Ingeniería de Computación y Sistemas** - **Universidad Privada San Juan Bautista**.# peru_bus_planner
