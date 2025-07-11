# utils/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec
import os
from typing import List, Dict, Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class BusSystemVisualizer:
    def __init__(self):
        """Inicializa el visualizador con configuraciones predeterminadas."""
        sns.set_style("whitegrid")
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_training_progress(self, rewards: List[float], epsilons: List[float], 
                            losses: Optional[List[float]] = None) -> None:
        """
        Visualiza el progreso del entrenamiento del agente.
        
        Args:
            rewards: Lista de recompensas por episodio
            epsilons: Lista de valores epsilon por episodio
            losses: Lista opcional de pérdidas por episodio
        """
        plt.figure(figsize=(15, 5))
        
        # Gráfico de recompensas
        plt.subplot(1, 3, 1)
        rolling_mean = pd.Series(rewards).rolling(10, min_periods=1).mean()
        plt.plot(rewards, alpha=0.3, label='Por episodio')
        plt.plot(rolling_mean, label='Media móvil (10)')
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.title("Recompensas durante el Entrenamiento")
        plt.legend()
        
        # Gráfico de exploración (epsilon)
        plt.subplot(1, 3, 2)
        plt.plot(epsilons)
        plt.xlabel("Episodio")
        plt.ylabel("Tasa de Exploración (ε)")
        plt.title("Decaimiento de la Exploración")
        
        # Gráfico de pérdidas si está disponible
        if losses:
            plt.subplot(1, 3, 3)
            plt.plot(losses)
            plt.xlabel("Paso de Entrenamiento")
            plt.ylabel("Pérdida")
            plt.title("Evolución de la Pérdida")
        
        plt.tight_layout()
        plt.show()
    
    def plot_action_distribution(self, action_counts: List[int], 
                               action_names: Optional[List[str]] = None) -> None:
        """
        Visualiza la distribución de acciones tomadas por el agente.
        
        Args:
            action_counts: Lista con conteo de cada acción
            action_names: Nombres descriptivos para cada acción
        """
        if not action_names:
            action_names = [f"Acción {i}" for i in range(len(action_counts))]
            
        plt.figure(figsize=(8, 5))
        bars = plt.bar(action_names, action_counts, color=self.color_palette)
        plt.title("Distribución de Acciones")
        plt.ylabel("Frecuencia")
        
        # Añadir valores encima de las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.show()
    
    def plot_episode_analysis(self, episode_data: List[Dict]) -> None:
        """
        Analiza un episodio específico con gráficos detallados.
        
        Args:
            episode_data: Lista de diccionarios con datos del episodio
        """
        df = pd.DataFrame(episode_data)
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Gráfico de pasajeros vs hora
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=df, x='hora', y='pasajeros', ax=ax1)
        ax1.set_title("Pasajeros por Hora")
        ax1.set_ylabel("Número de Pasajeros")
        
        # Gráfico de tráfico
        ax2 = fig.add_subplot(gs[0, 1])
        sns.countplot(data=df, x='trafico', ax=ax2, 
                     order=["Fluido", "Congestionado"])
        ax2.set_title("Distribución de Estados de Tráfico")
        
        # Gráfico de acciones
        ax3 = fig.add_subplot(gs[1, :])
        sns.countplot(data=df, x='hora', hue='accion', ax=ax3)
        ax3.set_title("Acciones por Hora")
        ax3.legend(title="Acción")
        
        # Gráfico de recompensas acumuladas
        ax4 = fig.add_subplot(gs[2, 0])
        df['recompensa_acumulada'] = df.groupby('episodio')['recompensa'].cumsum()
        sns.lineplot(data=df, x='hora', y='recompensa_acumulada', 
                    hue='episodio', ax=ax4)
        ax4.set_title("Recompensa Acumulada por Episodio")
        
        # Gráfico de recompensas por acción
        ax5 = fig.add_subplot(gs[2, 1])
        sns.boxplot(data=df, x='accion', y='recompensa', ax=ax5)
        ax5.set_title("Distribución de Recompensas por Acción")
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, test_results: List[Dict]) -> Optional[object]:
        """
        Crea un dashboard interactivo con Plotly (si está disponible).
        
        Args:
            test_results: Datos de prueba para visualizar
            
        Returns:
            Objeto del dashboard o None si Plotly no está disponible
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly no está instalado. No se puede crear el dashboard interactivo.")
            return None
            
        df = pd.DataFrame(test_results)
        
        # Gráfico interactivo de pasajeros vs hora
        fig1 = px.line(df, x='hora', y='pasajeros', color='episodio',
                      title="Pasajeros por Hora en Diferentes Episodios")
        
        # Gráfico de acciones por condición de tráfico
        fig2 = px.sunburst(df, path=['trafico', 'accion'], 
                          title="Distribución de Acciones por Estado de Tráfico")
        
        # Gráfico de recompensas
        fig3 = px.box(df, x='accion', y='recompensa', color='trafico',
                     title="Recompensas por Acción y Estado de Tráfico")
        
        return fig1, fig2, fig3
    
    def save_visualizations(self, prefix: str = "visualization") -> None:
        """
        Guarda las visualizaciones actuales en archivos PNG.
        
        Args:
            prefix: Prefijo para los nombres de archivo
        """
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        
        for i, fig in enumerate(plt.get_fignums(), start=1):
            plt.figure(fig)
            plt.savefig(f"visualizations/{prefix}_{i}.png", 
                       dpi=300, bbox_inches='tight')