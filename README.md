# Sección 1: Importación de bibliotecas necesarias
# numpy y matplotlib se utilizan para cálculos numéricos y visualización de resultados.
# defaultdict permite manejar la tabla Q, random y os para la manipulación del entorno y archivos.
# json se utiliza para la persistencia de datos, mientras que time gestiona pausas temporales.
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import json
import os
import time

# Sección 2: Clase MazeEnvironment
# Esta clase define el entorno del laberinto, incluyendo su generación, validación y mecánicas básicas de movimiento.
class MazeEnvironment:
    def __init__(self, size=100, seed=42):
        """
        Inicializa el entorno del laberinto con tamaño predeterminado y semilla para reproducibilidad.
        """
        self.size = size
        self._maze = None
        self.seed = seed
        self.start_pos = (1, 1)  # Posición inicial fija
        self.current_pos = self.start_pos
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Movimientos posibles: derecha, izquierda, abajo, arriba

    # Propiedad para acceder al laberinto y generarlo si aún no existe
  @property
    def maze(self):
        if self._maze is None:
            self._maze = self._create_maze()
        return self._maze

    # Generación del laberinto mediante valores aleatorios y configuración inicial
  def _create_maze(self):
        """
        Crea un laberinto utilizando un patrón predefinido y verifica que exista un camino válido.
        """
        # Inicialización de aleatoriedad
        random.seed(self.seed)
        np.random.seed(self.seed)

  maze = np.ones((self.size, self.size))  # Laberinto inicialmente lleno de paredes

        # Generación de caminos en celdas ímpares
 for i in range(1, self.size - 1, 2):
            for j in range(1, self.size - 1, 2):
                maze[i, j] = 0  # Crear espacio vacío
                # Crear caminos al azar hacia abajo o hacia la derecha
                if i < self.size - 2 and random.random() < 0.7:
                    maze[i + 1, j] = 0
                if j < self.size - 2 and random.random() < 0.7:
                    maze[i, j + 1] = 0

        # Configuración de inicio y meta
 maze[1, 1] = 0
        maze[self.size - 2, self.size - 2] = 2

        # Verificación de un camino válido hacia la meta
 if not self._verify_path_exists(maze):
            self._create_path_to_goal(maze)

        # Restaurar el estado de las semillas
random.seed()
        np.random.seed()

return maze

    # Verificación de caminos utilizando DFS
def _verify_path_exists(self, maze):
        """
        Verifica si existe un camino desde la posición inicial hasta la meta usando búsqueda en profundidad (DFS).
        """
        visited = set()

  def dfs(pos):
            if maze[pos] == 2:  # Meta alcanzada
                return True
            visited.add(pos)
            for dx, dy in self.actions:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if (0 <= new_pos[0] < self.size and
                    0 <= new_pos[1] < self.size and
                    maze[new_pos] != 1 and
                    new_pos not in visited):
                    if dfs(new_pos):
                        return True
            return False

return dfs(self.start_pos)

    # Función para crear un camino directo en caso de que no exista
 def _create_path_to_goal(self, maze):
        """
        Crea un camino directo desde la posición inicial hasta la meta.
        """
        current = self.start_pos
        goal = (self.size - 2, self.size - 2)

while current != goal:
            x, y = current
            if x < goal[0]:
                maze[x + 1, y] = 0
                current = (x + 1, y)
            elif y < goal[1]:
                maze[x, y + 1] = 0
                current = (x, y + 1)

# Sección 3: Clase PersistentQLearningAgent
# Define un agente que implementa Q-learning con la capacidad de persistir y cargar su tabla Q desde un archivo JSON.
class PersistentQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, save_file='q_table.json'):
        """
        Inicializa un agente con parámetros de aprendizaje, exploración, y soporte para persistencia.
        """
        self.save_file = save_file
        self.q_table = self._load_q_table()
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Tasa de exploración
        self.training_history = []

    # Método para cargar la tabla Q desde un archivo JSON
 def _load_q_table(self):
        """
        Carga la tabla Q desde un archivo si existe; de lo contrario, inicia una nueva tabla.
        """
        if os.path.exists(self.save_file) and os.path.getsize(self.save_file) > 0:
            try:
                with open(self.save_file, 'r') as f:
                    q_dict = json.load(f)
                    return defaultdict(lambda: np.zeros(4), 
                                       {tuple(map(int, k.strip('()').split(','))): np.array(v)
                                        for k, v in q_dict.items()})
            except json.JSONDecodeError:
                print(f"Advertencia: Error al decodificar JSON en {self.save_file}. Iniciando una nueva tabla Q.")
                return defaultdict(lambda: np.zeros(4))
        else:
            print(f"Info: Archivo '{self.save_file}' no encontrado. Iniciando nueva tabla Q.")
            return defaultdict(lambda: np.zeros(4))
