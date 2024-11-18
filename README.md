# Reinforcement-Learning
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import json
import os
import time

class MazeEnvironment:
    def __init__(self, size=100, seed=42):
        self.size = size
        self._maze = None
        self.seed = seed
        self.start_pos = (1, 1)
        self.current_pos = self.start_pos
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # derecha, izquierda, abajo, arriba

   @property
    def maze(self):
        if self._maze is None:
            self._maze = self._create_maze()
        return self._maze

  def _create_maze(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
      maze = np.ones((self.size, self.size))
        for i in range(1, self.size-1, 2):
            for j in range(1, self.size-1, 2):
                maze[i, j] = 0
                if i < self.size-2 and random.random() < 0.7:
                    maze[i+1, j] = 0
                if j < self.size-2 and random.random() < 0.7:
                    maze[i, j+1] = 0
        maze[1, 1] = 0
        maze[self.size-2, self.size-2] = 2
        if not self._verify_path_exists(maze):
            self._create_path_to_goal(maze)
        random.seed()
        np.random.seed()
        return maze
    def _verify_path_exists(self, maze):
        visited = set()
        def dfs(pos):
            if maze[pos] == 2:
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
    def _create_path_to_goal(self, maze):
        current = self.start_pos
        goal = (self.size-2, self.size-2)
        while current != goal:
            x, y = current
            if x < goal[0]:
                maze[x+1, y] = 0
                current = (x+1, y)
            elif y < goal[1]:
                maze[x, y+1] = 0
                current = (x, y+1)
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    def is_valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.size and
                0 <= y < self.size and
                self.maze[x, y] != 1)
    def step(self, action):
        dx, dy = self.actions[action]
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        if self.is_valid_move(new_pos):
            self.current_pos = new_pos
            if self.maze[new_pos] == 2:
                return new_pos, 100, True
            return new_pos, -1, False
        return self.current_pos, -10, False

class PersistentQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, save_file='q_table.json'):
        self.save_file = save_file
        self.q_table = self._load_q_table()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training_history = []
    def _load_q_table(self):
        if os.path.exists(self.save_file) and os.path.getsize(self.save_file) > 0: # Check if file exists and is not empty
            try:
                with open(self.save_file, 'r') as f:
                    q_dict = json.load(f)
                    # Convertir las claves de string a tuplas
                    return defaultdict(lambda: np.zeros(4),
                        {tuple(map(int, k.strip('()').split(','))): np.array(v)
                         for k, v in q_dict.items()})
            except json.JSONDecodeError:
                print(f"Warning: No se pudo decodificar JSON desde {self.save_file}. Comenzando con una nueva Q-table.")
                # Devuelve un nuevo diccionario predeterminado vacío en caso de error
                return defaultdict(lambda: np.zeros(4))
        else:
            print(f"Info: Q-table file '{self.save_file}' No se encontró o está vacío. Comenzando con uno nuevo Q-table.")
            # Devuelve un nuevo diccionario predeterminado vacío si no se encuentra el archivo
            return defaultdict(lambda: np.zeros(4))
    def save_q_table(self):
        # Convertir las tuplas a strings para poder serializar
        q_dict = {str(k): v.tolist() for k, v in self.q_table.items()}
        # Primero convierta el diccionario a una cadena JSON
        json_string = json.dumps(q_dict)
        # Luego escribe la cadena JSON en el archivo
        with open(self.save_file, 'w') as f:
            f.write(json_string)
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][best_next_action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q)

def train_and_visualize(env, agent, episodes=1000, show_visualization=True):
    rewards_history = []
    best_path = None
    best_reward = float('-inf')
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        current_path = [state]
        done = False
        steps = 0
        max_steps = 1000
        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            current_path.append(state)
            steps += 1
        rewards_history.append(total_reward)

        # Guardar el mejor camino encontrado
  if total_reward > best_reward:
            best_reward = total_reward
            best_path = current_path

  if episode % 10 == 0:
            print(f"Episodio {episode}, Recompensa Total: {total_reward}")

  if show_visualization:
        # Visualizar el progreso del entrenamiento
        plt.figure(figsize=(15, 5))

        # Gráfico del laberinto y camino
  plt.subplot(121)
        plt.imshow(env.maze, cmap='binary')
        if best_path:
            path = np.array(best_path)
            plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=2, label='Mejor camino')
        plt.title('Mejor camino encontrado')
        plt.legend()

        # Gráfico de recompensas
  plt.subplot(122)
        plt.plot(rewards_history)
        plt.title('Recompensas por episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa total')

   plt.tight_layout()
        plt.show()

        # Pequeña pausa para ver los resultados
  time.sleep(2)

    # Guardar el progreso
  agent.save_q_table()

  return rewards_history, best_path

# Ejemplo de uso
if __name__ == "__main__":
    # Crear el entorno y el agente
    env = MazeEnvironment(size=100, seed=42)  # Tamaño reducido para visualización más clara
    agent = PersistentQLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        save_file='maze_q_table.json'
    )

    # Entrenar por múltiples iteraciones
   num_iterations = 5  # Número de veces que queremos entrenar
    episodes_per_iteration = 100  # Episodios por cada iteración

   for iteration in range(num_iterations):
        print(f"\nIteración de entrenamiento {iteration + 1}/{num_iterations}")

        # Reducir epsilon gradualmente para explotar más el conocimiento adquirido
   agent.epsilon = max(0.01, 0.1 - (iteration * 0.02))

   rewards, best_path = train_and_visualize(
            env,
            agent,
            episodes=episodes_per_iteration,
            show_visualization=True
        )
