import gym
import numpy as np
import random
from tqdm import tqdm

# Crear entorno
env = gym.make("MountainCar-v0")

# Función para discretizar estados
def discretizar(estado, bins=20):
    escala = (estado - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
    return tuple(np.clip((escala * bins).astype(np.int32), 0, bins - 1))

# Inicializar la Q-Table con valores aleatorios
q_table = np.random.uniform(low=-1, high=1, size=[20, 20, env.action_space.n])

# Hiperparámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.95  # Factor de descuento
epsilon = 1.0  # Probabilidad de exploración inicial
epsilon_min = 0.01  # Valor mínimo de epsilon
epsilon_decay = 0.995  # Factor de reducción de epsilon por episodio
episodios = 5000  # Número total de episodios

# Almacena las recompensas por episodio
recompensas_totales = []

# Bucle de entrenamiento
for episodio in tqdm(range(episodios)):
    estado, _ = env.reset()
    estado = discretizar(estado)
    done = False
    recompensa_total = 0

    while not done:
        # Política ε-greedy
        if random.uniform(0, 1) < epsilon:
            accion = env.action_space.sample()  # Exploración
        else:
            accion = np.argmax(q_table[estado])  # Explotación

        # Tomar acción en el entorno
        nuevo_estado, recompensa, done, _, _ = env.step(accion)
        nuevo_estado = discretizar(nuevo_estado)

        # Actualizar Q-Table usando la ecuación de Bellman
        q_table[estado][accion] += alpha * (recompensa + gamma * np.max(q_table[nuevo_estado]) - q_table[estado][accion])

        # Actualizar estado
        estado = nuevo_estado
        recompensa_total += recompensa

    # Reducir ε gradualmente hasta su valor mínimo
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Almacenar recompensa del episodio
    recompensas_totales.append(recompensa_total)

    # Mostrar información cada 100 episodios
    if episodio % 100 == 0:
        print(f"Episodio {episodio} - Recompensa Promedio: {np.mean(recompensas_totales[-100:]):.2f}, Epsilon: {epsilon:.3f}")

    # Renderizar el entorno cada 500 episodios
    if episodio % 500 == 0:
        env.render()

env.close()
