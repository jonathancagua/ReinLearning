import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation  
from tqdm import tqdm

# Crear entorno con modo compatible para capturar frames
env = gym.make("MountainCar-v0", render_mode="rgb_array")  

# Definir número de bins como variable
num_bins = 40

# Función para discretizar estados usando la variable
def discretizar(estado, bins=num_bins):
    escala = (estado - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
    return tuple(np.clip((escala * bins).astype(np.int32), 0, bins - 1))

# Inicializar la Q-Table con la variable
q_table = np.random.uniform(low=-1, high=1, size=[num_bins, num_bins, env.action_space.n])

# Hiperparámetros
alpha = 0.02  # Tasa de aprendizaje
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
        if done:
            valor_futuro = 0
        else:
            valor_futuro = np.max(q_table[nuevo_estado])

        q_table[estado][accion] += alpha * (recompensa + gamma * valor_futuro - q_table[estado][accion])

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

# Gráfica de recompensas
plt.plot(np.convolve(recompensas_totales, np.ones(100) / 100, mode='valid'))
plt.title("Recompensa Promedio por Bloque de Episodios")
plt.xlabel("Episodios")
plt.ylabel("Recompensa Promedio")
plt.show()

# Evaluación después del entrenamiento
episodios_prueba = 10
recompensas_prueba = []
frames = []  

for episodio in range(episodios_prueba):
    estado, _ = env.reset()
    estado = discretizar(estado)
    done = False
    recompensa_total = 0

    while not done:
        accion = np.argmax(q_table[estado])
        nuevo_estado, recompensa, done, _, _ = env.step(accion)
        nuevo_estado = discretizar(nuevo_estado)
        estado = nuevo_estado
        recompensa_total += recompensa

        if episodio == episodios_prueba - 1:  
            frame = env.render()
            frames.append(frame)

    recompensas_prueba.append(recompensa_total)
    print(f"Episodio de Prueba {episodio + 1}: Recompensa = {recompensa_total}")

print(f"\nRecompensa Promedio en Evaluación: {np.mean(recompensas_prueba):.2f}")

fig = plt.figure()
img = plt.imshow(frames[0])

def animate(i):
    img.set_data(frames[i])
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=40, blit=True)
ani.save("mountaincar.gif", writer="pillow", fps=25)
plt.close()
