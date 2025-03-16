import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DQLAgent:
    """
    Agente de Deep Q-Learning para el entorno CartPole-v1.
    """
    def __init__(self):
        """
        Inicializa el agente con hiperparámetros y crea el modelo de red neuronal.
        """
        self.epsilon = 1.0  # Factor de exploración inicial
        self.epsilon_decay = 0.9975  # Tasa de reducción de epsilon
        self.epsilon_min = 0.1  # Límite mínimo de epsilon
        self.gamma = 0.9  # Factor de descuento para el futuro
        self.memory = deque(maxlen=2000)  # Memoria de experiencias pasadas
        self.batch_size = 32  # Tamaño del lote para el entrenamiento
        self.trewards = list()  # Recompensas por episodio
        self.max_treward = 0  # Máxima recompensa alcanzada
        self.env = gym.make('CartPole-v1')  # Creación del entorno
        self._create_model()  # Construcción del modelo de red neuronal
    
    def _create_model(self):
        """
        Crea la red neuronal para predecir las acciones.
        """
        self.model = Sequential([
            Dense(24, activation='relu', input_dim=4),
            Dense(24, activation='relu'),
            Dense(2, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    
    def act(self, state):
        """
        Selecciona una acción basada en una política epsilon-greedy.
        
        :param state: Estado actual del entorno
        :return: Acción seleccionada (0 o 1)
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])
    
    def replay(self):
        """
        Entrena la red neuronal utilizando un conjunto de experiencias almacenadas en la memoria.
        """
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=2, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self, episodes):
        """
        Entrena al agente durante un número determinado de episodios.
        
        :param episodes: Número de episodios de entrenamiento
        """
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, 4])
            for f in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                self.memory.append([state, action, next_state, reward, done])
                state = next_state
                if done or trunc:
                    self.trewards.append(f)
                    self.max_treward = max(self.max_treward, f)
                    print(f'episode={e:4d} | treward={f:4d} | max={self.max_treward:4d}', end='\r')
                    break
            if len(self.memory) > self.batch_size:
                self.replay()
        print()
    
    def test(self, episodes):
        """
        Prueba al agente después del entrenamiento sin exploración (epsilon = 0).
        
        :param episodes: Número de episodios de prueba
        """
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, 4])
            for f in range(1, 5001):
                action = np.argmax(self.model.predict(state)[0])
                state, reward, done, trunc, _ = self.env.step(action)
                state = np.reshape(state, [1, 4])
                if done or trunc:
                    print(f, end=' ')
                    break

# Entrenar el agente
agent = DQLAgent()
agent.learn(1500)  # Entrena por 1500 episodios

# Probar el agente entrenado
agent.test(15)  # Prueba el modelo con 15 episodios
