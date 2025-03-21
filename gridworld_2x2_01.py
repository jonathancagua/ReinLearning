# =====================================================================
# Algoritmo de Evaluación de Política
# =====================================================================
import numpy as np

'''
GridWorld de 2x2:
 ___ ___
|   |   |
| A | B |
|___|___|
|   |   |
| C | D |
|___|___|

=====================================================================
Entrada: π (la política a ser evaluada)
Inicializar un umbral θ > 0 (determina la precisión de la estimación)
Inicializar V(s) ← 0 para todo estado s

Repetir:
    Δ ← 0
    Para cada estado s en S:
        v ← V(s)
        V(s) ← Σ_a π(a | s) Σ_(s',r) p(s', r | s, a) [r + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
Hasta que Δ < θ
=====================================================================
'''

# Definición de los estados
estados = ["A", "B", "C", "D"]

# Factor de descuento
gamma = 0.7

# Inicialización de los valores de los estados
valores_estados = {estado: 0 for estado in estados}

# Umbral de convergencia
threshold = 0.00001

# Dinámica del entorno y política implícita
def calcular_valor_estado(estado, valores):
    if estado == "A":
        return (1/4 * (5 + gamma * valores["B"]) +
                1/4 * (0 + gamma * valores["C"]) +
                1/2 * (0 + gamma * valores["A"]))
    elif estado == "B":
        return (1/4 * (0 + gamma * valores["A"]) +
                1/4 * (0 + gamma * valores["D"]) +
                1/2 * (5 + gamma * valores["B"]))
    elif estado == "C":
        return (1/4 * (0 + gamma * valores["A"]) +
                1/4 * (0 + gamma * valores["D"]) +
                1/2 * (0 + gamma * valores["C"]))
    elif estado == "D":
        return (1/4 * (5 + gamma * valores["B"]) +
                1/4 * (0 + gamma * valores["C"]) +
                1/2 * (0 + gamma * valores["D"]))
    else:
        return 0  # Caso para estado desconocido (no debería ocurrir)

# Inicialización de variables
delta = float('inf')  # Diferencia máxima inicial
iteration = 0  # Contador de iteraciones

# Algoritmo de evaluación de política
while delta > threshold:
    delta = 0  # Restablece Δ para cada iteración
    iteration += 1
    valores_viejos = valores_estados.copy()  # Guarda los valores actuales
    for estado in estados:
        nuevo_valor = calcular_valor_estado(estado, valores_viejos)  # Calcula el nuevo valor
        valores_estados[estado] = nuevo_valor  # Actualiza el valor del estado
        delta = max(delta, abs(nuevo_valor - valores_viejos[estado]))  # Actualiza Δ

    # Imprime los valores de los estados y el delta de la iteración actual
    print(f"Iteración {iteration}: Valores de los estados: {valores_estados}")
    print(f"Delta: {delta:.6f}")

# Impresión de los resultados finales
print("\nValores de los estados después de la convergencia:")
for estado, valor in valores_estados.items():
    print(f"{estado}: {valor:.4f}")
