import numpy as np
import matplotlib.pyplot as plt

# Definición de los estados y acciones posibles
estados = ["alto", "medio", "bajo"]
acciones = ["reponer", "no reponer"]

# Política π: Probabilidad de tomar cada acción en cada estado
politica = {
    "alto": {"reponer": 0.0, "no reponer": 1.0},
    "medio": {"reponer": 0.4, "no reponer": 0.6},
    "bajo": {"reponer": 0.8, "no reponer": 0.2}
}

# Recompensas r(s, a, s')
recompensas = {
    ("alto", "no reponer", "alto"): 8,
    ("alto", "no reponer", "medio"): 5,
    ("medio", "reponer", "alto"): 5,
    ("medio", "reponer", "medio"): 3,
    ("medio", "no reponer", "bajo"): 0,
    ("medio", "no reponer", "medio"): 3,
    ("bajo", "reponer", "medio"): 5,
    ("bajo", "reponer", "bajo"): 2,
    ("bajo", "no reponer", "bajo"): 2
}

# Dinámica del entorno: p(s' | s, a)
def probabilidad_transicion(estado, accion, nuevo_estado):
    if estado == "alto":
        if accion == "no reponer":
            return {"alto": 0.6, "medio": 0.4}.get(nuevo_estado, 0)
    elif estado == "medio":
        if accion == "reponer":
            return {"alto": 0.7, "medio": 0.3}.get(nuevo_estado, 0)
        elif accion == "no reponer":
            return {"bajo": 0.6, "medio": 0.4}.get(nuevo_estado, 0)
    elif estado == "bajo":
        if accion == "reponer":
            return {"medio": 0.9, "bajo": 0.1}.get(nuevo_estado, 0)
        elif accion == "no reponer":
            return {"bajo": 1.0}.get(nuevo_estado, 0)
    return 0

# Parámetros del algoritmo
gamma = 0.9  # Factor de descuento
threshold = 0.00001  # Umbral de convergencia

# Inicialización de los valores de los estados
valores_estados = {estado: 0 for estado in estados}
historial = {estado: [] for estado in estados}  # Para almacenar la evolución
delta = float('inf')
iteracion = 0

# Algoritmo de Evaluación de Política
while delta > threshold:
    delta = 0
    iteracion += 1
    for estado in estados:
        valor_viejo = valores_estados[estado]
        valor_nuevo = 0

        for accion in acciones:
            v = sum(
                probabilidad_transicion(estado, accion, nuevo_estado) *
                (recompensas.get((estado, accion, nuevo_estado), 0) +
                 gamma * valores_estados[nuevo_estado])
                for nuevo_estado in estados
            )
            valor_nuevo += politica[estado][accion] * v

        valores_estados[estado] = valor_nuevo
        delta = max(delta, abs(valor_nuevo - valor_viejo))
        historial[estado].append(valor_nuevo)

    print(f"Iteración {iteracion}, valores: {valores_estados}")

# Resultados finales
print("\nValores finales de los estados:")
for estado, valor in valores_estados.items():
    print(f"{estado}: {valor:.4f}")

# Visualización de la evolución
plt.figure(figsize=(10, 6))
for estado, valores in historial.items():
    plt.plot(valores, label=f"Estado: {estado}")
plt.xlabel("Iteración")
plt.ylabel("Valor del estado")
plt.title("Evolución del valor de los estados")
plt.legend()
plt.grid(True)
plt.show()
