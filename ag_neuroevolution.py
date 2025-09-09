# ===================================================================
# ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE ARQUITECTURA DE REDES
# ===================================================================
# OBJETIVO: Encontrar la mejor arquitectura de red neuronal para un dataset.
# MÉTODO: Usar neuroevolution con algoritmo genético
# ===================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# ===================================================================
# 1. CONFIGURACIÓN DEL AG
# ===================================================================
RANDOM_STATE = 42
POBLACION_INICIAL = 10
GENERACIONES = 10
PROB_CRUCE = 0.8
PROB_MUTACION = 0.2
TORNEO_K = 3
EPOCHS = 20
BATCH_SIZE = 32

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ===================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ===================================================================
print("📊 Cargando dataset...")
df = pd.read_csv("dataset_1_convertido.csv")
df.columns = df.columns.str.strip()

if "stroke" not in df.columns:
    raise ValueError("No se encontró la columna objetivo 'stroke'")

y = df["stroke"].astype(int)
X = df.drop(columns=["stroke", "id"], errors="ignore")

# Codificar variables categóricas si existen
X = pd.get_dummies(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ===================================================================
# 3. REPRESENTACIÓN DE CROMOSOMAS
# ===================================================================
# Cada cromosoma: [n_capas, n_neuronas1, n_neuronas2, ..., activacion]
# - n_capas: entre 1 y 3
# - n_neuronas por capa: 4 a 128
# - activación: 'relu' o 'tanh'

ACTIVACIONES = ["relu", "tanh"]

def crear_individuo():
    n_capas = np.random.randint(1, 4)
    neuronas = [np.random.randint(4, 129) for _ in range(n_capas)]
    activacion = np.random.choice(ACTIVACIONES)
    return (n_capas, neuronas, activacion)

# ===================================================================
# 4. CREACIÓN DE RED DESDE EL CROMOSOMA
# ===================================================================
def crear_red(individuo):
    n_capas, neuronas, activacion = individuo
    model = Sequential()
    for n in neuronas:
        model.add(Dense(n, activation=activacion))
    model.add(Dense(1, activation="sigmoid"))  # salida binaria
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ===================================================================
# 5. FUNCIÓN DE APTITUD
# ===================================================================
def evaluar(individuo):
    model = crear_red(individuo)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, validation_data=(X_val, y_val))
    preds = (model.predict(X_val) > 0.5).astype(int).ravel()
    return accuracy_score(y_val, preds)

# ===================================================================
# 6. OPERADORES GENÉTICOS
# ===================================================================
def seleccion_por_torneo(poblacion, fitnesses, k=TORNEO_K):
    indices = np.random.choice(len(poblacion), size=k, replace=False)
    mejor = indices[np.argmax([fitnesses[i] for i in indices])]
    return poblacion[mejor]

def cruce(padre, madre):
    if np.random.rand() > PROB_CRUCE:
        return padre, madre

    # Cruzar neuronas y activación
    hijo1 = (padre[0], padre[1][:], padre[2])
    hijo2 = (madre[0], madre[1][:], madre[2])

    # Punto de cruce en número de neuronas
    if padre[0] > 0 and madre[0] > 0:
        punto = np.random.randint(0, min(len(padre[1]), len(madre[1])))
        hijo1[1][:punto] = madre[1][:punto]
        hijo2[1][:punto] = padre[1][:punto]

    return hijo1, hijo2

def mutacion(individuo):
    if np.random.rand() < PROB_MUTACION:
        # Cambiar activación
        nuevo_act = np.random.choice(ACTIVACIONES)
        individuo = (individuo[0], individuo[1], nuevo_act)
    if np.random.rand() < PROB_MUTACION:
        # Cambiar número de neuronas en una capa
        capa = np.random.randint(0, individuo[0])
        individuo[1][capa] = np.random.randint(4, 129)
    return individuo

# ===================================================================
# 7. CICLO PRINCIPAL DEL AG
# ===================================================================
print("\n🧬 Iniciando Neuroevolution...")

poblacion = [crear_individuo() for _ in range(POBLACION_INICIAL)]
fitnesses = [evaluar(ind) for ind in poblacion]

mejor_ind = poblacion[np.argmax(fitnesses)]
mejor_fit = max(fitnesses)

print(f"Generación 0 | Mejor accuracy: {mejor_fit:.4f} | {mejor_ind}")

for gen in range(1, GENERACIONES+1):
    nueva_poblacion = []
    while len(nueva_poblacion) < POBLACION_INICIAL:
        padre = seleccion_por_torneo(poblacion, fitnesses)
        madre = seleccion_por_torneo(poblacion, fitnesses)
        hijo1, hijo2 = cruce(padre, madre)
        hijo1, hijo2 = mutacion(hijo1), mutacion(hijo2)
        nueva_poblacion.extend([hijo1, hijo2])
    poblacion = nueva_poblacion[:POBLACION_INICIAL]
    fitnesses = [evaluar(ind) for ind in poblacion]

    # Actualizar mejor
    idx = np.argmax(fitnesses)
    if fitnesses[idx] > mejor_fit:
        mejor_fit = fitnesses[idx]
        mejor_ind = poblacion[idx]

    print(f"Generación {gen} | Mejor accuracy: {mejor_fit:.4f} | {mejor_ind}")

print("\n✅ Neuroevolution completado!")
print(f"🎯 Mejor arquitectura encontrada: {mejor_ind}")
print(f"📊 Accuracy validación: {mejor_fit:.4f}")
