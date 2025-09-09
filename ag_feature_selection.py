# ===================================================================
# ALGORITMO GENÉTICO PARA SELECCIÓN DE CARACTERÍSTICAS (FEATURE SELECTION)
# ===================================================================
# 
# OBJETIVO: Encontrar el subconjunto óptimo de características que
# maximiza el rendimiento de un modelo de machine learning.
#
# PROBLEMA: De todas las características disponibles en el dataset,
# ¿cuáles son las más importantes para predecir la variable objetivo?
#
# SOLUCIÓN: Usar un Algoritmo Genético para explorar diferentes
# combinaciones de características y encontrar la mejor.
# ===================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===================================================================
# 1. CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ===================================================================
# Estos parámetros controlan el comportamiento del algoritmo genético

RANDOM_STATE = 42           # Semilla para reproducibilidad
POBLACION_INICIAL = 30      # Tamaño de la población (número de individuos)
GENERACIONES = 25           # Número de generaciones (iteraciones)
PROB_CRUCE = 0.8           # Probabilidad de cruce entre padres (80%)
PROB_MUTACION = 0.05       # Probabilidad de mutación por gen (5%)
TORNEO_K = 3               # Tamaño del torneo para selección
KFOLD = 5                  # Número de folds para validación cruzada

# Establecer semilla aleatoria para resultados reproducibles
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("ALGORITMO GENÉTICO PARA SELECCIÓN DE CARACTERÍSTICAS")
print("=" * 60)
print(f"Población: {POBLACION_INICIAL} individuos")
print(f"Generaciones: {GENERACIONES}")
print(f"Probabilidad de cruce: {PROB_CRUCE}")
print(f"Probabilidad de mutación: {PROB_MUTACION}")
print(f"Validación cruzada: {KFOLD}-fold")
print("=" * 60)

# ===================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ===================================================================
# Cargamos el dataset y lo preparamos para el algoritmo

print("\n📊 CARGANDO DATOS...")
df = pd.read_csv("dataset_1_convertido.csv")
df.columns = df.columns.str.strip()  # Limpiar espacios en nombres de columnas

# Separar características (X) y variable objetivo (y)
if "stroke" not in df.columns:
    raise ValueError("No se encontró la columna objetivo 'stroke' en el dataset.")

y = df["stroke"].astype(int)  # Variable objetivo: 0 = no stroke, 1 = stroke

# Excluir columnas que no son características predictoras
cols_excluir = [c for c in ["stroke", "id"] if c in df.columns]
X = df.drop(columns=cols_excluir)  # Características predictoras

feature_names = X.columns.tolist()  # Nombres de las características
n_features = X.shape[1]            # Número total de características

print(f"✅ Dataset cargado: {X.shape[0]} muestras, {n_features} características")
print(f"✅ Variable objetivo: {y.value_counts().to_dict()}")
print(f"✅ Características disponibles: {feature_names}")

# ===================================================================
# 3. FUNCIÓN DE EVALUACIÓN (FITNESS)
# ===================================================================
# Esta función evalúa qué tan bueno es un cromosoma (combinación de características)

def evaluar_cromosoma(mask):
    """
    Evalúa la calidad de un cromosoma usando F1-Score con validación cruzada.
    
    Args:
        mask: Vector binario donde 1 = característica seleccionada, 0 = no seleccionada
    
    Returns:
        float: F1-Score promedio obtenido con validación cruzada
    """
    
    # Si no se selecciona ninguna característica, penalizar severamente
    if mask.sum() == 0:
        return 0.0
    
    # Seleccionar solo las características marcadas con 1
    X_seleccionado = X.loc[:, mask.astype(bool)]
    
    # Configurar validación cruzada estratificada
    # Estratificada = mantiene la proporción de clases en cada fold
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
    
    f1_scores = []  # Lista para almacenar F1-Scores de cada fold
    
    # Evaluar en cada fold de la validación cruzada
    for train_idx, val_idx in skf.split(X_seleccionado, y):
        # Dividir datos en entrenamiento y validación
        X_train, X_val = X_seleccionado.iloc[train_idx], X_seleccionado.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Crear y entrenar modelo
        model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Hacer predicciones y calcular F1-Score
        predictions = model.predict(X_val)
        f1 = f1_score(y_val, predictions, average="macro")  # Macro = promedio no ponderado
        f1_scores.append(f1)
    
    # Retornar el F1-Score promedio de todos los folds
    return float(np.mean(f1_scores))

# ===================================================================
# 4. OPERADORES GENÉTICOS
# ===================================================================
# Estas funciones implementan los operadores básicos del algoritmo genético

def crear_individuo_aleatorio():
    """
    Crea un individuo aleatorio (cromosoma) para la población inicial.
    
    Returns:
        numpy.array: Vector binario representando la selección de características
    """
    # Crear vector aleatorio con probabilidad 50% para cada característica
    individuo = (np.random.rand(n_features) < 0.5).astype(int)
    
    # Asegurar que al menos una característica esté seleccionada
    if individuo.sum() == 0:
        # Si ninguna está seleccionada, seleccionar una aleatoriamente
        idx_aleatorio = np.random.randint(0, n_features)
        individuo[idx_aleatorio] = 1
    
    return individuo

def seleccion_por_torneo(poblacion, fitnesses, k=TORNEO_K):
    """
    Selecciona un individuo usando torneo de tamaño k.
    
    Args:
        poblacion: Lista de individuos (cromosomas)
        fitnesses: Lista de valores de fitness correspondientes
        k: Tamaño del torneo
    
    Returns:
        numpy.array: Individuo seleccionado (copia)
    """
    # Seleccionar k individuos aleatorios de la población
    indices_competidores = np.random.choice(len(poblacion), size=k, replace=False)
    
    # Encontrar el competidor con mejor fitness
    fitnesses_competidores = [fitnesses[i] for i in indices_competidores]
    ganador_idx = indices_competidores[np.argmax(fitnesses_competidores)]
    
    # Retornar una copia del ganador
    return poblacion[ganador_idx].copy()

def cruce_de_un_punto(padre, madre):
    """
    Realiza cruce de un punto entre dos padres para crear dos hijos.
    
    Args:
        padre: Cromosoma del primer padre
        madre: Cromosoma del segundo padre
    
    Returns:
        tuple: (hijo1, hijo2) - Los dos descendientes
    """
    # Si no se cumple la probabilidad de cruce o hay menos de 2 genes, no cruzar
    if np.random.rand() > PROB_CRUCE or len(padre) < 2:
        return padre.copy(), madre.copy()
    
    # Seleccionar punto de cruce aleatorio (no puede ser 0 ni el final)
    punto_cruce = np.random.randint(1, len(padre))
    
    # Crear hijos intercambiando las partes
    hijo1 = np.concatenate([padre[:punto_cruce], madre[punto_cruce:]])
    hijo2 = np.concatenate([madre[:punto_cruce], padre[punto_cruce:]])
    
    # Asegurar que cada hijo tenga al menos una característica seleccionada
    if hijo1.sum() == 0:
        hijo1[np.random.randint(0, n_features)] = 1
    if hijo2.sum() == 0:
        hijo2[np.random.randint(0, n_features)] = 1
    
    return hijo1, hijo2

def mutacion_bit_flip(individuo):
    """
    Aplica mutación bit-flip a un individuo.
    
    Args:
        individuo: Cromosoma a mutar
    
    Returns:
        numpy.array: Cromosoma mutado
    """
    # Crear máscara de mutación: True donde debe mutar
    mascara_mutacion = np.random.rand(len(individuo)) < PROB_MUTACION
    
    # Crear copia del individuo
    individuo_mutado = individuo.copy()
    
    # Aplicar mutación: 1→0 o 0→1 donde la máscara es True
    individuo_mutado[mascara_mutacion] = 1 - individuo_mutado[mascara_mutacion]
    
    # Asegurar que al menos una característica esté seleccionada
    if individuo_mutado.sum() == 0:
        individuo_mutado[np.random.randint(0, n_features)] = 1
    
    return individuo_mutado

# ===================================================================
# 5. ALGORITMO GENÉTICO PRINCIPAL
# ===================================================================
# Esta función ejecuta el algoritmo genético completo

def ejecutar_algoritmo_genetico():
    """
    Ejecuta el algoritmo genético para selección de características.
    
    Returns:
        tuple: (mejor_cromosoma, mejor_fitness)
    """
    
    print("\n🧬 INICIANDO ALGORITMO GENÉTICO...")
    
    # ===================================================================
    # 5.1 INICIALIZACIÓN DE LA POBLACIÓN
    # ===================================================================
    print("📋 Generando población inicial...")
    
    # Crear población inicial aleatoria
    poblacion = [crear_individuo_aleatorio() for _ in range(POBLACION_INICIAL)]
    
    # Evaluar fitness de cada individuo en la población inicial
    print("⚡ Evaluando fitness de población inicial...")
    fitnesses = [evaluar_cromosoma(ind) for ind in poblacion]
    
    # Encontrar el mejor individuo inicial
    mejor_individuo = poblacion[int(np.argmax(fitnesses))].copy()
    mejor_fitness = float(np.max(fitnesses))
    
    print(f"🎯 Generación 0 | Mejor F1-Score: {mejor_fitness:.4f} | Características: {int(mejor_individuo.sum())}")
    
    # ===================================================================
    # 5.2 EVOLUCIÓN POR GENERACIONES
    # ===================================================================
    for generacion in range(1, GENERACIONES + 1):
        print(f"🔄 Procesando generación {generacion}...")
        
        nueva_poblacion = []
        
        # Crear nueva población manteniendo el tamaño original
        while len(nueva_poblacion) < POBLACION_INICIAL:
            # ===================================================================
            # 5.2.1 SELECCIÓN DE PADRES
            # ===================================================================
            padre1 = seleccion_por_torneo(poblacion, fitnesses)
            padre2 = seleccion_por_torneo(poblacion, fitnesses)
            
            # ===================================================================
            # 5.2.2 CRUCE Y MUTACIÓN
            # ===================================================================
            hijo1, hijo2 = cruce_de_un_punto(padre1, padre2)
            hijo1 = mutacion_bit_flip(hijo1)
            hijo2 = mutacion_bit_flip(hijo2)
            
            # Agregar hijos a la nueva población
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Asegurar que la población tenga exactamente el tamaño deseado
        poblacion = nueva_poblacion[:POBLACION_INICIAL]
        
        # ===================================================================
        # 5.2.3 EVALUACIÓN DE LA NUEVA POBLACIÓN
        # ===================================================================
        fitnesses = [evaluar_cromosoma(ind) for ind in poblacion]
        
        # ===================================================================
        # 5.2.4 ACTUALIZACIÓN DEL MEJOR INDIVIDUO
        # ===================================================================
        idx_mejor_actual = int(np.argmax(fitnesses))
        if fitnesses[idx_mejor_actual] > mejor_fitness:
            mejor_fitness = float(fitnesses[idx_mejor_actual])
            mejor_individuo = poblacion[idx_mejor_actual].copy()
        
        print(f"🎯 Generación {generacion} | Mejor F1-Score: {mejor_fitness:.4f} | Características: {int(mejor_individuo.sum())}")
    
    print("✅ Algoritmo genético completado!")
    return mejor_individuo, mejor_fitness

# ===================================================================
# 6. EJECUCIÓN DEL ALGORITMO
# ===================================================================
# Ejecutar el algoritmo genético y obtener resultados

mejor_cromosoma, mejor_score = ejecutar_algoritmo_genetico()

# ===================================================================
# 7. ANÁLISIS DE RESULTADOS
# ===================================================================
# Analizar y mostrar los resultados obtenidos

print("\n" + "=" * 60)
print("📊 RESULTADOS DEL ALGORITMO GENÉTICO")
print("=" * 60)

# Identificar características seleccionadas y no seleccionadas
caracteristicas_seleccionadas = [f for f, m in zip(feature_names, mejor_cromosoma) if m == 1]
caracteristicas_no_seleccionadas = [f for f, m in zip(feature_names, mejor_cromosoma) if m == 0]

print(f"🎯 Mejor F1-Score obtenido: {mejor_score:.4f}")
print(f"📈 Número de características seleccionadas: {len(caracteristicas_seleccionadas)}")
print(f"📉 Número de características descartadas: {len(caracteristicas_no_seleccionadas)}")

print(f"\n✅ CARACTERÍSTICAS SELECCIONADAS ({len(caracteristicas_seleccionadas)}):")
for i, caracteristica in enumerate(caracteristicas_seleccionadas, 1):
    print(f"   {i:2d}. {caracteristica}")

print(f"\n❌ CARACTERÍSTICAS DESCARTADAS ({len(caracteristicas_no_seleccionadas)}):")
for i, caracteristica in enumerate(caracteristicas_no_seleccionadas, 1):
    print(f"   {i:2d}. {caracteristica}")

# ===================================================================
# 8. ENTRENAMIENTO DEL MODELO FINAL
# ===================================================================
# Entrenar el modelo final usando solo las características seleccionadas

print(f"\n🤖 ENTRENANDO MODELO FINAL...")

# Seleccionar solo las características óptimas
X_optimizado = X[caracteristicas_seleccionadas]

# Crear y entrenar modelo final
modelo_final = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
modelo_final.fit(X_optimizado, y)

# Hacer predicciones y generar reporte
predicciones_finales = modelo_final.predict(X_optimizado)

print("\n" + "=" * 60)
print("📋 REPORTE DE CLASIFICACIÓN FINAL")
print("=" * 60)
print(classification_report(y, predicciones_finales, digits=4))

# ===================================================================
# 9. GUARDAR RESULTADOS
# ===================================================================
# Guardar los resultados en archivos para uso posterior

print(f"\n💾 GUARDANDO RESULTADOS...")

# Guardar lista de características seleccionadas
pd.Series(caracteristicas_seleccionadas, name="caracteristica").to_csv(
    "caracteristicas_seleccionadas_ag.csv", index=False
)

# Guardar dataset reducido con solo las características seleccionadas
dataset_reducido = pd.concat([X_optimizado, y], axis=1)
dataset_reducido.to_csv("dataset_reducido_ag.csv", index=False)

print("✅ Archivos generados:")
print("   📄 caracteristicas_seleccionadas_ag.csv")
print("   📄 dataset_reducido_ag.csv")

# ===================================================================
# 10. RESUMEN FINAL
# ===================================================================
print("\n" + "=" * 60)
print("🎉 RESUMEN FINAL")
print("=" * 60)
print(f"🎯 El algoritmo genético encontró {len(caracteristicas_seleccionadas)} características óptimas")
print(f"📊 F1-Score obtenido: {mejor_score:.4f}")
print(f"📉 Reducción de dimensionalidad: {n_features} → {len(caracteristicas_seleccionadas)} características")
print(f"💡 Porcentaje de características mantenidas: {len(caracteristicas_seleccionadas)/n_features*100:.1f}%")
print("=" * 60)

# ===================================================================
# EXPLICACIÓN DE LA METODOLOGÍA
# ===================================================================
"""
EXPLICACIÓN DETALLADA DE LA METODOLOGÍA:

1. REPRESENTACIÓN DEL PROBLEMA:
   - Cada cromosoma es un vector binario de longitud igual al número de características
   - 1 = característica seleccionada, 0 = característica no seleccionada
   - El objetivo es encontrar la combinación que maximiza el F1-Score

2. FUNCIÓN DE EVALUACIÓN:
   - Usa validación cruzada estratificada para evaluar cada cromosoma
   - Entrena un modelo de Regresión Logística con las características seleccionadas
   - Calcula F1-Score macro (promedio no ponderado de las clases)
   - Retorna el F1-Score promedio de todos los folds

3. OPERADORES GENÉTICOS:
   - SELECCIÓN: Torneo de tamaño k (selecciona el mejor de k individuos aleatorios)
   - CRUCE: Cruce de un punto (intercambia partes de los cromosomas padre y madre)
   - MUTACIÓN: Bit-flip (cambia 1→0 o 0→1 con probabilidad p_mutación)

4. VENTAJAS DE ESTE ENFOQUE:
   - Explora múltiples combinaciones simultáneamente
   - No se queda atrapado en óptimos locales
   - Usa validación cruzada para evaluaciones robustas
   - Mantiene diversidad genética a través de mutación

5. APLICACIONES:
   - Reducción de dimensionalidad en datasets grandes
   - Identificación de variables más predictivas
   - Mejora del rendimiento de modelos de ML
   - Reducción de costos computacionales
"""
