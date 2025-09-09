# ===================================================================
# ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE HIPERPARÁMETROS
# ===================================================================
# 
# OBJETIVO: Encontrar la combinación óptima de hiperparámetros que
# maximiza el rendimiento de un modelo de machine learning.
#
# PROBLEMA: Los modelos de ML tienen múltiples hiperparámetros que
# afectan su rendimiento. ¿Cómo encontrar los valores óptimos?
#
# SOLUCIÓN: Usar un Algoritmo Genético para explorar el espacio de
# hiperparámetros y encontrar la mejor combinación.
# ===================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===================================================================
# 1. CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ===================================================================
# Parámetros que controlan el comportamiento del algoritmo genético

RANDOM_STATE = 42           # Semilla para reproducibilidad
POBLACION_INICIAL = 20      # Tamaño de la población (menor que feature selection)
GENERACIONES = 15           # Número de generaciones (menor que feature selection)
PROB_CRUCE = 0.8           # Probabilidad de cruce entre padres
PROB_MUTACION = 0.1        # Probabilidad de mutación (mayor que feature selection)
TORNEO_K = 3               # Tamaño del torneo para selección
KFOLD = 5                  # Número de folds para validación cruzada

# Establecer semilla aleatoria
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE HIPERPARÁMETROS")
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
# Cargamos el dataset y lo preparamos para la optimización

print("\n📊 CARGANDO DATOS...")
df = pd.read_csv("dataset_1_convertido.csv")
df.columns = df.columns.str.strip()

# Separar características y variable objetivo
if "stroke" not in df.columns:
    raise ValueError("No se encontró la columna objetivo 'stroke' en el dataset.")

y = df["stroke"].astype(int)

# Excluir columnas no predictoras
cols_excluir = [c for c in ["stroke", "id"] if c in df.columns]
X = df.drop(columns=cols_excluir)

print(f"✅ Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"✅ Variable objetivo: {y.value_counts().to_dict()}")

# ===================================================================
# 3. DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# ===================================================================
# Definimos los hiperparámetros a optimizar y sus rangos de valores

# Configuración de hiperparámetros para diferentes algoritmos
HIPERPARAMETROS = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularización inversa
        'penalty': ['l1', 'l2'],                              # Tipo de regularización
        'solver': ['liblinear', 'saga'],                      # Algoritmo de optimización
        'max_iter': [100, 200, 500, 1000]                    # Máximo de iteraciones
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200, 300, 500],           # Número de árboles
        'max_depth': [3, 5, 10, 15, 20, None],              # Profundidad máxima
        'min_samples_split': [2, 5, 10, 20],                # Mínimo muestras para dividir
        'min_samples_leaf': [1, 2, 4, 8],                   # Mínimo muestras por hoja
        'max_features': ['sqrt', 'log2', 0.5, 0.8]          # Características por división
    },
    'SVC': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],         # Parámetro de regularización
        'kernel': ['linear', 'rbf', 'poly'],                # Tipo de kernel
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], # Coeficiente del kernel
        'degree': [2, 3, 4, 5]                              # Grado del polinomio (solo para poly)
    }
}

# Seleccionar el algoritmo a optimizar (cambiar aquí para probar otros)
ALGORITMO_SELECCIONADO = 'LogisticRegression'  # Cambiar a 'RandomForestClassifier' o 'SVC'

print(f"\n🎯 Algoritmo seleccionado para optimización: {ALGORITMO_SELECCIONADO}")
print(f"📋 Hiperparámetros a optimizar: {list(HIPERPARAMETROS[ALGORITMO_SELECCIONADO].keys())}")

# ===================================================================
# 4. FUNCIONES DE CODIFICACIÓN Y DECODIFICACIÓN
# ===================================================================
# Estas funciones convierten entre hiperparámetros y representación genética

def codificar_hiperparametros(hiperparametros_dict):
    """
    Convierte un diccionario de hiperparámetros a un vector numérico.
    
    Args:
        hiperparametros_dict: Diccionario con hiperparámetros
    
    Returns:
        numpy.array: Vector codificado
    """
    vector = []
    
    for param_name, param_value in hiperparametros_dict.items():
        if param_name in HIPERPARAMETROS[ALGORITMO_SELECCIONADO]:
            # Encontrar el índice del valor en la lista de opciones
            opciones = HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]
            if param_value in opciones:
                idx = opciones.index(param_value)
                vector.append(idx)
            else:
                # Si el valor no está en las opciones, usar valor aleatorio
                vector.append(np.random.randint(0, len(opciones)))
        else:
            # Si el parámetro no está en la configuración, usar 0
            vector.append(0)
    
    return np.array(vector)

def decodificar_hiperparametros(vector_codificado):
    """
    Convierte un vector numérico a un diccionario de hiperparámetros.
    
    Args:
        vector_codificado: Vector codificado
    
    Returns:
        dict: Diccionario con hiperparámetros
    """
    hiperparametros = {}
    param_names = list(HIPERPARAMETROS[ALGORITMO_SELECCIONADO].keys())
    
    for i, param_name in enumerate(param_names):
        if i < len(vector_codificado):
            opciones = HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]
            idx = int(vector_codificado[i]) % len(opciones)  # Asegurar índice válido
            hiperparametros[param_name] = opciones[idx]
        else:
            # Si no hay valor codificado, usar el primero de las opciones
            opciones = HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]
            hiperparametros[param_name] = opciones[0]
    
    return hiperparametros

def crear_modelo_con_hiperparametros(hiperparametros_dict):
    """
    Crea un modelo con los hiperparámetros especificados.
    
    Args:
        hiperparametros_dict: Diccionario con hiperparámetros
    
    Returns:
        Modelo de sklearn configurado
    """
    if ALGORITMO_SELECCIONADO == 'LogisticRegression':
        return LogisticRegression(random_state=RANDOM_STATE, **hiperparametros_dict)
    elif ALGORITMO_SELECCIONADO == 'RandomForestClassifier':
        return RandomForestClassifier(random_state=RANDOM_STATE, **hiperparametros_dict)
    elif ALGORITMO_SELECCIONADO == 'SVC':
        return SVC(random_state=RANDOM_STATE, **hiperparametros_dict)
    else:
        raise ValueError(f"Algoritmo no soportado: {ALGORITMO_SELECCIONADO}")

# ===================================================================
# 5. FUNCIÓN DE EVALUACIÓN (FITNESS)
# ===================================================================
# Evalúa qué tan bueno es un conjunto de hiperparámetros

def evaluar_hiperparametros(vector_codificado):
    """
    Evalúa la calidad de un conjunto de hiperparámetros usando F1-Score con CV.
    
    Args:
        vector_codificado: Vector codificado de hiperparámetros
    
    Returns:
        float: F1-Score promedio obtenido con validación cruzada
    """
    try:
        # Decodificar hiperparámetros
        hiperparametros = decodificar_hiperparametros(vector_codificado)
        
        # Crear modelo con los hiperparámetros
        modelo = crear_modelo_con_hiperparametros(hiperparametros)
        
        # Configurar validación cruzada estratificada
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
        
        f1_scores = []
        
        # Evaluar en cada fold
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar modelo
            modelo.fit(X_train, y_train)
            
            # Hacer predicciones y calcular F1-Score
            predictions = modelo.predict(X_val)
            f1 = f1_score(y_val, predictions, average="macro")
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))
    
    except Exception as e:
        # Si hay error en el entrenamiento, retornar fitness muy bajo
        print(f"⚠️ Error en evaluación: {e}")
        return 0.0

# ===================================================================
# 6. OPERADORES GENÉTICOS
# ===================================================================
# Implementación de los operadores básicos del algoritmo genético

def crear_individuo_aleatorio():
    """
    Crea un individuo aleatorio (conjunto de hiperparámetros).
    
    Returns:
        numpy.array: Vector codificado de hiperparámetros aleatorios
    """
    vector = []
    
    for param_name in HIPERPARAMETROS[ALGORITMO_SELECCIONADO].keys():
        opciones = HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]
        idx_aleatorio = np.random.randint(0, len(opciones))
        vector.append(idx_aleatorio)
    
    return np.array(vector)

def seleccion_por_torneo(poblacion, fitnesses, k=TORNEO_K):
    """
    Selecciona un individuo usando torneo de tamaño k.
    
    Args:
        poblacion: Lista de individuos (vectores de hiperparámetros)
        fitnesses: Lista de valores de fitness correspondientes
        k: Tamaño del torneo
    
    Returns:
        numpy.array: Individuo seleccionado (copia)
    """
    # Seleccionar k individuos aleatorios
    indices_competidores = np.random.choice(len(poblacion), size=k, replace=False)
    
    # Encontrar el competidor con mejor fitness
    fitnesses_competidores = [fitnesses[i] for i in indices_competidores]
    ganador_idx = indices_competidores[np.argmax(fitnesses_competidores)]
    
    return poblacion[ganador_idx].copy()

def cruce_aritmetico(padre, madre):
    """
    Realiza cruce aritmético entre dos padres.
    
    Args:
        padre: Cromosoma del primer padre
        madre: Cromosoma del segundo padre
    
    Returns:
        tuple: (hijo1, hijo2) - Los dos descendientes
    """
    if np.random.rand() > PROB_CRUCE or len(padre) < 2:
        return padre.copy(), madre.copy()
    
    # Cruce aritmético: promedio ponderado
    alpha = np.random.rand()  # Peso aleatorio
    
    hijo1 = alpha * padre + (1 - alpha) * madre
    hijo2 = (1 - alpha) * padre + alpha * madre
    
    # Redondear a enteros (ya que son índices)
    hijo1 = np.round(hijo1).astype(int)
    hijo2 = np.round(hijo2).astype(int)
    
    # Asegurar que los valores estén en rangos válidos
    for i, param_name in enumerate(HIPERPARAMETROS[ALGORITMO_SELECCIONADO].keys()):
        max_val = len(HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]) - 1
        hijo1[i] = max(0, min(hijo1[i], max_val))
        hijo2[i] = max(0, min(hijo2[i], max_val))
    
    return hijo1, hijo2

def mutacion_gaussiana(individuo):
    """
    Aplica mutación gaussiana a un individuo.
    
    Args:
        individuo: Cromosoma a mutar
    
    Returns:
        numpy.array: Cromosoma mutado
    """
    individuo_mutado = individuo.copy()
    
    for i, param_name in enumerate(HIPERPARAMETROS[ALGORITMO_SELECCIONADO].keys()):
        if np.random.rand() < PROB_MUTACION:
            # Mutación gaussiana
            max_val = len(HIPERPARAMETROS[ALGORITMO_SELECCIONADO][param_name]) - 1
            mutacion = np.random.normal(0, 0.5)  # Desviación estándar pequeña
            
            nuevo_valor = individuo_mutado[i] + mutacion
            nuevo_valor = int(np.round(nuevo_valor))
            
            # Asegurar que esté en rango válido
            individuo_mutado[i] = max(0, min(nuevo_valor, max_val))
    
    return individuo_mutado

# ===================================================================
# 7. ALGORITMO GENÉTICO PRINCIPAL
# ===================================================================
# Ejecuta el algoritmo genético para optimización de hiperparámetros

def ejecutar_algoritmo_genetico():
    """
    Ejecuta el algoritmo genético para optimización de hiperparámetros.
    
    Returns:
        tuple: (mejor_hiperparametros, mejor_fitness)
    """
    
    print("\n🧬 INICIANDO ALGORITMO GENÉTICO...")
    
    # ===================================================================
    # 7.1 INICIALIZACIÓN DE LA POBLACIÓN
    # ===================================================================
    print("📋 Generando población inicial...")
    
    # Crear población inicial aleatoria
    poblacion = [crear_individuo_aleatorio() for _ in range(POBLACION_INICIAL)]
    
    # Evaluar fitness de cada individuo
    print("⚡ Evaluando fitness de población inicial...")
    fitnesses = [evaluar_hiperparametros(ind) for ind in poblacion]
    
    # Encontrar el mejor individuo inicial
    mejor_individuo = poblacion[int(np.argmax(fitnesses))].copy()
    mejor_fitness = float(np.max(fitnesses))
    mejor_hiperparametros = decodificar_hiperparametros(mejor_individuo)
    
    print(f"🎯 Generación 0 | Mejor F1-Score: {mejor_fitness:.4f}")
    print(f"   Hiperparámetros: {mejor_hiperparametros}")
    
    # ===================================================================
    # 7.2 EVOLUCIÓN POR GENERACIONES
    # ===================================================================
    for generacion in range(1, GENERACIONES + 1):
        print(f"🔄 Procesando generación {generacion}...")
        
        nueva_poblacion = []
        
        # Crear nueva población
        while len(nueva_poblacion) < POBLACION_INICIAL:
            # Selección de padres
            padre1 = seleccion_por_torneo(poblacion, fitnesses)
            padre2 = seleccion_por_torneo(poblacion, fitnesses)
            
            # Cruce y mutación
            hijo1, hijo2 = cruce_aritmetico(padre1, padre2)
            hijo1 = mutacion_gaussiana(hijo1)
            hijo2 = mutacion_gaussiana(hijo2)
            
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Asegurar tamaño correcto de población
        poblacion = nueva_poblacion[:POBLACION_INICIAL]
        
        # Evaluar nueva población
        fitnesses = [evaluar_hiperparametros(ind) for ind in poblacion]
        
        # Actualizar mejor individuo
        idx_mejor_actual = int(np.argmax(fitnesses))
        if fitnesses[idx_mejor_actual] > mejor_fitness:
            mejor_fitness = float(fitnesses[idx_mejor_actual])
            mejor_individuo = poblacion[idx_mejor_actual].copy()
            mejor_hiperparametros = decodificar_hiperparametros(mejor_individuo)
        
        print(f"🎯 Generación {generacion} | Mejor F1-Score: {mejor_fitness:.4f}")
        print(f"   Hiperparámetros: {mejor_hiperparametros}")
    
    print("✅ Algoritmo genético completado!")
    return mejor_hiperparametros, mejor_fitness

# ===================================================================
# 8. EJECUCIÓN DEL ALGORITMO
# ===================================================================
# Ejecutar el algoritmo genético y obtener resultados

mejor_hiperparametros, mejor_score = ejecutar_algoritmo_genetico()

# ===================================================================
# 9. ANÁLISIS DE RESULTADOS
# ===================================================================
# Analizar y mostrar los resultados obtenidos

print("\n" + "=" * 60)
print("📊 RESULTADOS DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("=" * 60)

print(f"🎯 Mejor F1-Score obtenido: {mejor_score:.4f}")
print(f"🤖 Algoritmo optimizado: {ALGORITMO_SELECCIONADO}")

print(f"\n✅ HIPERPARÁMETROS ÓPTIMOS:")
for param_name, param_value in mejor_hiperparametros.items():
    print(f"   {param_name}: {param_value}")

# ===================================================================
# 10. ENTRENAMIENTO DEL MODELO FINAL
# ===================================================================
# Entrenar el modelo final con los hiperparámetros óptimos

print(f"\n🤖 ENTRENANDO MODELO FINAL CON HIPERPARÁMETROS ÓPTIMOS...")

# Crear modelo con hiperparámetros óptimos
modelo_final = crear_modelo_con_hiperparametros(mejor_hiperparametros)

# Entrenar con todos los datos
modelo_final.fit(X, y)

# Hacer predicciones y generar reporte
predicciones_finales = modelo_final.predict(X)

print("\n" + "=" * 60)
print("📋 REPORTE DE CLASIFICACIÓN FINAL")
print("=" * 60)
print(classification_report(y, predicciones_finales, digits=4))

# ===================================================================
# 11. COMPARACIÓN CON CONFIGURACIÓN POR DEFECTO
# ===================================================================
# Comparar con el modelo usando hiperparámetros por defecto

print(f"\n🔍 COMPARACIÓN CON CONFIGURACIÓN POR DEFECTO...")

# Crear modelo con configuración por defecto
if ALGORITMO_SELECCIONADO == 'LogisticRegression':
    modelo_default = LogisticRegression(random_state=RANDOM_STATE)
elif ALGORITMO_SELECCIONADO == 'RandomForestClassifier':
    modelo_default = RandomForestClassifier(random_state=RANDOM_STATE)
elif ALGORITMO_SELECCIONADO == 'SVC':
    modelo_default = SVC(random_state=RANDOM_STATE)

# Entrenar y evaluar modelo por defecto
modelo_default.fit(X, y)
predicciones_default = modelo_default.predict(X)
f1_default = f1_score(y, predicciones_default, average="macro")

print(f"📊 F1-Score con hiperparámetros por defecto: {f1_default:.4f}")
print(f"📊 F1-Score con hiperparámetros optimizados: {mejor_score:.4f}")
print(f"📈 Mejora obtenida: {((mejor_score - f1_default) / f1_default * 100):.2f}%")

# ===================================================================
# 12. GUARDAR RESULTADOS
# ===================================================================
# Guardar los resultados en archivos

print(f"\n💾 GUARDANDO RESULTADOS...")

# Crear DataFrame con los resultados
resultados_df = pd.DataFrame([mejor_hiperparametros])
resultados_df['f1_score'] = mejor_score
resultados_df['algoritmo'] = ALGORITMO_SELECCIONADO

# Guardar resultados
resultados_df.to_csv("hiperparametros_optimizados_ag.csv", index=False)

print("✅ Archivo generado:")
print("   📄 hiperparametros_optimizados_ag.csv")

# ===================================================================
# 13. RESUMEN FINAL
# ===================================================================
print("\n" + "=" * 60)
print("🎉 RESUMEN FINAL")
print("=" * 60)
print(f"🎯 Algoritmo optimizado: {ALGORITMO_SELECCIONADO}")
print(f"📊 F1-Score obtenido: {mejor_score:.4f}")
print(f"📈 Mejora vs configuración por defecto: {((mejor_score - f1_default) / f1_default * 100):.2f}%")
print(f"🔧 Hiperparámetros optimizados: {len(mejor_hiperparametros)} parámetros")
print("=" * 60)

# ===================================================================
# EXPLICACIÓN DE LA METODOLOGÍA
# ===================================================================
"""
EXPLICACIÓN DETALLADA DE LA METODOLOGÍA:

1. REPRESENTACIÓN DEL PROBLEMA:
   - Cada cromosoma es un vector numérico que representa índices de hiperparámetros
   - Cada posición del vector corresponde a un hiperparámetro específico
   - El valor en cada posición es el índice de la opción seleccionada

2. ESPACIO DE BÚSQUEDA:
   - LogisticRegression: C, penalty, solver, max_iter
   - RandomForestClassifier: n_estimators, max_depth, min_samples_split, etc.
   - SVC: C, kernel, gamma, degree
   - Cada hiperparámetro tiene un conjunto discreto de valores posibles

3. FUNCIÓN DE EVALUACIÓN:
   - Usa validación cruzada estratificada para evaluar cada configuración
   - Entrena el modelo con los hiperparámetros especificados
   - Calcula F1-Score macro promedio de todos los folds
   - Maneja errores de entrenamiento retornando fitness bajo

4. OPERADORES GENÉTICOS:
   - SELECCIÓN: Torneo de tamaño k
   - CRUCE: Cruce aritmético (promedio ponderado de padres)
   - MUTACIÓN: Mutación gaussiana (pequeños cambios aleatorios)

5. VENTAJAS DE ESTE ENFOQUE:
   - Explora múltiples configuraciones simultáneamente
   - No requiere gradientes (útil para algoritmos no diferenciables)
   - Puede manejar espacios de búsqueda discretos y continuos
   - Encuentra configuraciones que funcionan bien en conjunto

6. DIFERENCIAS CON FEATURE SELECTION:
   - Representación: Índices en lugar de binarios
   - Operadores: Cruce aritmético y mutación gaussiana
   - Espacio de búsqueda: Hiperparámetros vs características
   - Objetivo: Optimizar configuración del modelo vs seleccionar variables

7. APLICACIONES:
   - Optimización de cualquier algoritmo de ML
   - Tuning automático de hiperparámetros
   - Mejora del rendimiento de modelos
   - Reducción del tiempo de experimentación manual
"""
