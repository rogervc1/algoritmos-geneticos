# algoritmos-geneticos
# 🧬 Proyecto de Algoritmos Genéticos para Machine Learning

## 📋 Descripción General

Este proyecto implementa **Algoritmos Genéticos** para resolver dos problemas fundamentales en Machine Learning:

1. **Selección de Características (Feature Selection)**: Encontrar el subconjunto óptimo de variables predictoras
2. **Optimización de Hiperparámetros**: Encontrar la mejor configuración de parámetros para modelos de ML

El proyecto utiliza un dataset médico sobre **predicción de accidentes cerebrovasculares (stroke)** para demostrar la efectividad de estos algoritmos evolutivos.

---

## 🎯 Objetivos del Proyecto

### Problema Principal
- **Dataset**: Predicción de accidentes cerebrovasculares
- **Variable Objetivo**: `stroke` (0 = no tuvo ACV, 1 = sí tuvo ACV)
- **Desafío**: Optimizar el rendimiento del modelo de clasificación

### Objetivos Específicos
1. **Reducir dimensionalidad** eliminando características irrelevantes
2. **Mejorar precisión** optimizando hiperparámetros del modelo
3. **Automatizar** el proceso de optimización usando algoritmos evolutivos
4. **Comparar** rendimiento vs configuración por defecto

---

## 📁 Estructura del Proyecto

```
completo_final/
├── 📄 ag_feature_selection.py          # Algoritmo genético para selección de características
├── 📄 ag_hyperparameter_optimization.py # Algoritmo genético para optimización de hiperparámetros
├── 📄 ag_neuroevolution.py             # Algoritmo genético para hallar la mejor arquitectura de una RN
├── 📄 convertir.py                     # Script de conversión de datos
├── 📄 dataset_1_convertido.csv         # Dataset procesado y binarizado
├── 📄 dataset_reducido_ag.csv          # Dataset con características seleccionadas
├── 📄 caracteristicas_seleccionadas_ag.csv # Lista de características óptimas
├── 📄 hiperparametros_optimizados_ag.csv   # Hiperparámetros óptimos encontrados
└── 📁 conversion/
    ├── 📄 dataset_1.csv                # Dataset original
    ├── 📄 dataset_1_convertido.csv     # Dataset convertido
    └── 📄 convertir.py                 # Script de conversión
```

---

## 🔬 Metodología Científica

### 1. Preparación de Datos (`convertir.py`)

**Objetivo**: Convertir variables categóricas a formato binario para algoritmos genéticos.

**Transformaciones aplicadas**:
- **Género**: `Male` → 1, `Female` → 0
- **Edad**: ≥30 años → 1, <30 años → 0
- **Estado civil**: `Yes` (casado) → 1, `No` → 0
- **Tipo de trabajo**: `Private` → 1, otros → 0
- **Residencia**: `Urban` → 1, `Rural` → 0
- **Glucosa**: >100 → 1, ≤100 → 0
- **BMI**: 18.5-24.9 (saludable) → 1, otros → 0
- **Eliminación**: Columna `smoking_status` (no relevante)

**Resultado**: Dataset binario con 10 características + variable objetivo.

### 2. Selección de Características (`ag_feature_selection.py`)

**Problema**: ¿Cuáles de las 10 características son realmente importantes para predecir stroke?

**Solución**: Algoritmo Genético que explora combinaciones de características.

#### 🔧 Configuración del Algoritmo Genético
```python
POBLACION_INICIAL = 30      # 30 individuos por generación
GENERACIONES = 25           # 25 iteraciones evolutivas
PROB_CRUCE = 0.8           # 80% probabilidad de cruce
PROB_MUTACION = 0.05       # 5% probabilidad de mutación
TORNEO_K = 3               # Torneo de 3 individuos
KFOLD = 5                  # Validación cruzada 5-fold
```

#### 🧬 Representación Genética
- **Cromosoma**: Vector binario de longitud 10
- **Gen**: 1 = característica seleccionada, 0 = descartada
- **Ejemplo**: `[1,0,1,0,1,0,0,1,0,1]` → selecciona características 1,3,5,8,10

#### ⚡ Función de Evaluación (Fitness)
```python
def evaluar_cromosoma(mask):
    # 1. Seleccionar características marcadas con 1
    X_seleccionado = X.loc[:, mask.astype(bool)]
    
    # 2. Validación cruzada estratificada (5-fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    # 3. Entrenar modelo en cada fold
    for train_idx, val_idx in skf.split(X_seleccionado, y):
        modelo = LogisticRegression(max_iter=200)
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_val)
        f1 = f1_score(y_val, predicciones, average="macro")
    
    # 4. Retornar F1-Score promedio
    return np.mean(f1_scores)
```

#### 🔄 Operadores Genéticos

**1. Selección por Torneo**:
```python
def seleccion_por_torneo(poblacion, fitnesses, k=3):
    # Seleccionar 3 individuos aleatorios
    competidores = np.random.choice(len(poblacion), size=3)
    # Retornar el de mejor fitness
    return poblacion[mejor_competidor]
```

**2. Cruce de Un Punto**:
```python
def cruce_de_un_punto(padre, madre):
    punto_cruce = np.random.randint(1, len(padre))
    hijo1 = np.concatenate([padre[:punto_cruce], madre[punto_cruce:]])
    hijo2 = np.concatenate([madre[:punto_cruce], padre[punto_cruce:]])
    return hijo1, hijo2
```

**3. Mutación Bit-Flip**:
```python
def mutacion_bit_flip(individuo):
    mascara = np.random.rand(len(individuo)) < PROB_MUTACION
    individuo_mutado[mascara] = 1 - individuo_mutado[mascara]
    return individuo_mutado
```

#### 📊 Resultados Obtenidos
- **Características seleccionadas**: 5 de 10 (50% reducción)
- **F1-Score**: 0.7455
- **Características óptimas**: `age`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`

### 3. Optimización de Hiperparámetros (`ag_hyperparameter_optimization.py`)

**Problema**: ¿Cuáles son los mejores valores para los hiperparámetros del modelo?

**Solución**: Algoritmo Genético que explora el espacio de hiperparámetros.

#### 🔧 Configuración del Algoritmo Genético
```python
POBLACION_INICIAL = 20      # Menor población (más costoso evaluar)
GENERACIONES = 15           # Menos generaciones
PROB_CRUCE = 0.8           # 80% probabilidad de cruce
PROB_MUTACION = 0.1        # 10% probabilidad de mutación (mayor)
```

#### 🧬 Espacio de Hiperparámetros
**LogisticRegression**:
- `C`: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] (regularización)
- `penalty`: ['l1', 'l2'] (tipo de regularización)
- `solver`: ['liblinear', 'saga'] (algoritmo de optimización)
- `max_iter`: [100, 200, 500, 1000] (máximo iteraciones)

#### 🔄 Operadores Genéticos Especializados

**1. Cruce Aritmético**:
```python
def cruce_aritmetico(padre, madre):
    alpha = np.random.rand()  # Peso aleatorio
    hijo1 = alpha * padre + (1 - alpha) * madre
    hijo2 = (1 - alpha) * padre + alpha * madre
    return np.round(hijo1).astype(int), np.round(hijo2).astype(int)
```

**2. Mutación Gaussiana**:
```python
def mutacion_gaussiana(individuo):
    for i in range(len(individuo)):
        if np.random.rand() < PROB_MUTACION:
            mutacion = np.random.normal(0, 0.5)
            individuo[i] = max(0, min(individuo[i] + mutacion, max_val))
    return individuo
```

#### 📊 Resultados Obtenidos
- **Hiperparámetros óptimos**: `C=1.0`, `penalty='l2'`, `solver='saga'`, `max_iter=500`
- **F1-Score**: 0.7455
- **Mejora vs configuración por defecto**: Variable según el dataset

---

## 🧪 Bases Teóricas y Fundamentos

### Algoritmos Genéticos
Los algoritmos genéticos son **algoritmos de optimización** inspirados en la evolución biológica:

1. **Población**: Conjunto de soluciones candidatas
2. **Selección**: Los mejores individuos tienen más probabilidad de reproducirse
3. **Cruce**: Combinación de características de padres para crear hijos
4. **Mutación**: Introducción de variación aleatoria
5. **Evaluación**: Función objetivo que mide la calidad de cada solución

### Ventajas de los Algoritmos Genéticos
- ✅ **Exploración global**: No se quedan atrapados en óptimos locales
- ✅ **Paralelizable**: Pueden evaluar múltiples soluciones simultáneamente
- ✅ **Robustos**: Funcionan bien con funciones no diferenciables
- ✅ **Flexibles**: Se adaptan a diferentes tipos de problemas

### Métricas de Evaluación
- **F1-Score Macro**: Promedio no ponderado de precisión y recall
- **Validación Cruzada**: Evalúa robustez del modelo
- **Estratificación**: Mantiene proporción de clases en cada fold

---

## 📈 Resultados y Análisis

### 1. Selección de Características

**Características seleccionadas** (5 de 10):
1. `age` - Edad (≥30 años)
2. `ever_married` - Estado civil (casado)
3. `work_type` - Tipo de trabajo (privado)
4. `Residence_type` - Tipo de residencia (urbana)
5. `avg_glucose_level` - Nivel de glucosa (>100)

**Características descartadas** (5 de 10):
1. `gender` - Género
2. `hypertension` - Hipertensión
3. `heart_disease` - Enfermedad cardíaca
4. `bmi` - Índice de masa corporal

**Interpretación médica**:
- ✅ **Edad**: Factor de riesgo principal para ACV
- ✅ **Estado civil**: Posible indicador de estrés/estilo de vida
- ✅ **Trabajo privado**: Posible indicador de estrés laboral
- ✅ **Residencia urbana**: Mayor exposición a factores de riesgo
- ✅ **Glucosa alta**: Factor de riesgo cardiovascular

### 2. Optimización de Hiperparámetros

**Configuración óptima**:
- `C=1.0`: Regularización moderada
- `penalty='l2'`: Regularización L2 (Ridge)
- `solver='saga'`: Algoritmo eficiente para datasets pequeños
- `max_iter=500`: Suficientes iteraciones para convergencia

**Interpretación técnica**:
- ✅ **C=1.0**: Balance entre sesgo y varianza
- ✅ **L2 penalty**: Penaliza coeficientes grandes, previene overfitting
- ✅ **SAGA solver**: Eficiente para regularización L1 y L2
- ✅ **500 iteraciones**: Garantiza convergencia sin sobre-entrenamiento

---

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos
```bash
pip install numpy pandas scikit-learn
```

### 1. Preparar Datos
```bash
cd conversion/
python convertir.py
```

### 2. Selección de Características
```bash
python ag_feature_selection.py
```

### 3. Optimización de Hiperparámetros
```bash
python ag_hyperparameter_optimization.py
```

### Archivos Generados
- `caracteristicas_seleccionadas_ag.csv`: Lista de características óptimas
- `dataset_reducido_ag.csv`: Dataset con características seleccionadas
- `hiperparametros_optimizados_ag.csv`: Hiperparámetros óptimos

---

## 🔍 Interpretación de Resultados

### Archivos de Salida

#### `caracteristicas_seleccionadas_ag.csv`
```csv
caracteristica
age
ever_married
work_type
Residence_type
avg_glucose_level
```
**Significado**: Lista de las 5 características más importantes para predecir stroke.

#### `dataset_reducido_ag.csv`
```csv
age,ever_married,work_type,Residence_type,avg_glucose_level,stroke
1,0,0,1,1,1
1,0,0,0,1,1
...
```
**Significado**: Dataset optimizado con solo las características relevantes.

#### `hiperparametros_optimizados_ag.csv`
```csv
C,penalty,solver,max_iter,f1_score,algoritmo
1.0,l2,saga,500,0.7455122655122655,LogisticRegression
```
**Significado**: Configuración óptima del modelo con su rendimiento.

---

## 🎓 Aplicaciones y Casos de Uso

### Selección de Características
- **Medicina**: Identificar biomarcadores más relevantes
- **Finanzas**: Seleccionar indicadores económicos predictivos
- **Marketing**: Encontrar variables demográficas clave
- **IoT**: Reducir sensores necesarios para predicción

### Optimización de Hiperparámetros
- **Deep Learning**: Optimizar arquitecturas de redes neuronales
- **Sistemas de Recomendación**: Ajustar parámetros de algoritmos
- **Procesamiento de Imágenes**: Optimizar filtros y transformaciones
- **NLP**: Ajustar parámetros de modelos de lenguaje

---

## 🔬 Metodología Científica Detallada

### Diseño Experimental
1. **Hipótesis**: Los algoritmos genéticos pueden encontrar configuraciones óptimas
2. **Variables independientes**: Características e hiperparámetros
3. **Variable dependiente**: F1-Score del modelo
4. **Control**: Comparación con configuración por defecto
5. **Reproducibilidad**: Semilla aleatoria fija (RANDOM_STATE=42)

### Validación
- **Validación cruzada**: 5-fold estratificado
- **Métricas robustas**: F1-Score macro
- **Múltiples evaluaciones**: Cada cromosoma evaluado 5 veces
- **Manejo de errores**: Fitness bajo para configuraciones inválidas

### Limitaciones
- **Tamaño de población**: Limitado por recursos computacionales
- **Generaciones**: Número fijo, no adaptativo
- **Espacio de búsqueda**: Discreto para hiperparámetros
- **Dataset pequeño**: 42 muestras, puede limitar generalización

---

## 🏆 Conclusiones

### Logros del Proyecto
1. ✅ **Reducción efectiva**: 50% menos características manteniendo rendimiento
2. ✅ **Optimización automática**: Encontró hiperparámetros óptimos sin intervención manual
3. ✅ **Metodología robusta**: Validación cruzada y métricas apropiadas
4. ✅ **Interpretabilidad**: Resultados médicamente coherentes

### Impacto Científico
- **Eficiencia computacional**: Menos características = modelos más rápidos
- **Interpretabilidad**: Características seleccionadas tienen sentido médico
- **Automatización**: Reduce tiempo de experimentación manual
- **Escalabilidad**: Metodología aplicable a otros problemas

### Futuras Mejoras
- **Población adaptativa**: Ajustar tamaño según convergencia
- **Operadores avanzados**: Cruce uniforme, mutación adaptativa
- **Múltiples objetivos**: Optimizar precisión y velocidad simultáneamente
- **Ensemble methods**: Combinar múltiples algoritmos genéticos

---

## 📚 Referencias y Fundamentos

### Algoritmos Genéticos
- Holland, J.H. (1975). "Adaptation in Natural and Artificial Systems"
- Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"

### Machine Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
- Bishop, C.M. (2006). "Pattern Recognition and Machine Learning"

### Feature Selection
- Guyon, I., & Elisseeff, A. (2003). "An introduction to variable and feature selection"
- Saeys, Y., et al. (2007). "A review of feature selection techniques in bioinformatics"

---

## 👨‍💻 Autor

**Proyecto desarrollado para demostrar la aplicación de Algoritmos Genéticos en problemas de Machine Learning**

*Metodología implementada siguiendo las mejores prácticas de la literatura científica en algoritmos evolutivos y machine learning.*
