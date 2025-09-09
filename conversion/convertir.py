import pandas as pd

# 1. Leer el dataset original
df = pd.read_csv("dataset_1.csv")

# 2. Limpiar nombres de columnas (por si tienen espacios ocultos)
df.columns = df.columns.str.strip()

# 3. Conversiones a binario según tus reglas:

# ➤ Género: Hombre = 1, Mujer = 0
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# ➤ Edad: si es mayor o igual a 30 años = 1, si es menor = 0
df["age"] = df["age"].apply(lambda x: 1 if x >= 30 else 0)

# ➤ Hipertensión: ya está en 0 = no, 1 = sí (no se cambia)
# ➤ Enfermedad cardíaca: ya está en 0 = no, 1 = sí (no se cambia)

# ➤ Estado civil: Si estuvo casado ("Yes") = 1, si nunca lo estuvo ("No") = 0
df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})

# ➤ Tipo de trabajo: si es "Private" = 1, cualquier otro valor = 0
df["work_type"] = df["work_type"].apply(lambda x: 1 if x == "Private" else 0)

# ➤ Tipo de residencia: "Urban" = 1, "Rural" = 0
df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})

# ➤ Nivel de glucosa: si es mayor a 100 = 1, si es menor o igual a 100 = 0
df["avg_glucose_level"] = df["avg_glucose_level"].apply(lambda x: 1 if x > 100 else 0)

# ➤ Índice de masa corporal (BMI): 
#    si está en el rango saludable 18.5 – 24.9 = 1,
#    en cualquier otro rango (o N/A) = 0
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")  # convierte N/A a NaN
df["bmi"] = df["bmi"].apply(lambda x: 1 if 18.5 <= x <= 24.9 else 0 if pd.notna(x) else 0)

# ➤ Estado de fumador: esta columna ya no se usa → eliminar
if "smoking_status" in df.columns:
    df = df.drop(columns=["smoking_status"])

# ➤ Stroke: ya es binaria (0 = no tuvo ACV, 1 = sí tuvo ACV)

# 4. Guardar el nuevo dataset
df.to_csv("dataset_1_convertido.csv", index=False)

print("✅ Conversión completada. Archivo guardado como dataset_1_convertido.csv")
