import requests
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# 📡 Configuración de Adafruit IO
AIO_USERNAME = os.getenv("AIO_USERNAME")
AIO_KEY = os.getenv("AIO_KEY")

FEEDS = {
    'temperatura': 'temperatura',
    'humedad_aire': 'humedad-aire',
    'humedad_suelo': 'humedad-suelo',
    'iluminacion': 'iluminacion',
    'nh3': 'nh3',
    'pm25': 'pm25',
    'pm10': 'pm10'
}

HEADERS = {'X-AIO-Key': AIO_KEY, 'User-Agent': 'iot-informe-avicola/1.0'}

def obtener_feed(feed):
    url = f"https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds/{feed}/data?limit=100"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        raise Exception(f"Error {r.status_code} al acceder al feed '{feed}': {r.text}")
    return r.json()

# 📥 Obtener datos
data = {k: obtener_feed(v) for k, v in FEEDS.items()}
n = min(len(v) for v in data.values())

# 📊 Construir DataFrame
df = pd.DataFrame({
    "fecha": pd.to_datetime([x['created_at'] for x in data['temperatura'][:n]]),
    **{k: [float(x['value']) for x in data[k][:n]] for k in FEEDS if k != 'fecha'}
})

# 🔍 Clustering
X = df.drop(columns=["fecha"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
cent = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)


print("👉 Cent shape:", cent.shape)
print(cent.head())


# 🗺 Mapa de calor
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. Aumentar tamaño general de texto y figura
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# 2. Verifica contenido de los centroides
if cent.empty:
    raise ValueError("⚠️ El DataFrame 'cent' está vacío. No se puede generar el mapa de calor.")

# 3. Renombrar filas como Patrón 1, Patrón 2, etc.
cent.index = [f"Patrón {i+1}" for i in range(cent.shape[0])]

# 4. Normalizar por columna para comparar patrones
cent_norm = cent.copy()
for col in cent.columns:
    col_min = cent[col].min()
    col_max = cent[col].max()
    if col_max - col_min == 0:
        cent_norm[col] = 0.5  # valor neutro si no hay variación
    else:
        cent_norm[col] = (cent[col] - col_min) / (col_max - col_min)

# 5. Crear figura y ejes
fig, ax = plt.subplots(figsize=(14, 7))

# 6. Dibujar mapa de calor sin anotaciones automáticas
sns.heatmap(
    cent_norm,
    cmap="coolwarm",
    cbar_kws={"shrink": 0.7},
    annot=False,
    ax=ax
)

# 7. Añadir valores manualmente con control total del texto
for i in range(cent.shape[0]):
    for j in range(cent.shape[1]):
        val = cent.iloc[i, j]
        ax.text(
            j + 0.5, i + 0.5,
            f"{val:.1f}",
            ha='center', va='center',
            fontsize=14,
            color='black',
            fontweight='bold'
        )

# 8. Ajustar etiquetas y título
ax.set_xticklabels(cent.columns, rotation=45, ha='right', fontsize=13)
ax.set_yticklabels(cent.index, rotation=0, fontsize=13)
plt.title("📊 Mapa de calor de condiciones por patrón", fontsize=18)
plt.tight_layout()

# 9. Guardar imagen correctamente
plt.savefig("heatmap.png")
plt.close()





# 🔘 Clústeres en 2D
colores = ["red", "blue", "green"]
plt.figure()
for c in range(3):
    grupo = df[df["cluster"] == c]
    plt.scatter(grupo["temperatura"], grupo["nh3"], color=colores[c], label=f"Patrón {c+1}")
plt.xlabel("Temperatura (°C)")
plt.ylabel("NH₃ (ppm)")
plt.legend()
plt.title("Agrupación de Comportamientos")
plt.tight_layout()
plt.savefig("clusters.png")

# 📈 Tendencias
plt.figure()
df_ordenado = df.sort_values("fecha")
for var in ["temperatura", "humedad_aire", "nh3"]:
    plt.plot(df_ordenado["fecha"], df_ordenado[var], label=var)
plt.xticks(rotation=45)
plt.legend()
plt.title("Tendencias principales")
plt.tight_layout()
plt.savefig("tendencia.png")

# 🧠 Interpretación de clústeres
interpretaciones = []
for idx_num, (idx_name, row) in enumerate(cent.iterrows()):
    temp = row["temperatura"]
    hum_aire = row["humedad_aire"]
    hum_suelo = row["humedad_suelo"]
    nh3 = row["nh3"]
    ilum = row["iluminacion"]
    pm25 = row["pm25"]
    pm10 = row["pm10"]
    color = colores[idx_num]

    interp = f"<li><span style='color:{color}'><b>{idx_name}</b>: "

    if temp > 29 and hum_aire > 70 and nh3 > 25:
        interp += "🔴 Riesgo sanitario: alta temperatura, humedad y NH₃.</span></li>"
    elif nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += "🟠 Polvo y NH₃ elevados: alerta respiratoria.</span></li>"
    elif hum_suelo > 50 and nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += "🟤 Cama empapada con gases y polvo: foco de enfermedades.</span></li>"
    elif temp > 30 and ilum > 400:
        interp += "🔶 Estrés lumínico-térmico: calor + luz excesiva.</span></li>"
    elif temp > 29 and hum_aire < 40:
        interp += "🟡 Estrés térmico seco: calor con humedad baja. Riesgo de deshidratación.</span></li>"
    elif ilum < 150:
        interp += "🟣 Oscuridad prolongada: baja iluminación. Posible letargo o baja actividad.</span></li>"
    elif pm25 > 70 or pm10 > 200:
        interp += "⚫ Contaminación crítica: niveles peligrosos de polvo. Ventilación urgente.</span></li>"
    elif 22 <= temp <= 25 and 60 <= hum_aire <= 80 and ilum < 200:
        interp += "⚪ Noche saludable: condiciones adecuadas para el descanso.</span></li>"
    elif 24 <= temp <= 28 and 50 <= hum_aire <= 70 and nh3 < 20 and pm25 < 35 and pm10 < 100 and 200 <= ilum <= 500:
        interp += "🟢 Condiciones ideales de confort ambiental y productivo.</span></li>"
    else:
        interp += "ℹ️ Combinación atípica: requiere seguimiento técnico.</span></li>"

    interpretaciones.append(interp)

# 📝 Generar HTML
html = f'''
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Informe IoT</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 1rem; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2 {{ color: #2c3e50; }}
    ul {{ padding-left: 1.2rem; }}
    li {{ margin-bottom: 0.7rem; }}
  </style>
</head>
<body>
<h1>📊 Informe Automático IoT - Granjas Avícolas</h1>
<p>📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>📌 Agrupación de comportamientos</h2>
<img src="clusters.png" width="600"><br><br>
<ul>{''.join(interpretaciones)}</ul>

<h2>📈 Tendencias principales</h2>
<img src="tendencia.png" width="600"><br><br>

<h2>🗺 Mapa de calor de variables por patrón</h2>
<img src="heatmap.png" width="600">
</body>
</html>
'''

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
