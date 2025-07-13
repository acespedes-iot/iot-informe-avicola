import requests
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
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

# 🗺 Mapa de calor
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

cent.index = [f"Patrón {i+1}" for i in cent.index]
cent_norm = cent.copy()
for col in cent.columns:
    min_val = cent[col].min()
    max_val = cent[col].max()
    cent_norm[col] = 0.5 if max_val - min_val == 0 else (cent[col] - min_val) / (max_val - min_val)
cent_norm.index = cent.index

fig, ax = plt.subplots(figsize=(12, 5))
sns_heatmap = sns.heatmap(
    cent_norm,
    annot=cent,
    fmt=".1f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.7},
    ax=ax
)
for text in sns_heatmap.texts:
    text.set_fontsize(12)
plt.title("📊 Mapa de calor de condiciones por patrón", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
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
html = f"""

<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>Informe IoT</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 1rem; }
    img { max-width: 100%; height: auto; }
    h1, h2 { color: #2c3e50; }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 0.7rem; }
  </style>
</head>
<body>

<h1>📊 Informe Automático IoT - Granjas Avícolas</h1>
<p>📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>📌 Agrupación de comportamientos</h2>
<img src='clusters.png' width='600'><br><br>
<ul>{''.join(interpretaciones)}</ul>

<h2>📈 Tendencias principales</h2>
<img src='tendencia.png' width='600'><br><br>

<h2>🗺 Mapa de calor de variables por patrón</h2>
<img src='heatmap.png' width='600'>

</body></html>
"""

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
