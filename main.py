# main.py

import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import numpy as np

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

df = pd.DataFrame({
    "fecha": pd.to_datetime([x['created_at'] for x in data['temperatura'][:n]]) - timedelta(hours=4),
    **{k: [float(x['value']) for x in data[k][:n]] for k in FEEDS if k != 'fecha'}
})

# 🧹 Suavizado de curvas y eliminación de valores erráticos
for var in FEEDS:
    df = df[np.abs(df[var] - df[var].rolling(window=5, min_periods=1).mean()) < 3 * df[var].std()]

df[FEEDS.keys()] = df[FEEDS.keys()].rolling(window=3, min_periods=1).mean()

# 🔍 Clustering
X = df.drop(columns=["fecha"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
cent = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
cent.index = [f"Patrón {i+1}" for i in range(cent.shape[0])]

# 📊 Resumen de estadísticas
resumen = df.describe().loc[["mean", "min", "max", "std"]].rename(index={
    "mean": "Promedio", "min": "Mínimo", "max": "Máximo", "std": "Desviación"
}).round(2)

tabla_html = resumen.to_html(classes="tabla", border=1, justify="center", col_space=100)

# 🧠 Interpretaciones de patrones
colores = ["red", "blue", "green"]
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
        interp += "🔴 Riesgo sanitario: alta temperatura, humedad y NH₃. "
        interp += "➡️ Mejorar ventilación y revisar manejo de cama.</span></li>"
    elif nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += "🟠 Polvo y NH₃ elevados: alerta respiratoria. "
        interp += "➡️ Aumentar ventilación y evaluar frecuencia de limpieza.</span></li>"
    elif hum_suelo > 50 and nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += "🟤 Cama empapada con gases y polvo: foco de enfermedades. "
        interp += "➡️ Cambiar cama y reducir humedad excesiva en galpón.</span></li>"
    elif temp > 30 and ilum > 400:
        interp += "🔶 Estrés lumínico-térmico: calor + luz excesiva. "
        interp += "➡️ Reducir intensidad lumínica y mejorar ventilación.</span></li>"
    elif temp > 29 and hum_aire < 40:
        interp += "🟡 Estrés térmico seco: calor con humedad baja. "
        interp += "➡️ Añadir nebulización o humidificadores.</span></li>"
    elif ilum < 150:
        interp += "🟣 Oscuridad prolongada: baja iluminación. "
        interp += "➡️ Revisión de temporizador de luz o bombillas LED.</span></li>"
    elif pm25 > 70 or pm10 > 200:
        interp += "⚫ Contaminación crítica: niveles peligrosos de polvo. "
        interp += "➡️ Revisar entradas de aire y uso de filtros si es posible.</span></li>"
    elif 22 <= temp <= 25 and 60 <= hum_aire <= 80 and ilum < 200:
        interp += "⚪ Noche saludable: condiciones adecuadas para el descanso. "
        interp += "➡️ Mantener programación nocturna controlada.</span></li>"
    elif 24 <= temp <= 28 and 50 <= hum_aire <= 70 and nh3 < 20 and pm25 < 35 and pm10 < 100 and 200 <= ilum <= 500:
        interp += "🟢 Condiciones ideales de confort ambiental y productivo. "
        interp += "✅ Mantener monitoreo y ajustes finos.</span></li>"
    else:
        interp += "ℹ️ Combinación atípica: requiere seguimiento técnico. "
        interp += "➡️ Revisión integral del ambiente.</span></li>"

    interpretaciones.append(interp)

# 📊 Correlación entre variables
correlacion = df[FEEDS.keys()].corr().round(2)
cor_html = correlacion.to_html(classes="tabla", border=1, justify="center", col_space=100)

# 📝 HTML final
meses = {
    "01": "enero", "02": "febrero", "03": "marzo", "04": "abril",
    "05": "mayo", "06": "junio", "07": "julio", "08": "agosto",
    "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre"
}
now = datetime.now() - timedelta(hours=4)
fecha_str = f"{now.day} de {meses[now.strftime('%m')]} de {now.year} - {now.strftime('%H:%M')} (GMT-4)"

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
    .tabla td, .tabla th {{ padding: 6px; text-align: center; border: 1px solid #ccc; }}
    .tabla th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
<h1>📊 Informe Automático IoT - Granjas Avícolas</h1>
<p>📅 Fecha: {fecha_str}</p>

<h2>📌 Agrupación de comportamientos</h2>
<img src="clusters.png"><br><br>
<ul>{''.join(interpretaciones)}</ul>

<h2>🗺 Mapa de calor de variables por patrón</h2>
<img src="heatmap.png"><br><br>

<h2>📈 Tendencias recientes - Ambiente</h2>
<img src="tendencia_1.png"><br><br>

<h2>📈 Tendencias recientes - Contaminantes</h2>
<img src="tendencia_2.png"><br><br>

<h2>📋 Resumen numérico de variables</h2>
{tabla_html}<br><br>

<h2>🔗 Correlación entre variables</h2>
{cor_html}
</body>
</html>
'''

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
