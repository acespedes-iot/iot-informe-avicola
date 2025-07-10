import requests
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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

data = {k: obtener_feed(v) for k, v in FEEDS.items()}
n = min(len(v) for v in data.values())

df = pd.DataFrame({
    "fecha": pd.to_datetime([x['created_at'] for x in data['temperatura'][:n]]),
    **{k: [float(x['value']) for x in data[k][:n]] for k in FEEDS if k != 'fecha'}
})

X = df.drop(columns=["fecha"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
cent = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)

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

plt.figure()
df_ordenado = df.sort_values("fecha")
for var in ["temperatura", "humedad_aire", "nh3"]:
    plt.plot(df_ordenado["fecha"], df_ordenado[var], label=var)
plt.xticks(rotation=45)
plt.legend()
plt.title("Tendencias principales")
plt.tight_layout()
plt.savefig("tendencia.png")

interpretaciones = []
for idx, row in cent.iterrows():
    temp = row["temperatura"]
    nh3 = row["nh3"]
    interp = f"<li><span style='color:{colores[idx]}'><b>Patrón {idx+1}</b>: Temp. ~{temp:.1f}°C, NH₃ ~{nh3:.1f} ppm — "
    if nh3 > 50:
        interp += "⚠️ Nivel crítico de amoníaco."
    elif nh3 > 25:
        interp += "🟠 Nivel elevado de amoníaco."
    else:
        interp += "🟢 Nivel seguro de amoníaco."
    interp += "</span></li>"
    interpretaciones.append(interp)

html = f"""
<html><head><meta charset='utf-8'><title>Informe Avícola IoT</title></head><body>
<h1>📊 Informe Automático IoT - Granjas Avícolas</h1>
<p>📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>📌 Clustering</h2>
<img src='clusters.png' width='600'><br><br>
<ul>{''.join(interpretaciones)}</ul>
<h2>📈 Tendencias</h2>
<img src='tendencia.png' width='600'>
</body></html>
"""

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
