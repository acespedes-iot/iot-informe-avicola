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

# 📊 Mapa de calor de promedios por patrón
import seaborn as sns  # asegúrate de que seaborn esté en requirements.txt

plt.figure(figsize=(10, 4))
sns.heatmap(cent, annot=True, cmap="coolwarm", fmt=".1f")
plt.title("🗺 Mapa de calor de condiciones por patrón")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()


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
    hum_aire = row["humedad_aire"]
    hum_suelo = row["humedad_suelo"]
    nh3 = row["nh3"]
    ilum = row["iluminacion"]
    pm25 = row["pm25"]
    pm10 = row["pm10"]
    color = colores[idx]

    interp = f"<li><span style='color:{color}'><b>Patrón {idx+1}</b>: "

    # Evaluar patrones críticos
    if temp > 29 and hum_aire > 70 and nh3 > 25:
        interp += f"🔴 Riesgo sanitario: alta temperatura, humedad y NH₃.</span></li>"
    elif nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += f"🟠 Polvo y NH₃ elevados: alerta respiratoria.</span></li>"
    elif hum_suelo > 50 and nh3 > 25 and (pm25 > 60 or pm10 > 150):
        interp += f"🟤 Cama empapada con gases y polvo: foco de enfermedades.</span></li>"
    elif temp > 30 and ilum > 400:
        interp += f"🔶 Estrés lumínico-térmico: calor + luz excesiva.</span></li>"
    elif 24 <= temp <= 28 and 50 <= hum_aire <= 70 and nh3 < 20 and pm25 < 35 and pm10 < 100:
        interp += f"🟢 Condiciones ideales de confort ambiental.</span></li>"
    else:
        interp += f"ℹ️ Combinación atípica: requiere seguimiento.</span></li>"

    interpretaciones.append(interp)

html = f"""
<html><head><meta charset='utf-8'><title>Informe IoT</title></head><body>
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
