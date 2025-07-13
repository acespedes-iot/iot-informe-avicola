import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

# 📡 Configuración Adafruit IO
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

# 🔍 Clustering
X = df.drop(columns=["fecha"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
cent = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
cent.index = [f"Patrón {i+1}" for i in range(cent.shape[0])]

# 🗺 Mapa de calor
mpl.rcParams.update({'font.size': 12})
cent_norm = cent.copy()
for col in cent.columns:
    cmin, cmax = cent[col].min(), cent[col].max()
    cent_norm[col] = 0.5 if cmin == cmax else (cent[col] - cmin) / (cmax - cmin)

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(
    cent_norm, cmap="coolwarm", cbar_kws={"shrink": 0.6},
    annot=False, linewidths=0.5, linecolor='gray', ax=ax
)
for i in range(cent.shape[0]):
    for j in range(cent.shape[1]):
        ax.text(j + 0.5, i + 0.5, f"{cent.iloc[i, j]:.1f}", ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

ax.set_xticklabels(cent.columns, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(cent.index, rotation=0, fontsize=12)
plt.title("Mapa de calor de condiciones por patrón", fontsize=16)
plt.savefig("heatmap.png", bbox_inches='tight')
plt.close()

# 🔘 Clústeres 2D
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
plt.close()

# 📈 Tendencia 1: Temperatura y Humedades
plt.figure(figsize=(10, 4))
for var in ["temperatura", "humedad_aire", "humedad_suelo"]:
    plt.plot(df["fecha"], df[var], label=var)
plt.xticks(rotation=45)
plt.title("Tendencias: Temperatura y Humedades", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)
plt.tight_layout()
plt.savefig("tendencia_1.png", bbox_inches='tight')
plt.close()

# 📈 Tendencia 2: Iluminación y Calidad de Aire
plt.figure(figsize=(10, 4))
for var in ["iluminacion", "nh3", "pm25", "pm10"]:
    plt.plot(df["fecha"], df[var], label=var)
plt.xticks(rotation=45)
plt.title("Tendencias: Iluminación y Calidad del Aire", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4)
plt.tight_layout()
plt.savefig("tendencia_2.png", bbox_inches='tight')
plt.close()

# 🧠 Interpretación
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

# 📝 HTML
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

<h2>📈 Tendencias: Temperatura y Humedades</h2>
<img src="tendencia_1.png"><br><br>

<h2>📈 Tendencias: Iluminación y Calidad del Aire</h2>
<img src="tendencia_2.png"><br><br>

</body>
</html>
'''

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
