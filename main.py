import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import locale

# 🌐 Configurar localización para fechas en español
# 🔄 Diccionario de traducción de meses
MESES_ES = {
    "January": "enero", "February": "febrero", "March": "marzo",
    "April": "abril", "May": "mayo", "June": "junio",
    "July": "julio", "August": "agosto", "September": "septiembre",
    "October": "octubre", "November": "noviembre", "December": "diciembre"
}

def fecha_formato_es(dt):
    nombre_mes = dt.strftime('%B')
    mes_es = MESES_ES.get(nombre_mes, nombre_mes)
    return dt.strftime(f'%-d de {mes_es} de %Y')


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

# 🗺 Mapa de calor de condiciones por patrón
mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

cent.index = [f"Patrón {i+1}" for i in range(cent.shape[0])]
cent_norm = cent.copy()
for col in cent.columns:
    cmin, cmax = cent[col].min(), cent[col].max()
    cent_norm[col] = 0.5 if cmin == cmax else (cent[col] - cmin) / (cmax - cmin)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    cent_norm,
    cmap="coolwarm",
    cbar_kws={"shrink": 0.7},
    annot=False,
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)

for i in range(cent.shape[0]):
    for j in range(cent.shape[1]):
        val = cent.iloc[i, j]
        ax.text(
            j + 0.5, i + 0.5,
            f"{val:.1f}",
            ha='center', va='center',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

ax.set_xticklabels(cent.columns, rotation=45, ha='right', fontsize=13)
ax.set_yticklabels(cent.index, rotation=0, fontsize=13)
plt.title("📊 Mapa de calor de condiciones por patrón", fontsize=18)
plt.savefig("heatmap.png", bbox_inches='tight')
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

# 📈 Tendencias con doble eje
fig, ax1 = plt.subplots(figsize=(12, 6))
color1 = "tab:red"
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Temperatura / Humedad / NH₃", color=color1)
ax1.plot(df["fecha"], df["temperatura"], label="Temperatura", color="red")
ax1.plot(df["fecha"], df["humedad_aire"], label="Humedad aire", color="orange")
ax1.plot(df["fecha"], df["nh3"], label="NH₃", color="green")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:blue"
ax2.set_ylabel("Iluminación / PM2.5 / PM10", color=color2)
ax2.plot(df["fecha"], df["iluminacion"], label="Iluminación", color="blue")
ax2.plot(df["fecha"], df["pm25"], label="PM2.5", color="purple")
ax2.plot(df["fecha"], df["pm10"], label="PM10", color="black")
ax2.tick_params(axis='y', labelcolor=color2)

fig.autofmt_xdate()
fig.tight_layout()
plt.title("📈 Tendencias recientes")
plt.savefig("tendencia.png")
plt.close()

# 🧠 Interpretación de patrones
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

# 📝 Generar informe HTML
fecha_actual = datetime.now() - timedelta(hours=4)
fecha_es = fecha_formato_es(fecha_actual)


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
<p>📅 Fecha: {fecha_es}</p>

<h2>📌 Agrupación de comportamientos</h2>
<img src="clusters.png"><br><br>
<ul>{''.join(interpretaciones)}</ul>

<h2>🗺 Mapa de calor de variables por patrón</h2>
<img src="heatmap.png"><br><br>

<h2>📈 Tendencias principales</h2>
<img src="tendencia.png">
</body>
</html>
'''

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado correctamente")
