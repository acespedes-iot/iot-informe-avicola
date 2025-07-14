import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

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

# 📊 Construcción inicial del DataFrame
df = pd.DataFrame({
    "fecha": pd.to_datetime([x['created_at'] for x in data['temperatura'][:n]]) - timedelta(hours=4),
    **{k: [float(x['value']) for x in data[k][:n]] for k in FEEDS if k != 'fecha'}
})

# 🧼 Eliminación de valores atípicos por desviación estándar
for col in FEEDS:
    if col in df:
        media, std = df[col].mean(), df[col].std()
        df[col] = np.where((df[col] < media - 3 * std) | (df[col] > media + 3 * std), np.nan, df[col])
        df[col] = df[col].interpolate(limit_direction="both")

# 📉 Suavizado de curvas
df = df.sort_values("fecha")
df[FEEDS.keys()] = df[FEEDS.keys()].rolling(window=3, min_periods=1).mean()

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

fig, ax = plt.subplots(figsize=(14, 5.5))
sns.heatmap(
    cent_norm,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5,
    linecolor='gray',
    ax=ax,
    cbar_kws={
        "orientation": "horizontal",
        "shrink": 0.6,
        "pad": 0.25
    }
)
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)
for i in range(cent.shape[0]):
    for j in range(cent.shape[1]):
        val = cent.iloc[i, j]
        ax.text(j + 0.5, i + 0.5, f"{val:.1f}", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
ax.set_xticklabels(cent.columns, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(cent.index, rotation=0, fontsize=12)
plt.title("Mapa de calor de condiciones por patrón", fontsize=16)
plt.subplots_adjust(bottom=0.35)
plt.savefig("heatmap.png", bbox_inches='tight')
plt.close()

# 🔘 Clústeres 2D
colores = ["red", "blue", "green"]
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})
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

# 📈 Tendencias (2 gráficos separados)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# 1️⃣ Tendencias - Ambiente
plt.figure(figsize=(6.4, 5.5))
for var in ["temperatura", "humedad_aire", "humedad_suelo"]:
    plt.plot(df["fecha"], df[var], label=var, linewidth=2.5)
plt.ylabel("°C / % humedad")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.title("📈 Tendencias recientes - Ambiente")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.subplots_adjust(bottom=0.35)
plt.savefig("tendencia_1.png")
plt.close()

# 2️⃣ Tendencias - Contaminantes
plt.figure(figsize=(6.4, 5.5))
for var in ["iluminacion", "nh3", "pm25", "pm10"]:
    plt.plot(df["fecha"], df[var], label=var, linewidth=2.5)
plt.ylabel("Lux / ppm")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.title("📈 Tendencias recientes - Contaminantes")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.subplots_adjust(bottom=0.35)
plt.savefig("tendencia_2.png")
plt.close()

# 📋 Resumen de tendencias
resumen = df[FEEDS.keys()].describe().loc[["mean", "min", "max"]].round(1)
resumen_html = resumen.to_html(classes="table", border=0)

# 🔗 Correlaciones
correlacion = df[FEEDS.keys()].corr().round(2)
correlacion_html = correlacion.to_html(classes="table", border=0)

# 🚨 Alertas simples
alertas = []
if df["temperatura"].max() > 32:
    alertas.append("⚠️ Alerta: Temperatura muy elevada.")
if df["nh3"].max() > 40:
    alertas.append("⚠️ Alerta: Niveles críticos de amoníaco.")
if df["pm10"].max() > 250:
    alertas.append("⚠️ Alerta: Alta concentración de partículas PM10.")
alertas_html = "<ul>" + "".join(f"<li>{a}</li>" for a in alertas) + "</ul>" if alertas else "<p>No se detectaron alertas críticas.</p>"

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
    .table {{ border-collapse: collapse; width: 100%; }}
    .table th, .table td {{ border: 1px solid #ccc; padding: 6px; text-align: center; }}
  </style>
</head>
<body>
<h1>📊 Informe Automático IoT - Granjas Avícolas</h1>
<p>📅 Fecha: {fecha_str}</p>

<h2>📌 Agrupación de comportamientos</h2>
<img src="clusters.png"><br><br>

<h2>🗺 Mapa de calor de variables por patrón</h2>
<img src="heatmap.png"><br><br>

<h2>📈 Tendencias recientes - Ambiente</h2>
<img src="tendencia_1.png"><br><br>

<h2>📈 Tendencias recientes - Contaminantes</h2>
<img src="tendencia_2.png"><br><br>

<h2>📋 Resumen estadístico</h2>
{resumen_html}

<h2>🔗 Correlaciones entre variables</h2>
{correlacion_html}

<h2>🚨 Alertas detectadas</h2>
{alertas_html}

</body>
</html>
'''

with open("informe.html", "w") as f:
    f.write(html)

print("✅ Informe generado con análisis avanzado correctamente")
