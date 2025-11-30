import os
import requests
from duckduckgo_search import DDGS
import time

# Configuración
BUSQUEDA = "horse mask street"
CANTIDAD = 100
CARPETA = "data/downloads/horse mask street"

# Crear carpeta si no existe
if not os.path.exists(CARPETA):
    os.makedirs(CARPETA)

print(f"🔍 Buscando {CANTIDAD} imágenes de: '{BUSQUEDA}'...")

# 1. Buscar las URLs usando DuckDuckGo
try:
    results = DDGS().images(
        keywords=BUSQUEDA,
        region="wt-wt",
        safesearch="off",
        max_results=CANTIDAD + 20 # Pedimos de sobra por si algunas fallan
    )
except Exception as e:
    print(f"Error en la búsqueda: {e}")
    results = []

# 2. Descargar las imágenes una por una
count = 0
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"Encontrados {len(results)} enlaces. Iniciando descarga...")

for r in results:
    if count >= CANTIDAD:
        break
    
    url = r['image']
    try:
        # Intentamos descargar
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Guardar archivo
            ext = url.split('.')[-1].split('?')[0] # Intentar sacar extensión
            if len(ext) > 4 or len(ext) < 2: ext = "jpg" # Fallback por si la url es rara
            
            filename = f"{CARPETA}/caballo_{count+1}.{ext}"
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ [{count+1}/{CANTIDAD}] Descargado: {filename}")
            count += 1
        else:
            print(f"❌ Error {response.status_code} en una imagen.")
            
    except Exception as e:
        print(f"⚠️ Salto imagen por error de conexión.")

print(f"\n🎉 ¡Listo! Se descargaron {count} imágenes en '{CARPETA}'.")