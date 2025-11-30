"""Utility script to download horse mask reference images using Pexels API.

This script uses Pexels API to search for specific horse mask images.
Results are stored under ``data/downloads/horse_mask``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Any

import requests

# Configuración
API_KEY = "BSDKSKTlWwBz3Yrl2XAIfBdnHkoXtJswdFSzHQDPIWaP43MYAKa1CR6y"  # Usando tu API key
DOWNLOAD_DIR = Path("data/downloads/horse_mask")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def search_pexels_images(query: str, per_page: int = 20, page: int = 1) -> List[Dict[str, Any]]:
    """Search for images on Pexels with pagination."""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_KEY}
    params = {
        "query": query,
        "per_page": min(per_page, 80),  # Máximo permitido por Pexels
        "page": page,
        "orientation": "portrait"  # Mejor para selfies
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("photos", [])
    except Exception as e:
        print(f"Error buscando '{query}': {e}")
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Esperando {retry_after} segundos antes de reintentar...")
            time.sleep(retry_after)
            return search_pexels_images(query, per_page, page)
        return []

def download_image(url: str, destination: Path) -> bool:
    """Download an image from URL to destination."""
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"✗ Error descargando {url}: {e}")
        return False

def download_horse_mask_dataset(limit: int = 100) -> None:
    """Download horse mask images using Pexels API with specific search terms."""
    if API_KEY == "YOUR_PEXELS_API_KEY":
        print("Por favor, actualiza el script con tu API key de Pexels")
        return

    queries = [
        "Horse head mask person portrait",
        "Creepy horse mask selfie portrait"
    ]
    
    downloaded = 0
    per_page = 20  # Número de resultados por página (máx 80)
    max_pages = 10  # Límite de páginas a buscar por término
    
    print(f"\n🔍 Iniciando descarga de {limit} imágenes...\n")
    
    for query in queries:
        if downloaded >= limit:
            break
            
        print(f"\n🔎 Buscando: '{query}'")
        
        for page in range(1, max_pages + 1):
            if downloaded >= limit:
                break
                
            photos = search_pexels_images(query, per_page=per_page, page=page)
            if not photos:
                print("No se encontraron más resultados.")
                break
                
            for photo in photos:
                if downloaded >= limit:
                    break
                    
                if not (image_url := photo.get("src", {}).get("large")):
                    continue

                file_name = f"horse_mask_{downloaded + 1:03d}.jpg"
                destination = DOWNLOAD_DIR / file_name

                if destination.exists():
                    print(f"⏩ {file_name} ya existe, omitiendo.")
                    downloaded += 1
                    continue

                print(f"⬇️  Descargando {file_name}...")
                if download_image(image_url, destination):
                    downloaded += 1
                    print(f"✅ Descargada: {file_name}")
                    time.sleep(0.5)  # Respetar límites de la API
                else:
                    print(f"❌ Error al descargar {file_name}")

    print(f"\n🎉 Descarga completada: {downloaded} archivos en {DOWNLOAD_DIR.resolve()}")

if __name__ == "__main__":
    download_horse_mask_dataset(limit=100)