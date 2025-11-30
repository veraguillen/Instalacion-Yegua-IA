"""Herramienta de diagnóstico previa a la instalación interactiva Yegua.
Adaptada para verificar los modelos generados por train_yegua.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Intentar importar dependencias críticas con mensajes amigables
try:
    import cv2
    import numpy as np
    import pygame
    import tensorflow as tf
except ImportError as e:
    print(f"❌ Error crítico: Falta una dependencia necesaria: {e}")
    print("   Ejecuta: pip install tensorflow opencv-python pygame numpy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
CHECK_OK = "✅"
CHECK_FAIL = "❌"
SAFE_CHECK_OK = "[OK]"
SAFE_CHECK_FAIL = "[FAIL]"

# Rutas esperadas (Coincidentes con train_yegua.py)
PATH_SAVEDMODEL = ROOT / "data" / "modelo_yegua_savedmodel"
PATH_KERAS = ROOT / "data" / "modelo_yegua.keras"


def report(label: str, success: bool, detail: str | None = None) -> None:
    def _choose_symbol() -> str:
        symbol = CHECK_OK if success else CHECK_FAIL
        enc = sys.stdout.encoding or "utf-8"
        try:
            symbol.encode(enc)
            return symbol
        except Exception:
            return SAFE_CHECK_OK if success else SAFE_CHECK_FAIL

    symbol = _choose_symbol()
    if detail:
        print(f"{symbol} {label}: {detail}")
    else:
        print(f"{symbol} {label}")


def check_filesystem() -> bool:
    """Verifica que existan los archivos y carpetas necesarios."""
    
    # 1. Verificar Modelo (Cualquiera de los dos formatos sirve)
    model_exists = False
    model_detail = "No se encontró ningún modelo"
    
    if PATH_SAVEDMODEL.exists() and PATH_SAVEDMODEL.is_dir():
        model_exists = True
        model_detail = f"Encontrado SavedModel (Principal): {PATH_SAVEDMODEL.name}"
    elif PATH_KERAS.exists():
        model_exists = True
        model_detail = f"Encontrado Keras backup: {PATH_KERAS.name}"
        
    report("Sistema de archivos - Modelo", model_exists, model_detail)

    # 2. Verificar Assets
    assets_targets = {
        "Fuente": ROOT / "assets" / "inicio.ttf",
        "Audio Acoso": ROOT / "assets" / "acoso.mp3",
    }
    
    assets_ok = True
    for label, path in assets_targets.items():
        exists = path.exists()
        detail = f"{path.name} {'encontrado' if exists else 'NO existe'}"
        report(f"Sistema de archivos - {label}", exists, detail)
        assets_ok &= exists

    return model_exists and assets_ok


def check_audio_hardware() -> bool:
    try:
        # Inicializar mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        duration = 0.5
        sample_rate = 44100
        
        # Generar tono de prueba (La 440Hz)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t) * np.iinfo(np.int16).max
        waveform = np.column_stack((tone, tone)).astype(np.int16)
        
        sound = pygame.sndarray.make_sound(waveform)
        channel = sound.play()
        
        # Esperar a que termine
        start = time.time()
        while channel.get_busy() and (time.time() - start) < duration + 0.2:
            pygame.time.delay(50)
            
        report("Hardware de audio", True, "Mixer inicializado y test reproducido")
        return True
    except Exception as exc:
        report("Hardware de audio", False, f"Error: {exc}")
        return False
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.quit()


def check_camera(indices=(0, 1)) -> bool:
    any_success = False
    details = []
    
    # Intentar detectar backend preferido en Windows
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY

    for idx in indices:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            details.append(f"Cam {idx}: ❌")
            cap.release()
            continue
        
        # Leer un frame para asegurar que funciona
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            any_success = True
            h, w = frame.shape[:2]
            details.append(f"Cam {idx}: ✅ ({w}x{h})")
        else:
            details.append(f"Cam {idx}: ⚠️ (Abre pero no lee)")
        cap.release()
        
    report("Hardware de cámara", any_success, " | ".join(details) or "Sin cámaras detectadas")
    return any_success


def check_model() -> bool:
    """Intenta cargar el modelo con prioridad en SavedModel."""
    
    # Lista de prioridades según train_yegua.py
    candidates = [
        PATH_SAVEDMODEL,  # 1. SavedModel (Carpeta) - Mejor para .exe
        PATH_KERAS,       # 2. .keras (Archivo) - Backup
    ]
    
    # Encontrar el primer candidato que exista
    model_path = next((p for p in candidates if p.exists()), None)
    
    if not model_path:
        report("Carga de modelo", False, "No se encontraron archivos de modelo en data/")
        return False
    
    print(f"⏳ Cargando modelo desde: {model_path}...")
    
    model = None
    loaded_type = "unknown"
    
    try:
        # Aumentar límite de recursión por seguridad con TF
        sys.setrecursionlimit(5000)
        
        # --- CASO 1: SAVEDMODEL (CARPETA) ---
        if model_path.is_dir():
            print("   📦 Formato detectado: SavedModel (Carpeta)")
            try:
                tf_model = tf.saved_model.load(str(model_path))
                
                # Verificar firma 'serving_default'
                if hasattr(tf_model, 'signatures') and 'serving_default' in tf_model.signatures:
                    inference_func = tf_model.signatures['serving_default']
                    
                    # Crear wrapper compatible
                    def predict_wrapper(input_data):
                        # SavedModel espera tensores, no numpy arrays directos a veces
                        tensor_in = tf.constant(input_data, dtype=tf.float32)
                        out = inference_func(tensor_in)
                        # Extraer el tensor de salida (generalmente 'output_prediction')
                        return list(out.values())[0].numpy()
                    
                    model = predict_wrapper
                    loaded_type = "SavedModel Signature"
                    print("   ✅ Cargado correctamente usando 'serving_default'")
                else:
                    # Fallback si no tiene firmas (raro si se usó el script correcto)
                    model = tf_model
                    loaded_type = "SavedModel Raw"
                    print("   ⚠️  Cargado sin firmas explícitas")
                    
            except Exception as e:
                report("Carga de modelo", False, f"Error cargando SavedModel: {e}")
                return False

        # --- CASO 2: KERAS FILE (.keras) ---
        else:
            print("   📦 Formato detectado: Archivo Keras")
            try:
                # Intentar cargar sin compilar (más rápido y seguro para inferencia)
                model = tf.keras.models.load_model(str(model_path), compile=False)
                loaded_type = "Keras Model"
                print("   ✅ Cargado correctamente (compile=False)")
            except Exception as e:
                print(f"   ⚠️  Falló carga simple: {e}")
                try:
                    # Intento desesperado con compile=True
                    model = tf.keras.models.load_model(str(model_path), compile=True)
                    loaded_type = "Keras Model (Compiled)"
                    print("   ✅ Cargado con compilación")
                except Exception as e2:
                    report("Carga de modelo", False, f"Error fatal cargando .keras: {e2}")
                    return False

        # --- PRUEBA DE INFERENCIA ---
        print("   ⏳ Ejecutando predicción de prueba...")
        
        # Imagen dummy negra (batch_size=1, 224, 224, 3)
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        
        try:
            if loaded_type == "SavedModel Signature":
                prediction = model(dummy_input)
            elif hasattr(model, 'predict'):
                prediction = model.predict(dummy_input, verbose=0)
            else:
                # Objeto raw callable
                prediction = model(dummy_input)
            
            # Verificar forma de la salida
            shape = np.shape(prediction)
            score = float(np.mean(prediction))
            
            detail = f"Tipo: {loaded_type} | Output Shape: {shape} | Score dummy: {score:.4f}"
            report("Carga de modelo", True, detail)
            return True
            
        except Exception as pred_err:
            report("Carga de modelo", False, f"El modelo cargó pero falló al predecir: {pred_err}")
            return False
            
    except Exception as e:
        report("Carga de modelo", False, f"Error inesperado: {e}")
        return False


def main() -> None:
    print("========================================")
    print("   CHECKLIST DE VUELO - PROYECTO YEGUA  ")
    print("========================================")
    
    results = [
        check_filesystem(),
        check_audio_hardware(),
        check_camera(),
        check_model(),
    ]
    
    print("========================================")
    if all(results):
        print(f"{CHECK_OK} TODO LISTO. El sistema es funcional.")
    else:
        print(f"{CHECK_FAIL} Se detectaron errores. Revisa el reporte anterior.")
        sys.exit(1)


if __name__ == "__main__":
    main()