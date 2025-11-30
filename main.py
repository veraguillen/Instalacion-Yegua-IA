import cv2
import numpy as np
import pygame
import threading
import time
import sys
import random
import os
from pathlib import Path

# Silenciar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- CONFIGURACIÓN ---
CONFIG_FILE = "camera_config.txt"
CONFIDENCE_THRESHOLD = 0.70  # Confianza para la IA

# ⚠️ NUEVA CONFIGURACIÓN PARA DETECTAR "VACIO"
# Si el movimiento es menor a esto, se considera que no hay nadie.
# Ajusta este valor: 500 es muy sensible, 5000 es poco sensible.
MOTION_THRESHOLD = 1000 
SHOW_DEBUG_CAM = True        

# --- RUTAS ---
PATH_MODEL = "data/modelo_yegua.keras"
PATH_MODEL_SAVEDMODEL = "data/modelo_yegua_savedmodel"
PATH_FONT = "assets/inicio.ttf"
PATH_AUDIO_ACOSO = "assets/acoso.mp3"
PATH_AUDIO_SAFE = "assets/elevador.mp3"

# Frases
FRASES_ACOSO = [
    "Mami rica", "Zorra", "Que buenas tetas", "Te acompaño a casa", 
    "¿Por qué tan seria?", "Guapa", "Sonríe más", "QUE RICA", 
    "¡Qué buen culo!", "¡Mamasita!", "¡Se te nota todo!", 
    "¡Qué sexy!", "Te comería a besos", "No seas histérica"
]

def detect_available_cameras(max_test=10):
    available = []
    print("\n🔍 Detectando cámaras disponibles...")
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    name = f"Cámara {i} ({w}x{h})"
                except:
                    name = f"Cámara {i}"
                available.append((i, name, True))
                print(f"   ✅ {name}")
            else:
                cap.release()
        else:
            cap.release()
    if not available:
        available.append((0, "Cámara 0 (predeterminada)", False))
    return available

def select_camera():
    saved_index = None
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_index = int(f.read().strip())
        except:
            pass
    
    cameras = detect_available_cameras()
    if len(cameras) == 1 and not cameras[0][2]:
        return cameras[0][0]
    
    print("\n" + "=" * 60)
    print("📷 SELECCIÓN DE CÁMARA")
    print("=" * 60)
    for idx, (cam_idx, name, available) in enumerate(cameras):
        marker = "✅" if available else "⚠️"
        saved = " (guardada)" if saved_index == cam_idx else ""
        print(f"   {idx + 1}. {marker} {name}{saved}")
    
    print(f"\n   0. Usar guardada ({saved_index if saved_index is not None else 'ninguna'})")
    print("   Enter. Usar primera disponible")
    
    while True:
        try:
            choice = input("\n👉 Selecciona (número): ").strip()
            if choice == "":
                for cam_idx, name, avail in cameras:
                    if avail: return cam_idx
                return cameras[0][0]
            if choice == "0":
                if saved_index is not None: return saved_index
                else: continue
            
            c_num = int(choice)
            if 1 <= c_num <= len(cameras):
                sel = cameras[c_num - 1]
                try:
                    with open(CONFIG_FILE, 'w') as f: f.write(str(sel[0]))
                except: pass
                return sel[0]
        except:
            pass

class YeguaInstallation:
    def __init__(self, camera_index=0):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.w, self.h = self.screen.get_size()
        pygame.display.set_caption("YEGUA")
        pygame.mouse.set_visible(False)

        self.fade_surface = pygame.Surface((self.w, self.h))
        self.fade_surface.fill((0, 0, 0))
        self.fade_surface.set_alpha(30) 

        self.load_assets()

        # CARGAR IA
        print(f"⏳ Cargando modelo...")
        try:
            savedmodel_path = Path(PATH_MODEL_SAVEDMODEL)
            if savedmodel_path.exists() and (savedmodel_path / 'saved_model.pb').exists():
                loaded = tf.saved_model.load(str(savedmodel_path))
                if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
                    self.predict_fn = loaded.signatures['serving_default']
                    self.use_signature = True
                else:
                    self.model = loaded
                    self.use_signature = False
            else:
                self.model = tf.keras.models.load_model(PATH_MODEL, compile=False)
                self.use_signature = False
            print("✅ Modelo cargado")
        except Exception as e:
            print(f"❌ Error modelo: {e}")
            sys.exit()

        self.running = True
        self.visual_state = "VACIO"
        self.debug_frame = None
        self.timer_red = 0
        self.text_red = ""
        self.frame_count = 0
        
        # VARIABLES DE MOVIMIENTO
        self.prev_gray = None
        self.motion_level = 0

        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        self.thread = threading.Thread(target=self.vision_loop)
        self.thread.daemon = True
        self.thread.start()

    def load_assets(self):
        try:
            self.font_big = pygame.font.Font(PATH_FONT, 80)
            self.font_med = pygame.font.Font(PATH_FONT, 50)
            self.font_small = pygame.font.Font(PATH_FONT, 20)
        except:
            self.font_big = pygame.font.SysFont("Arial", 80)
            self.font_med = pygame.font.SysFont("Arial", 50)
            self.font_small = pygame.font.SysFont("Arial", 20)

        try:
            self.snd_acoso = pygame.mixer.Sound(PATH_AUDIO_ACOSO)
            self.snd_safe = pygame.mixer.Sound(PATH_AUDIO_SAFE)
            self.ch_acoso = pygame.mixer.Channel(0)
            self.ch_safe = pygame.mixer.Channel(1)
        except:
            self.snd_acoso = None

    def detect_motion(self, frame):
        """Calcula cuánto movimiento hay respecto al frame anterior."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0

        # Calcular diferencia absoluta
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Actualizar frame anterior (promedio ponderado para suavizar)
        # Usamos accumulatedWeighted para que el fondo se actualice lentamente
        # pero para movimiento rápido usamos prev directo
        self.prev_gray = gray 
        
        # Cantidad de pixeles blancos (movimiento)
        count = np.sum(thresh)
        return count

    def vision_loop(self):
        while self.running:
            if not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if not ret: 
                time.sleep(0.1)
                continue

            # 1. DETECCIÓN DE MOVIMIENTO
            # Dividimos por 255 para normalizar un poco, pero mantenemos el número grande
            current_motion = self.detect_motion(frame) / 255
            self.motion_level = current_motion # Guardar para debug

            # Debug visual
            if SHOW_DEBUG_CAM:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.debug_frame = pygame.surfarray.make_surface(np.rot90(frame_rgb))

            # 2. PREDICCIÓN IA
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            batch = np.expand_dims(img, axis=0)

            try:
                if self.use_signature:
                    batch_tensor = tf.constant(batch, dtype=tf.float32)
                    result = self.predict_fn(input_image=batch_tensor)
                    output_key = list(result.keys())[0]
                    preds = result[output_key].numpy()[0]
                else:
                    preds = self.model.predict(batch, verbose=0)[0]
                
                # INTERPRETACIÓN
                if np.size(preds) == 1:
                    score = float(preds)
                    if score > 0.5:
                        idx = 1 # Yegua
                        conf = score
                    else:
                        idx = 0 # Nada
                        conf = 1.0 - score
                else:
                    idx = np.argmax(preds)
                    conf = preds[idx]

                # 3. LÓGICA DE ESTADOS MEJORADA
                # Prioridad 1: Yegua (Acoso)
                if idx == 1 and conf > CONFIDENCE_THRESHOLD:
                    nuevo_estado = "ACOSO"
                
                # Prioridad 2: No es Yegua, ¿Pero hay alguien?
                else:
                    # Aquí usamos el movimiento para decidir entre SEGURO y VACIO
                    if self.motion_level > MOTION_THRESHOLD:
                        # La IA dice "Nada" + Hay Movimiento -> Es una persona
                        nuevo_estado = "SEGURO"
                    else:
                        # La IA dice "Nada" + NO hay Movimiento -> Está vacío
                        nuevo_estado = "VACIO"

                self.visual_state = nuevo_estado
                
                sys.stdout.write(f"\r👁️ Mov: {int(self.motion_level)} | IA: {conf:.2f} (idx {idx}) | Estado: {self.visual_state}   ")
                sys.stdout.flush()
            
            except Exception as e:
                print(f"\n❌ Error visión: {e}")
            
            time.sleep(0.05)

    def draw_multiline_text(self, text, font, color, center_x, center_y):
        lines = text.split('\n')
        total_height = len(lines) * font.get_height()
        current_y = center_y - (total_height / 2)
        for line in lines:
            surf = font.render(line, True, color)
            rect = surf.get_rect(center=(center_x, current_y + (font.get_height()/2)))
            self.screen.blit(surf, rect)
            current_y += font.get_height()

    def draw_acoso(self):
        self.screen.blit(self.fade_surface, (0, 0))
        if self.snd_safe and self.ch_safe.get_busy(): self.ch_safe.fadeout(300)
        if self.snd_acoso and not self.ch_acoso.get_busy(): self.ch_acoso.play(self.snd_acoso, loops=-1)

        if self.frame_count % 5 == 0:
            phrase = random.choice(FRASES_ACOSO)
            font = self.font_med if random.random() > 0.5 else self.font_small
            grey = random.randint(100, 255)
            surf = font.render(phrase, True, (grey, grey, grey))
            x = random.randint(0, self.w - surf.get_width())
            y = random.randint(0, self.h - surf.get_height())
            self.screen.blit(surf, (x, y))

        now = pygame.time.get_ticks()
        if now - self.timer_red > 2000:
            self.text_red = random.choice(FRASES_ACOSO)
            self.timer_red = now
        self.draw_multiline_text(self.text_red, self.font_big, (255, 0, 0), self.w//2, self.h//2)

    def draw_seguro(self):
        self.screen.fill((255, 255, 255))
        if self.snd_acoso and self.ch_acoso.get_busy(): self.ch_acoso.fadeout(500)
        if self.snd_safe and not self.ch_safe.get_busy(): self.ch_safe.play(self.snd_safe, loops=-1)
        self.draw_multiline_text("Ahora estás segura", self.font_med, (0, 0, 0), self.w//2, self.h//2)

    def draw_vacio(self):
        self.screen.fill((0, 0, 0))
        if self.ch_acoso.get_busy(): self.ch_acoso.fadeout(1000)
        if self.ch_safe.get_busy(): self.ch_safe.fadeout(1000)
        texto = "¿Sabes qué se siente ser una\nMujer/Yegua en la calle?"
        self.draw_multiline_text(texto, self.font_med, (255, 255, 255), self.w//2, self.h//2)

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False

            if self.visual_state == "ACOSO": self.draw_acoso()
            elif self.visual_state == "SEGURO": self.draw_seguro()
            else: self.draw_vacio()

            # RECUADRO DEBUG CON INFO DE MOVIMIENTO
            if SHOW_DEBUG_CAM and self.debug_frame:
                thumb_w, thumb_h = 213, 160
                thumb = pygame.transform.scale(self.debug_frame, (thumb_w, thumb_h))
                
                # Color del marco
                color = (0,0,255) # VACIO
                if self.visual_state == "ACOSO": color = (255,0,0)
                elif self.visual_state == "SEGURO": color = (0,255,0)
                
                pos_x, pos_y = 10, self.h - thumb_h - 10
                pygame.draw.rect(self.screen, color, (pos_x-2, pos_y-2, thumb_w+4, thumb_h+4), 4)
                self.screen.blit(thumb, (pos_x, pos_y))
                
                # Barra de movimiento para calibrar
                bar_len = min(thumb_w, int((self.motion_level / (MOTION_THRESHOLD * 2)) * thumb_w))
                pygame.draw.rect(self.screen, (100, 100, 100), (pos_x, pos_y - 15, thumb_w, 10))
                # Si supera el threshold es Verde, si no es Gris
                bar_color = (0, 255, 0) if self.motion_level > MOTION_THRESHOLD else (150, 150, 150)
                pygame.draw.rect(self.screen, bar_color, (pos_x, pos_y - 15, bar_len, 10))
                # Línea de umbral
                threshold_x = int(thumb_w * 0.5) # Asumiendo que threshold es la mitad de la barra max
                pygame.draw.line(self.screen, (255, 255, 0), (pos_x + int(thumb_w/2), pos_y-15), (pos_x + int(thumb_w/2), pos_y-5), 2)

            pygame.display.flip()
            clock.tick(30)

        self.running = False 
        if self.cap.isOpened(): self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    cam_idx = select_camera()
    app = YeguaInstallation(cam_idx)
    app.run()