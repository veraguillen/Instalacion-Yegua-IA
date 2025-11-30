# app.py - Experiencia Inmersiva Yegua
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import os
import tempfile
import random
import time
import base64
import logging
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# Configurar el nivel de registro para suprimir mensajes no deseados
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar oneDNN
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Evitar advertencias de paralelismo

# Configurar el registro de Streamlit
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit_webrtc').setLevel(logging.ERROR)

# Configuración de TensorFlow para optimizar el rendimiento
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Suprimir mensajes de TensorFlow
    # Configuración para optimizar el rendimiento de la CPU
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.set_soft_device_placement(True)
except Exception as e:
    pass

# Configuración de WebRTC mejorada
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ],
        "iceTransportPolicy": "relay",  # Mejora la conexión en redes restrictivas
        "bundlePolicy": "max-bundle",  # Reduce la cantidad de conexiones
        "rtcpMuxPolicy": "require",    # Mejora la eficiencia
        "sdpSemantics": "unified-plan"  # Usa el plan unificado de WebRTC
    }
)

# Frases de impacto para la experiencia inmersiva
FRASES_IMPACTO = [
    "¡Mira cómo camina esa yegua!",
    "¿A dónde vas tan solita, yegua?",
    "¡Qué rica estás, yegua!",
    "No te hagas la difícil, yegua",
    "Sonríe, yegua, no seas amargada",
    "¿Por qué te enojas? Es un piropo",
    "Con ese cuerpo, ¿cómo no te van a decir algo?",
    "¿Te ofendiste? Era broma, yegua"
]

# Efectos visuales
class EfectoVisual:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        
    def add_particle(self, x, y):
        self.particles.append({
            'x': x,
            'y': y,
            'size': random.randint(5, 15),
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            'life': 100
        })
    
    def update(self):
        for p in self.particles[:]:
            p['y'] -= random.uniform(1, 3)
            p['x'] += random.uniform(-1, 1)
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def draw(self, frame):
        for p in self.particles:
            cv2.circle(frame, (int(p['x']), int(p['y'])), p['size'], p['color'], -1)

# Cargar el modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_yegua.keras")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.efecto = EfectoVisual(640, 480)
        self.ultima_frase = ""
        self.ultimo_cambio = 0
        self.prob_umbral = 70  # Umbral más alto para activar la transformación
        self._running = True
        
    async def on_ended(self):
        # Cleanup when the connection is closed
        self._running = False
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'efecto'):
            del self.efecto
        
    async def recv(self, frame):
        try:
            print("Recibiendo frame de la cámara...")  # Debug
            img = frame.to_ndarray(format="bgr24")
            
            if img is None or img.size == 0:
                print("Error: Frame vacío o inválido")  # Debug
                return frame
                
            h, w = img.shape[:2]
            print(f"Tamaño del frame: {w}x{h}")  # Debug
            
            # Verificar si la imagen es válida
            if img is None or img.size == 0:
                return frame
                
            # Preprocesar la imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
            img_batch = np.expand_dims(img_resized, axis=0)
            
            # Hacer la predicción
            prediction = self.model.predict(img_batch, verbose=0)
            prob_yegua = prediction[0][0] * 100
            
            # Determinar si se detecta una yegua
            es_yegua = prob_yegua > self.prob_umbral
            
            # Añadir partículas aleatorias
            if es_yegua and random.random() > 0.7:
                self.efecto.add_particle(
                    random.randint(0, w),
                    random.randint(h//2, h)
                )
            
            # Cambiar la frase periódicamente
            tiempo_actual = time.time()
            if tiempo_actual - self.ultimo_cambio > 3:  # Cambiar cada 3 segundos
                self.ultima_frase = random.choice(FRASES_IMPACTO)
                self.ultimo_cambio = tiempo_actual
            
            # Aplicar efecto de distorsión
            if random.random() > 0.3:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Actualizar y dibujar partículas
            self.efecto.update()
            self.efecto.draw(img)
            
            # Limitar la tasa de actualización para mejorar el rendimiento
            current_time = time.time()
            if hasattr(self, 'last_update'):
                time_diff = current_time - self.last_update
                if time_diff < 0.03:  # ~30 FPS
                    return frame
            self.last_update = current_time
            
            # Dibujar la interfaz
            color = (0, 255, 0) if es_yegua else (100, 100, 100)
            
            # Dibujar contorno
            cv2.rectangle(img, (10, 10), (w-10, h-10), color, 2)
            
            # Mostrar el estado de la transformación
            estado = "TRANSFORMACIÓN ACTIVADA" if es_yegua else "BUSCANDO IDENTIDAD"
            cv2.putText(img, estado, (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Mostrar la frase de impacto si está activa
            if es_yegua and self.ultima_frase:
                # Fondo semitransparente para el texto
                overlay = img.copy()
                cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                # Texto centrado
                text_size = cv2.getTextSize(self.ultima_frase, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(img, self.ultima_frase, (text_x, h-20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Asegurarse de que la imagen sea válida antes de devolverla
            if img is not None and img.size > 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            return frame
            
        except Exception as e:
            print(f"Error en el procesamiento del frame: {str(e)}")
            return frame

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="YEGUA: Experiencia Inmersiva",
        page_icon="🐎",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Ocultar el menú de Streamlit y el pie de página
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Estilos CSS personalizados
    st.markdown("""
        <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #8B4513 !important;
            color: white !important;
            border: none;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #A0522D !important;
            transform: scale(1.05);
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .stRadio [role="radiogroup"] {
            flex-direction: row;
            gap: 20px;
        }
        .stRadio [data-baseweb="radio"] {
            margin-right: 5px;
        }
        .stRadio label {
            cursor: pointer;
            padding: 10px 20px;
            border-radius: 20px;
            background: #f0f2f6;
            transition: all 0.3s;
        }
        .stRadio [data-baseweb="radio"]:checked + div {
            background: #8B4513;
            color: white;
        }
        .stMarkdown h1 {
            text-align: center;
            color: #5D4037;
            margin-bottom: 10px;
        }
        .stMarkdown h2 {
            color: #5D4037;
            border-bottom: 2px solid #D2B48C;
            padding-bottom: 5px;
        }
        .stMarkdown h3 {
            color: #5D4037;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Título y descripción
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1>🐎 YEGUA: Experiencia Inmersiva</h1>
            <p style='color: #795548; font-size: 1.1rem;'>
                Una exploración artística de la identidad y el acoso callejero
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Instrucciones
    with st.expander("ℹ️ Cómo funciona esta experiencia", expanded=True):
        st.markdown("""
        Esta instalación te invita a experimentar la transformación en una 'yegua' a través de:
        
        1. **La Máscara**: Al colocarte la máscara de yegua, activas la experiencia inmersiva.
        2. **La Transformación**: El sistema detecta la máscara y comienza la transformación.
        3. **La Experiencia**: Frases de acoso aparecerán, reflejando la violencia de género asociada al término 'yegua'.
        4. **La Reflexión**: Al terminar, te invitamos a compartir tu experiencia.
        
        > *"No es solo una máscara, es una piel que revela realidades ocultas"*
        """)
    
    # Selección de modo
    mode = st.radio(
        "Selecciona el modo de experiencia:",
        ["Cámara en Vivo", "Subir Video", "Subir Imagen"],
        horizontal=True
    )
    
    if mode == "Cámara en Vivo":
        st.markdown("### 🎭 Iniciar Experiencia Inmersiva")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                ### Instrucciones:
                1. Colócate en un espacio bien iluminado
                2. Asegúrate de que tu rostro sea claramente visible
                3. Usa la máscara de yegua para activar la experiencia
                4. Permite el acceso a la cámara cuando se te solicite
                
                **Sugerencia:** Para una mejor experiencia, usa auriculares.
            """)
            
            # Controles adicionales
            st.markdown("### ⚙️ Controles")
            sensibilidad = st.slider("Sensibilidad de detección", 50, 95, 70, 5,
                                  help="Ajusta qué tan sensible es la detección de la máscara")
            
            if st.button("🔴 Iniciar Experiencia", use_container_width=True):
                st.session_state.experiencia_iniciada = True
                st.rerun()
        
        with col2:
            if st.session_state.get('experiencia_iniciada', False):
                st.markdown("### 🎥 Experiencia Activa")
                st.warning("La experiencia puede contener contenido sensible. Por favor, ten en cuenta tu bienestar emocional.")
                
                # Configuración mejorada de la cámara
                print("🔍 Inicializando cámara...")
                
                # Mostrar mensaje de carga
                with st.spinner("Inicializando cámara, por favor espera..."):
                    webrtc_ctx = webrtc_streamer(
                        key="yegua_experience",
                        video_processor_factory=VideoTransformer,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 640},
                                "height": {"ideal": 480},
                                "facingMode": "user"
                            },
                            "audio": False
                        },
                        async_processing=True,
                        video_html_attrs={
                            "style": {
                                "width": "100%",
                                "margin": "0 auto",
                                "border": "2px solid #8B4513",
                                "background": "#000"
                            }
                        }
                    )
                
                # Verificar si la cámara se inicializó correctamente
                if webrtc_ctx is None:
                    st.error("No se pudo acceder a la cámara. Por favor, verifica los permisos e intenta de nuevo.")
                    st.stop()
                
                # Configurar el procesador de video si está disponible
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.prob_umbral = sensibilidad
                
                # Agregar mensaje de estado
                if webrtc_ctx.state.playing:
                    st.success("✅ Cámara activa")
                else:
                    st.warning("⚠️ Esperando conexión de la cámara...")
                
                st.markdown("---")
                if st.button("Finalizar Experiencia", type="primary"):
                    st.session_state.experiencia_iniciada = False
                    st.rerun()
                    
                    # Sección de reflexión post-experiencia
                    st.markdown("### 💭 Reflexión")
                    st.markdown("¿Cómo te sentiste durante la experiencia?")
                    reflexion = st.text_area("Comparte tus pensamientos...")
                    
                    if st.button("Enviar reflexión"):
                        st.success("¡Gracias por compartir tu experiencia!")
                        # Aquí podrías guardar la reflexión en una base de datos
            else:
                st.image("https://via.placeholder.com/600x400/8B4513/FFFFFF?text=Preparando+la+experiencia...", 
                         use_column_width=True, 
                         caption="Preparando la experiencia inmersiva...")
                st.markdown("""
                    ### ¿Listo para la experiencia?
                    Presiona el botón "Iniciar Experiencia" para comenzar.
                """)
        
    elif mode == "Subir Video":
        st.markdown("### 📤 Subir Video")
        video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Guardar el archivo temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            # Mostrar el video con las detecciones
            stframe = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            model = load_model()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Procesar frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
                img_batch = np.expand_dims(img_resized, axis=0)
                
                # Predicción
                prediction = model.predict(img_batch, verbose=0)
                prob_yegua = prediction[0][0] * 100
                
                # Dibujar resultados
                label = "YEGUA" if prob_yegua > 50 else "NO YEGUA"
                color = (0, 255, 0) if prob_yegua > 50 else (0, 0, 255)
                
                cv2.putText(frame, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"{prob_yegua:.1f}%", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Mostrar frame
                stframe.image(frame, channels="BGR", use_column_width=True)
                
            cap.release()
            os.unlink(tfile.name)
            
    else:  # Modo imagen
        st.markdown("### 📷 Subir Imagen")
        img_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
        
        if img_file is not None:
            image = Image.open(img_file).convert('RGB')
            img_array = np.array(image.resize((224, 224))) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            model = load_model()
            prediction = model.predict(img_batch, verbose=0)
            prob_yegua = prediction[0][0] * 100
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagen original", use_column_width=True)
                
            with col2:
                result = "ES UNA YEGUA" if prob_yegua > 50 else "NO ES UNA YEGUA"
                color = "green" if prob_yegua > 50 else "red"
                
                st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; 
                                border: 2px solid {color}; margin: 10px 0;">
                        <h3 style="color: {color};">{result}</h3>
                        <p>Confianza: {prob_yegua if prob_yegua > 50 else 100 - prob_yegua:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #795548; font-size: 0.9rem; margin-top: 50px;'>
            <p>"YEGUA" - Una experiencia de arte inmersivo sobre identidad y género</p>
            <p>Desarrollado con TensorFlow, OpenCV y Streamlit</p>
            <p style='font-size: 0.8rem; margin-top: 10px;'>
                <a href='#' style='color: #8B4513; text-decoration: none;'>Términos de Uso</a> | 
                <a href='#' style='color: #8B4513; text-decoration: none;'>Política de Privacidad</a> | 
                <a href='#' style='color: #8B4513; text-decoration: none;'>Créditos</a>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Script para manejar el estado de la experiencia
    st.markdown("""
    <script>
    // Almacenar el estado de la experiencia
    if (window.location.search.includes('experiencia_iniciada=true')) {
        window.parent.document.querySelector('section[data-testid="stSidebar"]').style.display = 'none';
        window.parent.document.querySelector('section[data-testid="stHeader"]').style.display = 'none';
        window.parent.document.querySelector('section[data-testid="stAppViewContainer"]').style.padding = '0';
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()