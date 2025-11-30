import os
import shutil
from PIL import Image

def process_images(folder_path, base_name):
    print(f"--- Procesando carpeta: {folder_path} ---")
    
    # 1. Verificar que la carpeta existe
    if not os.path.exists(folder_path):
        print(f"❌ Error: La carpeta no existe: {folder_path}")
        return

    # Extensiones a buscar
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp','.gif')
    
    # Listar archivos
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    print(f"📸 Encontradas {len(files)} imágenes.")

    # Carpeta temporal para no sobrescribir mientras leemos
    temp_folder = os.path.join(folder_path, "temp_processing")
    os.makedirs(temp_folder, exist_ok=True)

    count = 0
    
    # 2. Convertir y guardar en temporal
    for i, filename in enumerate(files, 1):
        try:
            original_path = os.path.join(folder_path, filename)
            
            with Image.open(original_path) as img:
                # Convertir a RGB (quita transparencias)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')
                
                # Guardar como JPG en carpeta temporal
                new_name = f"{base_name}_{i:04d}.jpg" # Ejemplo: yegua_0001.jpg
                temp_path = os.path.join(temp_folder, new_name)
                
                img.save(temp_path, 'JPEG', quality=95)
                count += 1
                
        except Exception as e:
            print(f"⚠️ Error con {filename}: {e}")

    # 3. Reemplazar archivos viejos con los nuevos
    print(f"✅ Conversión terminada. Reemplazando archivos...")
    
    # Borrar todos los archivos originales de la carpeta raíz
    for f in files:
        try:
            os.remove(os.path.join(folder_path, f))
        except:
            pass

    # Mover los nuevos desde temp a la raíz
    for f in os.listdir(temp_folder):
        shutil.move(os.path.join(temp_folder, f), os.path.join(folder_path, f))
    
    # Borrar carpeta temporal
    os.rmdir(temp_folder)
    print(f"🎉 Listo: {count} imágenes estandarizadas en {folder_path}\n")

def main():
    # RUTAS FIJAS (Asegúrate que coinciden con tus carpetas)
    # Usamos r"" para que Windows no se queje de las barras \
    
    path_yegua = r"data\train\yegua"
    path_nada = r"data\train\nada"
    
    process_images(path_yegua, 'yegua')
    process_images(path_nada, 'nada')

if __name__ == "__main__":
    main()