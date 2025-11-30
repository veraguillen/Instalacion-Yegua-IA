"""Training script for the horse-mask ("yegua") binary classifier.

Este script entrena un modelo compatible con TensorFlow 2.15.x para .exe.
Usa SOLO tf.keras (NO keras independiente).
Optimizado para guardar como SavedModel (mejor formato para .exe).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

# ⚠️ IMPORTANTE: Solo usar tensorflow.keras, NO importar keras independiente
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Silenciar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR_DEFAULT = Path("data/train")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
# ✅ SavedModel como formato principal (mejor para .exe)
MODEL_SAVEDMODEL_PATH = Path("data/modelo_yegua_savedmodel")
# .keras como backup/compatibilidad
MODEL_KERAS_PATH = Path("data/modelo_yegua.keras")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 binary classifier for yegua vs nada.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python train_yegua.py
  python train_yegua.py --epochs 30 --batch-size 16
  python train_yegua.py --data-dir data/train
        """
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR_DEFAULT,
        help="Root directory containing class subfolders (default: data/train)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Maximum number of epochs to train (default: 20)"
    )
    parser.add_argument(
        "--savedmodel-path",
        type=Path,
        default=MODEL_SAVEDMODEL_PATH,
        help="Where to store the SavedModel (default: data/modelo_yegua_savedmodel)",
    )
    parser.add_argument(
        "--keras-path",
        type=Path,
        default=MODEL_KERAS_PATH,
        help="Where to store the .keras backup (default: data/modelo_yegua.keras)",
    )
    return parser.parse_args()


def ensure_dataset(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de datos: {path.as_posix()}")
    required = {"yegua", "nada"}
    existing = {p.name for p in path.iterdir() if p.is_dir()}
    if not required.issubset(existing):
        raise FileNotFoundError(
            f"La carpeta debe contener subdirectorios {required}, pero solo se hallaron: {sorted(existing)}"
        )


def build_generators(data_dir: Path, batch_size: int) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator]:
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model() -> Model:
    """Construye el modelo usando MobileNetV2 como base."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid", name="output_prediction")(x)

    model = Model(inputs, outputs, name="yegua_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_savedmodel_with_signature(model: Model, savedmodel_path: Path) -> None:
    """Guarda el modelo como SavedModel con firma de predicción explícita.
    
    Esto es importante para .exe porque permite usar la firma 'serving_default'
    de manera consistente.
    """
    # Crear una función de predicción con firma explícita
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input_image")])
    def predict_fn(input_image):
        return {"output_prediction": model(input_image)}
    
    # Guardar con la firma
    tf.saved_model.save(
        model,
        str(savedmodel_path),
        signatures={"serving_default": predict_fn}
    )


def main() -> None:
    args = parse_args()
    ensure_dataset(args.data_dir)

    # Verificar versión de Keras (Keras 2.15.0 es necesario y compatible)
    try:
        import keras
        keras_version = getattr(keras, '__version__', 'desconocida')
        if keras_version.startswith('3.'):
            print("⚠️  ADVERTENCIA: Keras 3.x está instalado")
            print("   Esto puede causar problemas. Instala Keras 2.15.0:")
            print("   pip install keras==2.15.0")
            print()
        elif keras_version.startswith('2.15'):
            print("✅ Keras 2.15.0 instalado (compatible con TensorFlow 2.15.1)")
            print()
        else:
            print(f"⚠️  Keras {keras_version} instalado (verificar compatibilidad)")
            print()
    except ImportError:
        print("⚠️  Keras no está instalado. TensorFlow 2.15.1 requiere Keras 2.15.0")
        print("   Instala con: pip install keras==2.15.0")
        print()

    print("=" * 60)
    print("🚀 ENTRENAMIENTO MODELO YEGUA (Optimizado para SavedModel)")
    print("=" * 60)
    print()
    print(f"📦 TensorFlow: {tf.__version__}")
    
    # Obtener versión de Keras de manera segura
    try:
        import keras
        keras_ver = getattr(keras, '__version__', 'N/A')
        print(f"📦 Keras: {keras_ver}")
    except ImportError:
        print(f"📦 Keras: No instalado")
    
    print()
    print(f"[*] Usando datos desde: {args.data_dir.resolve()}")
    
    train_gen, val_gen = build_generators(args.data_dir, args.batch_size)
    
    print(f"   Clases encontradas: {train_gen.class_indices}")
    print(f"   Imágenes entrenamiento: {train_gen.samples}")
    print(f"   Imágenes validación: {val_gen.samples}")
    print()

    model = build_model()
    model.summary()

    # ✅ Asegurar que los directorios existen
    args.savedmodel_path.parent.mkdir(parents=True, exist_ok=True)
    args.keras_path.parent.mkdir(parents=True, exist_ok=True)

    # Callback para guardar el mejor modelo como .keras durante entrenamiento
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(args.keras_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    steps_per_epoch = max(1, train_gen.samples // args.batch_size)
    validation_steps = max(1, val_gen.samples // args.batch_size)

    print("\n🎯 Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print("\n[*] Entrenamiento completo. Evaluando modelo guardado...")
    # ✅ Cargar el mejor modelo sin compilar
    best_model = tf.keras.models.load_model(str(args.keras_path), compile=False)
    
    # =========================================================================
    # CORRECCIÓN: Compilar manualmente el modelo cargado
    # Como se cargó con compile=False, .evaluate() fallaría si no hacemos esto.
    # =========================================================================
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # =========================================================================

    val_loss, val_acc = best_model.evaluate(val_gen, verbose=0)
    final_accuracy = history.history.get("accuracy", [0])[-1]
    final_val_accuracy = history.history.get("val_accuracy", [0])[-1]

    print("\n===== REPORTE FINAL =====")
    print(f"Precisión entrenamiento (última época): {final_accuracy:.4f}")
    print(f"Precisión validación (última época): {final_val_accuracy:.4f}")
    print(f"Mejor modelo guardado en: {args.keras_path}")
    print(f"Evaluación del modelo guardado - val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")
    
    # ✅ GUARDAR COMO SAVEDMODEL CON FIRMA EXPLÍCITA (PRINCIPAL PARA .EXE)
    print("\n💾 Guardando modelo como SavedModel (formato principal para .exe)...")
    try:
        save_savedmodel_with_signature(best_model, args.savedmodel_path)
        print(f"✅ SavedModel guardado en: {args.savedmodel_path}/")
        print("   ✅ Firma 'serving_default' creada correctamente")
        print("   ✅ Este es el formato recomendado para crear .exe")
        
        # Verificar que el SavedModel se puede cargar
        print("\n   🔍 Verificando SavedModel...")
        loaded = tf.saved_model.load(str(args.savedmodel_path))
        if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
            print("   ✅ Firma 'serving_default' verificada")
            # Probar con un dummy
            dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
            result = loaded.signatures['serving_default'](input_image=dummy)
            print(f"   ✅ Predicción de prueba exitosa: shape={result['output_prediction'].shape}")
        else:
            print("   ⚠️  Firma no encontrada, pero el modelo se guardó correctamente")
            
    except Exception as e:
        print(f"❌ Error guardando SavedModel: {e}")
        import traceback
        traceback.print_exc()
        print("\n   Intentando método alternativo...")
        try:
            # Método alternativo sin firma personalizada
            best_model.save(str(args.savedmodel_path), save_format="tf")
            print(f"   ✅ SavedModel guardado (método alternativo) en: {args.savedmodel_path}/")
        except Exception as e2:
            print(f"   ❌ Método alternativo también falló: {e2}")
            print("   El modelo .keras seguirá funcionando")
    
    # ✅ RE-GUARDAR .KERAS CON FORMATO EXPLÍCITO (BACKUP)
    print("\n💾 Re-guardando modelo .keras con formato explícito (backup)...")
    try:
        best_model.save(str(args.keras_path), save_format="keras")
        print(f"✅ Modelo .keras re-guardado en: {args.keras_path}")
    except Exception as e:
        print(f"⚠️  No se pudo re-guardar .keras: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\n📋 Modelos guardados:")
    print(f"   🎯 {args.savedmodel_path}/ (SavedModel - PRINCIPAL para .exe)")
    print(f"   📦 {args.keras_path} (formato .keras - backup)")
    print("\n💡 Para usar en main_yegua_keras.py:")
    print(f"   Usa: PATH_MODEL_SAVEDMODEL = '{args.savedmodel_path}'")
    print("\n📝 El SavedModel tiene la firma 'serving_default' lista para inferencia")


if __name__ == "__main__":
    main()