"""Evaluation utilities for the binary horse-mask ("yegua") classifier."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ANALYSIS_DIR = Path("analysis")
MODEL_DEFAULT = Path("data/modelo_yegua.keras")
DATA_DIR_DEFAULT = Path("data/train")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LATENCY_ITERATIONS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación integral del clasificador yegua vs nada",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DEFAULT,
        help="Ruta al modelo .keras entrenado",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR_DEFAULT,
        help="Directorio raíz con subcarpetas por clase",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Tamaño de lote para el generador de validación",
    )
    return parser.parse_args()


def ensure_inputs(model_path: Path, data_dir: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {model_path.as_posix()}"
        )
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de datos: {data_dir.as_posix()}"
        )


def build_validation_generator(data_dir: Path, batch_size: int):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )


def collect_predictions(model: tf.keras.Model, val_gen):
    probs = model.predict(val_gen, verbose=1)
    y_pred = (probs.flatten() >= 0.5).astype(int)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    return y_true, y_pred, np.array(class_labels)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray) -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - Modelo Yegua")
    plt.tight_layout()

    output_path = ANALYSIS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray
) -> Path:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
    print("\n===== Classification Report =====")
    print(report)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ANALYSIS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def measure_latency(model: tf.keras.Model, val_gen) -> float:
    if val_gen.samples == 0:
        raise ValueError("El conjunto de validación está vacío; no se puede medir la latencia.")

    random_index = random.randint(0, val_gen.samples - 1)
    batch_index = random_index // val_gen.batch_size
    offset = random_index % val_gen.batch_size

    images, _ = val_gen[batch_index]
    sample = images[offset : offset + 1]

    start = time.perf_counter()
    for _ in range(LATENCY_ITERATIONS):
        _ = model.predict(sample, verbose=0)
    elapsed = (time.perf_counter() - start) / LATENCY_ITERATIONS * 1000
    return elapsed


def main() -> None:
    args = parse_args()
    ensure_inputs(args.model_path, args.data_dir)

    print("Cargando modelo...")
    model = tf.keras.models.load_model(args.model_path)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Preparando generador de validación...")
    val_gen = build_validation_generator(args.data_dir, args.batch_size)

    print("Generando predicciones...")
    y_true, y_pred, class_labels = collect_predictions(model, val_gen)

    cm_path = save_confusion_matrix(y_true, y_pred, class_labels)
    report_path = save_classification_report(y_true, y_pred, class_labels)

    latency_ms = measure_latency(model, val_gen)

    print(f"📊 Matriz de confusión guardada en: {cm_path}")
    print(f"📝 Reporte de clasificación guardado en: {report_path}")
    print(f"⚡ Latencia promedio por inferencia ({LATENCY_ITERATIONS} iteraciones): {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
