#!/usr/bin/env python3
"""
FastAPI prediction service and CLI utilities for Place de Marché P6.

Usage examples:
  - Start API server:
      python Laureen_Dademeule_3_script_Python_092025.py serve --host 0.0.0.0 --port 8000
  - Predict on a single image via CLI:
      python Laureen_Dademeule_3_script_Python_092025.py predict --image /path/to/image.jpg
  - Generate predictions.csv on a random sample from the dataset:
      python Laureen_Dademeule_3_script_Python_092025.py predict-sample --num 50

Model artifacts created by the classification notebook:
  - data/artifacts/mobilenet_classifier.keras
  - data/artifacts/label_mapping.json
"""

import argparse
import io
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# Lazy imports for FastAPI/uvicorn to allow CLI without these deps
try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

try:
    import uvicorn
    HAVE_UVICORN = True
except Exception:
    HAVE_UVICORN = False

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


PROJECT_ROOT = Path('/Users/laureendademeule/Documents/Projets/P6')
DATA_DIR = PROJECT_ROOT / 'data' / 'Flipkart'
IMAGES_DIR = DATA_DIR / 'Images'
CSV_PATH = DATA_DIR / 'flipkart_com-ecommerce_sample_1050.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'data' / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'mobilenet_classifier.keras'
LABELS_PATH = ARTIFACTS_DIR / 'label_mapping.json'
PREDICTIONS_CSV = PROJECT_ROOT / 'predictions.csv'


def load_model_and_labels() -> Tuple[tf.keras.Model, Dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train and export it first.")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label mapping not found at {LABELS_PATH}.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        label_mapping = json.load(f)
    
    # Adapter le format : soit {"0": "classe1", "1": "classe2"} soit {"classes": [...]}
    if isinstance(label_mapping, dict):
        if 'classes' in label_mapping:
            # Format avec clé 'classes'
            classes = label_mapping['classes']
            label_info = label_mapping
        else:
            # Format direct index → classe (votre cas actuel)
            classes = [label_mapping[str(i)] for i in range(len(label_mapping))]
            label_info = {'classes': classes, 'mapping': label_mapping}
    else:
        raise ValueError('Invalid label mapping format')
    
    if not classes:
        raise ValueError('Invalid label mapping: missing classes')
    return model, label_info


def load_and_preprocess_image(image_bytes_or_path) -> np.ndarray:
    if isinstance(image_bytes_or_path, (str, Path)):
        img = Image.open(image_bytes_or_path).convert('RGB').resize((224, 224))
    else:
        img = Image.open(io.BytesIO(image_bytes_or_path)).convert('RGB').resize((224, 224))
    arr = np.asarray(img).astype(np.float32)
    arr = mobilenet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model: tf.keras.Model, classes: list, image_source) -> Tuple[str, float]:
    x = load_and_preprocess_image(image_source)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    score = float(preds[idx])
    label = classes[idx]
    return label, score


def cmd_predict(image_path: str) -> None:
    model, label_info = load_model_and_labels()
    classes = label_info['classes']
    label, score = predict_image(model, classes, image_path)
    print(json.dumps({'image_path': image_path, 'predicted_label': label, 'score': round(score, 6)}, indent=2))


def cmd_predict_sample(num: int = 50, seed: int = 42) -> None:
    import pandas as pd
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['image_path'] = df['image'].apply(lambda x: str(IMAGES_DIR / x) if isinstance(x, str) else None)
    df = df[df['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
    if len(df) == 0:
        raise RuntimeError('No images found in dataset directory.')
    df = df.sample(n=min(num, len(df)), random_state=seed).reset_index(drop=True)

    model, label_info = load_model_and_labels()
    classes = label_info['classes']

    rows = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label, score = predict_image(model, classes, img_path)
        rows.append({'image_path': img_path, 'predicted_label': label, 'score': score})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f'Wrote {len(out_df)} predictions to {PREDICTIONS_CSV}')


def build_app() -> 'FastAPI':
    if not HAVE_FASTAPI:
        raise RuntimeError('FastAPI not installed. Install with: pip install fastapi uvicorn')
    app = FastAPI(title='Place de Marché — Product Classifier API', version='1.0.0')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    model, label_info = load_model_and_labels()
    classes = label_info['classes']

    @app.get('/health')
    def health():
        return {'status': 'ok', 'num_classes': len(classes)}

    @app.post('/predict')
    async def predict(file: UploadFile = File(...)):
        data = await file.read()
        label, score = predict_image(model, classes, data)
        return {'predicted_label': label, 'score': score}

    return app


def cmd_serve(host: str = '0.0.0.0', port: int = 8000) -> None:
    if not HAVE_UVICORN or not HAVE_FASTAPI:
        raise RuntimeError('FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn')
    app = build_app()
    uvicorn.run(app, host=host, port=port, log_level='info')


def main():
    parser = argparse.ArgumentParser(description='P6 API and CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_srv = sub.add_parser('serve', help='Run FastAPI server')
    p_srv.add_argument('--host', default='0.0.0.0')
    p_srv.add_argument('--port', type=int, default=8000)

    p_pred = sub.add_parser('predict', help='Predict on one image (CLI)')
    p_pred.add_argument('--image', required=True, help='Path to image file')

    p_batch = sub.add_parser('predict-sample', help='Generate predictions.csv on dataset sample')
    p_batch.add_argument('--num', type=int, default=50, help='Number of images to sample')
    p_batch.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.cmd == 'serve':
        cmd_serve(host=args.host, port=args.port)
    elif args.cmd == 'predict':
        cmd_predict(args.image)
    elif args.cmd == 'predict-sample':
        cmd_predict_sample(num=args.num, seed=args.seed)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


