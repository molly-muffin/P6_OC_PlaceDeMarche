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
  - data/artifacts/deep_learning_p6_multimodal.keras (Meilleur modèle: 94.4% accuracy)
  - data/artifacts/tfidf_vectorizer.pkl
  - data/artifacts/image_scaler.pkl
  - data/artifacts/label_mapping.json
"""

import argparse
import io
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

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

# Meilleur modèle: Deep Learning Multimodal (94.4% accuracy)
MODEL_PATH = ARTIFACTS_DIR / 'deep_learning_p6_multimodal.keras'
TFIDF_PATH = ARTIFACTS_DIR / 'tfidf_vectorizer.pkl'
IMAGE_SCALER_PATH = ARTIFACTS_DIR / 'image_scaler.pkl'
LABELS_PATH = ARTIFACTS_DIR / 'label_mapping.json'

# Predictions.csv dans le même dossier que le script
SCRIPT_DIR = Path(__file__).parent.resolve()
PREDICTIONS_CSV = SCRIPT_DIR / 'predictions.csv'

# Modèle MobileNetV2 pour extraction de features
MOBILENET_MODEL = None  # Chargé une seule fois


def load_mobilenet_base():
    """Charge le modèle MobileNetV2 pour extraction de features (une seule fois)"""
    global MOBILENET_MODEL
    if MOBILENET_MODEL is None:
        from tensorflow.keras.applications import MobileNetV2
        MOBILENET_MODEL = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        print('✅ MobileNetV2 chargé pour extraction de features')
    return MOBILENET_MODEL


def load_model_and_preprocessors() -> Tuple[tf.keras.Model, Dict, object, object]:
    """
    Charge le modèle multimodal et tous les préprocesseurs nécessaires.
    
    Returns:
        model: Modèle Deep Learning Multimodal (94.4% accuracy)
        label_info: Informations sur les classes
        tfidf_vectorizer: Vectoriseur TF-IDF pour le texte
        image_scaler: StandardScaler pour les features image
    """
    # Vérification des fichiers
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train and export it first.")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label mapping not found at {LABELS_PATH}.")
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {TFIDF_PATH}.")
    if not IMAGE_SCALER_PATH.exists():
        raise FileNotFoundError(f"Image scaler not found at {IMAGE_SCALER_PATH}.")
    
    # Chargement du modèle multimodal
    print(f'📦 Chargement du modèle multimodal depuis {MODEL_PATH.name}...')
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Chargement des labels
    with open(LABELS_PATH, 'r') as f:
        label_mapping = json.load(f)
    
    # Extraction des classes
    if isinstance(label_mapping, dict):
        if 'classes' in label_mapping:
            classes = label_mapping['classes']
            label_info = label_mapping
        else:
            classes = [label_mapping[str(i)] for i in range(len(label_mapping))]
            label_info = {'classes': classes, 'mapping': label_mapping}
    else:
        raise ValueError('Invalid label mapping format')
    
    if not classes:
        raise ValueError('Invalid label mapping: missing classes')
    
    # Chargement du vectoriseur TF-IDF
    print(f'📦 Chargement du vectoriseur TF-IDF...')
    with open(TFIDF_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Chargement du scaler image
    print(f'📦 Chargement du scaler image...')
    with open(IMAGE_SCALER_PATH, 'rb') as f:
        image_scaler = pickle.load(f)
    
    print(f'✅ Modèle multimodal chargé avec succès ({len(classes)} classes)')
    return model, label_info, tfidf_vectorizer, image_scaler


def extract_image_features(image_bytes_or_path, scaler) -> np.ndarray:
    """
    Extrait les features MobileNetV2 d'une image et applique la normalisation.
    
    Args:
        image_bytes_or_path: Chemin ou bytes de l'image
        scaler: StandardScaler pour normaliser les features
    
    Returns:
        Features normalisées (1280 dimensions)
    """
    # Chargement et prétraitement de l'image
    if isinstance(image_bytes_or_path, (str, Path)):
        img = Image.open(image_bytes_or_path).convert('RGB').resize((224, 224))
    else:
        img = Image.open(io.BytesIO(image_bytes_or_path)).convert('RGB').resize((224, 224))
    
    # Conversion en array et prétraitement MobileNetV2
    arr = np.asarray(img).astype(np.float32)
    arr = mobilenet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    
    # Extraction des features avec MobileNetV2
    mobilenet = load_mobilenet_base()
    features = mobilenet.predict(arr, verbose=0)[0]
    
    # Normalisation avec le scaler entraîné
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    return features_scaled


def clean_text(text: str) -> str:
    """
    Nettoie le texte (même pipeline que dans le notebook).
    
    Args:
        text: Texte brut
    
    Returns:
        Texte nettoyé (minuscules, sans stopwords, lemmatisé)
    """
    import re
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Télécharger les ressources si nécessaire
        try:
            stopwords.words('english')
        except:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        
        STOPWORDS = set(stopwords.words('english'))
        LEMMATIZER = WordNetLemmatizer()
        TOKENIZER_RE = re.compile(r"[A-Za-z]+")
        
        if not isinstance(text, str):
            return ''
        
        text = text.lower()
        tokens = TOKENIZER_RE.findall(text)
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
        return ' '.join(tokens)
    except Exception as e:
        print(f'⚠️ Erreur nettoyage texte: {e}')
        return text.lower() if isinstance(text, str) else ''


def predict_multimodal(model: tf.keras.Model, classes: list, 
                       text: str, image_source,
                       tfidf_vectorizer, image_scaler) -> Tuple[str, float]:
    """
    Prédiction avec le modèle multimodal (texte + image).
    
    Args:
        model: Modèle Deep Learning Multimodal
        classes: Liste des noms de classes
        text: Description textuelle du produit
        image_source: Chemin ou bytes de l'image
        tfidf_vectorizer: Vectoriseur TF-IDF pour le texte
        image_scaler: Scaler pour les features image
    
    Returns:
        (label, score): Classe prédite et score de confiance
    """
    # 1. Extraction des features textuelles
    text_clean = clean_text(text)
    text_features = tfidf_vectorizer.transform([text_clean])
    text_features_dense = text_features.toarray()
    
    # 2. Extraction des features visuelles
    image_features = extract_image_features(image_source, image_scaler)
    
    # 3. Prédiction avec le modèle multimodal
    preds = model.predict([text_features_dense, image_features], verbose=0)[0]
    
    # 4. Extraction de la classe avec la plus haute probabilité
    idx = int(np.argmax(preds))
    score = float(preds[idx])
    label = classes[idx]
    
    return label, score


def get_text_from_dataset(image_path: str) -> str:
    """
    Récupère le texte (nom + description) associé à une image depuis le dataset Flipkart.
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        Texte combiné (product_name + description) ou None si non trouvé
    """
    try:
        # Charger le dataset
        if not CSV_PATH.exists():
            return None
        
        df = pd.read_csv(CSV_PATH)
        
        # Extraire le nom du fichier depuis le chemin
        image_filename = Path(image_path).name
        
        # Chercher la ligne correspondante dans le dataset
        matching_row = df[df['image'] == image_filename]
        
        if len(matching_row) == 0:
            return None
        
        # Extraire nom et description
        product_name = matching_row['product_name'].iloc[0] if 'product_name' in matching_row else ''
        description = matching_row['description'].iloc[0] if 'description' in matching_row else ''
        
        # Combiner
        text = f"{product_name} {description}".strip()
        
        return text if text else None
        
    except Exception as e:
        print(f'⚠️ Erreur lors de la récupération du texte: {e}')
        return None


def cmd_predict(image_path: str, text: str = None) -> None:
    """
    Prédiction CLI sur une image avec description optionnelle.
    
    Args:
        image_path: Chemin vers l'image
        text: Description textuelle du produit (optionnel, cherche automatiquement dans le dataset)
    """
    # Chargement des modèles et préprocesseurs
    model, label_info, tfidf_vec, img_scaler = load_model_and_preprocessors()
    classes = label_info['classes']
    
    # Si pas de texte fourni, essayer de le récupérer du dataset
    if text is None:
        text = get_text_from_dataset(image_path)
        if text is None:
            # Fallback : utiliser le nom du fichier
            text = Path(image_path).stem.replace('_', ' ').replace('-', ' ')
            print(f'⚠️ Image non trouvée dans le dataset, utilisation du nom de fichier: "{text}"')
        else:
            print(f'✅ Description récupérée du dataset Flipkart')
    
    # Prédiction multimodale
    label, score = predict_multimodal(model, classes, text, image_path, tfidf_vec, img_scaler)
    
    result = {
        'image_path': image_path,
        'text_input': text[:100] + '...' if len(text) > 100 else text,
        'predicted_label': label,
        'confidence_score': round(score, 6),
        'model': 'Deep Learning Multimodal (94.4% accuracy)'
    }
    
    print(json.dumps(result, indent=2))


def cmd_predict_sample(num: int = 50, seed: int = 42) -> None:
    """
    Génère predictions.csv sur un échantillon du dataset Flipkart.
    
    Args:
        num: Nombre d'échantillons à prédire
        seed: Graine aléatoire pour reproductibilité
    """
    import time
    
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    
    # Fonction helper pour extraire la vraie catégorie
    def parse_true_category(cat_str: str) -> str:
        """Extrait la catégorie principale depuis product_category_tree"""
        if not isinstance(cat_str, str) or len(cat_str) == 0:
            return 'Unknown'
        try:
            s = cat_str.strip()
            if s.startswith('['):
                s = s[1:-1]
            s = s.strip().strip('"')
            parts = [p.strip() for p in s.split('>>') if len(p.strip()) > 0]
            return parts[0] if parts else 'Unknown'
        except Exception:
            return 'Unknown'
    
    # Chargement du dataset
    print(f'📊 Chargement du dataset depuis {CSV_PATH.name}...')
    df = pd.read_csv(CSV_PATH)
    
    # Préparation des chemins d'images et textes
    df['image_path'] = df['image'].apply(lambda x: str(IMAGES_DIR / x) if isinstance(x, str) else None)
    df['text'] = (df['product_name'].fillna('') + ' ' + df['description'].fillna('')).astype(str)
    df['true_category'] = df['product_category_tree'].apply(parse_true_category)
    
    # Filtrage des images existantes
    df = df[df['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
    if len(df) == 0:
        raise RuntimeError('No images found in dataset directory.')
    
    # Échantillonnage
    df = df.sample(n=min(num, len(df)), random_state=seed).reset_index(drop=True)
    print(f'📦 {len(df)} produits sélectionnés pour prédiction')

    # Chargement du modèle multimodal
    model, label_info, tfidf_vec, img_scaler = load_model_and_preprocessors()
    classes = label_info['classes']

    # Prédictions sur tous les échantillons
    print(f'🔮 Prédictions en cours...')
    rows = []
    total_time = 0
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        text = row['text']
        true_category = row['true_category']
        
        try:
            # Mesure du temps de prédiction
            start_time = time.time()
            label, score = predict_multimodal(model, classes, text, img_path, tfidf_vec, img_scaler)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Vérification de la prédiction
            is_correct = (label == true_category)
            
            rows.append({
                'image_path': img_path,
                'product_name': row.get('product_name', ''),
                'description': row.get('description', '')[:100] + '...' if len(row.get('description', '')) > 100 else row.get('description', ''),
                'true_category': true_category,
                'predicted_label': label,
                'confidence_score': round(score, 4),
                'is_correct': is_correct,
                'processing_time_ms': round(processing_time * 1000, 2)  # en millisecondes
            })
        except Exception as e:
            print(f'⚠️ Erreur sur {img_path}: {e}')
            continue
        
        # Progression
        if (idx + 1) % 10 == 0:
            print(f'  Traité {idx + 1}/{len(df)} produits...')

    # Sauvegarde des résultats
    out_df = pd.DataFrame(rows)
    out_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f'✅ {len(out_df)} prédictions sauvegardées dans {PREDICTIONS_CSV}')
    
    # Statistiques détaillées
    print(f'\n📈 STATISTIQUES DES PRÉDICTIONS:')
    print(f'   Nombre total: {len(out_df)}')
    print(f'   Prédictions correctes: {out_df["is_correct"].sum()} ({out_df["is_correct"].mean()*100:.1f}%)')
    print(f'   Prédictions incorrectes: {(~out_df["is_correct"]).sum()} ({(~out_df["is_correct"]).mean()*100:.1f}%)')
    print(f'   Score de confiance moyen: {out_df["confidence_score"].mean():.4f}')
    print(f'   Temps moyen par prédiction: {out_df["processing_time_ms"].mean():.2f} ms')
    print(f'   Temps total: {total_time:.2f} secondes')
    
    print(f'\n📊 Distribution des prédictions:')
    for label, count in out_df['predicted_label'].value_counts().items():
        accuracy = out_df[out_df['predicted_label'] == label]['is_correct'].mean() * 100
        print(f'  {label}: {count} prédictions (accuracy: {accuracy:.1f}%)')


def build_app() -> 'FastAPI':
    """
    Construit l'application FastAPI avec le modèle multimodal.
    
    Returns:
        Application FastAPI configurée
    """
    if not HAVE_FASTAPI:
        raise RuntimeError('FastAPI not installed. Install with: pip install fastapi uvicorn')
    
    from fastapi import Form
    
    app = FastAPI(
        title='Place de Marché — Product Classifier API',
        description='API de classification multimodale (texte + image) avec Deep Learning (94.4% accuracy)',
        version='2.0.0'
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Chargement unique du modèle au démarrage
    print('🚀 Démarrage de l\'API...')
    model, label_info, tfidf_vec, img_scaler = load_model_and_preprocessors()
    classes = label_info['classes']
    print(f'✅ API prête avec {len(classes)} classes')

    @app.get('/')
    def root():
        """Page d'accueil de l'API"""
        return {
            'name': 'Place de Marché Product Classifier',
            'version': '2.0.0',
            'model': 'Deep Learning Multimodal',
            'accuracy': '94.4%',
            'endpoints': {
                'GET /health': 'Vérifier l\'état de l\'API',
                'POST /predict': 'Classifier un produit (texte + image)',
                'GET /classes': 'Lister les classes disponibles',
                'GET /docs': 'Documentation Swagger UI'
            }
        }

    @app.get('/health')
    def health():
        """Endpoint de santé pour monitoring"""
        return {
            'status': 'ok',
            'model': 'Deep Learning Multimodal',
            'num_classes': len(classes),
            'classes': classes
        }
    
    @app.get('/classes')
    def get_classes():
        """Liste les classes disponibles"""
        return {
            'num_classes': len(classes),
            'classes': classes
        }

    @app.post('/predict')
    async def predict(
        file: UploadFile = File(..., description='Image du produit'),
        text: str = Form(..., description='Description textuelle du produit (nom + description)')
    ):
        """
        Endpoint de prédiction multimodale.
        
        Args:
            file: Image du produit (JPEG, PNG, etc.)
            text: Description textuelle (nom + description du produit)
        
        Returns:
            predicted_label: Catégorie prédite
            confidence_score: Score de confiance (0-1)
            model: Nom du modèle utilisé
        """
        try:
            # Lecture de l'image uploadée
            image_data = await file.read()
            
            # Prédiction multimodale
            label, score = predict_multimodal(
                model, classes, text, image_data, tfidf_vec, img_scaler
            )
            
            return {
                'predicted_label': label,
                'confidence_score': round(score, 6),
                'text_input': text[:100] + '...' if len(text) > 100 else text,
                'model': 'Deep Learning Multimodal (94.4% accuracy)'
            }
        
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f'Erreur de prédiction: {str(e)}')

    return app


def cmd_serve(host: str = '0.0.0.0', port: int = 8000) -> None:
    if not HAVE_UVICORN or not HAVE_FASTAPI:
        raise RuntimeError('FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn')
    app = build_app()
    uvicorn.run(app, host=host, port=port, log_level='info')


def main():
    parser = argparse.ArgumentParser(
        description='P6 API et CLI - Classification multimodale de produits (94.4% accuracy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Démarrer l'API
  python script.py serve --host 0.0.0.0 --port 8000
  
  # Prédire sur une image avec description
  python script.py predict --image laptop.jpg --text "Dell Laptop 15 inch screen"
  
  # Prédire sur une image sans description (utilise le nom du fichier)
  python script.py predict --image laptop.jpg
  
  # Générer predictions.csv sur 100 échantillons
  python script.py predict-sample --num 100 --seed 42
        """
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # Commande: serve
    p_srv = sub.add_parser('serve', help='Démarrer le serveur FastAPI')
    p_srv.add_argument('--host', default='0.0.0.0', help='Adresse IP (défaut: 0.0.0.0)')
    p_srv.add_argument('--port', type=int, default=8000, help='Port (défaut: 8000)')

    # Commande: predict
    p_pred = sub.add_parser('predict', help='Prédire la catégorie d\'un produit')
    p_pred.add_argument('--image', required=True, help='Chemin vers l\'image du produit')
    p_pred.add_argument('--text', default=None, help='Description textuelle du produit (optionnel)')

    # Commande: predict-sample
    p_batch = sub.add_parser('predict-sample', help='Générer predictions.csv sur échantillon du dataset')
    p_batch.add_argument('--num', type=int, default=50, help='Nombre d\'échantillons (défaut: 50)')
    p_batch.add_argument('--seed', type=int, default=42, help='Graine aléatoire (défaut: 42)')

    args = parser.parse_args()
    
    # Dispatch des commandes
    if args.cmd == 'serve':
        cmd_serve(host=args.host, port=args.port)
    elif args.cmd == 'predict':
        cmd_predict(image_path=args.image, text=args.text)
    elif args.cmd == 'predict-sample':
        cmd_predict_sample(num=args.num, seed=args.seed)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
