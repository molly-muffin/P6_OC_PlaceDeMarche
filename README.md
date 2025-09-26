<p align="center">
  <img src="https://user.oc-static.com/upload/2019/02/24/15510259240381_Projet%20textimage%20logo.png" alt="Projet P6" />
</p>

# Classification automatique des biens de consommation (Texte & Image)

## Contexte du projet : 
Ce projet a été réalisé dans le cadre du parcours AI Engineer d'OpenClassrooms. L'objectif est de développer un système de classification automatique des biens de consommation à partir de descriptions textuelles, en utilisant des techniques de machine learning et de traitement du langage naturel (NLP).

## Le projet inclu :
- un notebook contenant :
    - l'EDA complète, features texte/image, projections 2D, clustering + interprétations
    - le transfer learning (MobileNetV2), split stratifié, entraînement/fine‑tuning, métriques, export artefacts
- un script Python pour l'API FastAPI + CLI (prédiction image, batch → `predictions.csv`)
- un diaporama contenant la présentation visuelle de l'analyse des données

## Environnement de développement et outils utiles à la réalisation du projet :
`Python`

    Librairies : 
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - nltk
    - pillow
    - opencv-python (cv2)
    - tensorflow  (ou tensorflow-macos + tensorflow-metal sur Apple Silicon)
    - gensim (optionnel)
    - sentence-transformers (optionnel)
    - umap-learn (optionnel)
    - fastapi
    - uvicorn
    - requests

`Jupyter notebook`

`Google slides` 

`GitHub` 

## Environnement (recommandé macOS Apple Silicon)
Activer le venv Python 3.10 du projet et installer les dépendances:
```bash
source /Users/laureendademeule/Documents/Projets/P6/p6_env_py310/bin/activate
python -m pip install --upgrade pip
# Apple Silicon (accélération Metal)
python -m pip install fastapi uvicorn pillow numpy pandas scikit-learn tensorflow-macos tensorflow-metal requests
# (Alternative CPU) python -m pip install tensorflow
```

## Exécuter les notebooks
1. Ouvrir Jupyter/Lab et exécuter `notebooks/Laureen_Dademeule_1_notebook_pretraitement_feature_extraction_faisabilite_092025.ipynb` (EDA, features, clustering)
2. Exécuter `notebooks/Laureen_Dademeule_2_notebook_classification_092025.ipynb` pour entraîner et exporter les artefacts:

## Lancer l’API
Depuis la racine du projet (après avoir généré les artefacts ci‑dessus):
```bash
python Laureen_Dademeule_3_script_Python_092025.py serve --host 0.0.0.0 --port 8000
```
Endpoints:
- Healthcheck: `GET /health`
- Prédiction image: `POST /predict` (multipart `file`)
- Docs Swagger: `http://localhost:8000/docs`

Exemple cURL:
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ABSOLUTE/PATH/TO/image.jpg"
```

## CLI (sans serveur)
- Prédire une image:
```bash
python Laureen_Dademeule_3_script_Python_092025.py predict --image /ABSOLUTE/PATH/TO/image.jpg
```
- Échantillon et CSV:
```bash
python Laureen_Dademeule_3_script_Python_092025.py predict-sample --num 50
# → predictions.csv
```

> Laureenda Demeule
> OpenClassroom
