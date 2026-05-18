"""
config.py
---------
All file paths in one place.
Everything is relative — works on any machine.
Just change NER_MODEL_PATH after training.
"""

import os

# Root of the project (where app.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

# NER model (saved after step3)
NER_MODEL_PATH = os.path.join(MODELS_DIR, 'ner_model')

# Ranking model files (saved after step4)
RANKING_MODEL_PATH = os.path.join(MODELS_DIR, 'ranking_model.pkl')
FEATURE_COLS_PATH  = os.path.join(MODELS_DIR, 'feature_columns.pkl')
SCALER_PATH        = os.path.join(MODELS_DIR, 'scaler.pkl')

# ONE raw CSV — cleaning + scaling happens inside the training notebook
CANDIDATES_CSV = os.path.join(DATA_DIR, 'candidates.csv')

# NER training data
NER_DATA_PATH = os.path.join(DATA_DIR, 'ner_training_data.json')