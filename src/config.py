# Config centralisée du projet
import os

# Chemins
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data/processed')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports/figures')

# Coûts métier pour le credit scoring
COST_FALSE_NEGATIVE = 10  # Coût d'un défaut non détecté (perte de crédit)
COST_FALSE_POSITIVE = 1   # Coût d'un refus abusif (opportunité perdue)

# Entraînement
RANDOM_SEED = 42
DEFAULT_SCALE_POS_WEIGHT = 10  # Pondération XGBoost pour déséquilibre de classes
