"""
Chargement des données de test et d'entraînement.

Ce module fournit les fonctions pour charger les données JSON
et calculer des statistiques sur les ensembles de données.
"""

import json
from typing import List, Dict
from collections import defaultdict

from .metrics import get_document_name


def load_test_data(json_path: str) -> List[Dict]:
    """
    Charge les données de test depuis le fichier JSON.
    
    Args:
        json_path: Chemin vers le fichier JSON
        
    Returns:
        Liste des échantillons de test
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["test"]


def load_training_data(json_path: str) -> List[Dict]:
    """
    Charge les données d'entraînement depuis le fichier JSON.
    
    Args:
        json_path: Chemin vers le fichier JSON
        
    Returns:
        Liste des échantillons d'entraînement
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("train", [])


def count_training_pages_per_document(train_data: List[Dict]) -> Dict[str, int]:
    """
    Compte le nombre de pages d'entraînement par document.
    
    Args:
        train_data: Liste des échantillons d'entraînement
        
    Returns:
        Dictionnaire {nom_document: nombre_pages}
    """
    pages_per_doc = defaultdict(int)
    for sample in train_data:
        doc_name = get_document_name(sample['name'])
        pages_per_doc[doc_name] += 1
    return dict(pages_per_doc)
