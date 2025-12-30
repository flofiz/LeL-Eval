"""
Fonctions de normalisation de texte pour les variantes CER.

Ce module fournit différentes fonctions de normalisation pour comparer
les transcriptions de manière plus flexible (sans accents, minuscules, etc.).
"""

import unicodedata
from typing import Dict, Callable


def normalize_no_accents(text: str) -> str:
    """
    Supprime les accents du texte (é→e, à→a, etc.).
    Utilise la décomposition NFD puis filtre les marques diacritiques.
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def normalize_lowercase(text: str) -> str:
    """Convertit le texte en minuscules."""
    return text.lower()


def normalize_special_chars(text: str) -> str:
    """
    Normalise les caractères spéciaux (apostrophes, guillemets, tirets).
    Convertit les variantes typographiques vers leurs équivalents ASCII.
    """
    char_map = {
        # Apostrophes
        ''': "'", ''': "'", '‚': "'", '`': "'", 'ʼ': "'",
        # Guillemets
        '"': '"', '"': '"', '„': '"', '«': '"', '»': '"',
        '‹': "'", '›': "'",
        # Tirets
        '–': '-', '—': '-', '−': '-', '‐': '-', '‑': '-',
        # Points de suspension
        '…': '...',
        # Espaces spéciaux
        '\xa0': ' ', '\u2002': ' ', '\u2003': ' ', '\u2009': ' ',
    }
    for old, new in char_map.items():
        text = text.replace(old, new)
    return text


def normalize_no_punctuation(text: str) -> str:
    """
    Supprime toute la ponctuation du texte.
    Conserve les lettres, chiffres et espaces.
    """
    return ''.join(c for c in text if unicodedata.category(c)[0] != 'P')


def normalize_historical_abbrev(text: str) -> str:
    """
    Normalise les abréviations historiques des documents anciens.
    
    Règles de normalisation:
    - # → lt (abréviation courante dans les manuscrits médiévaux)
    
    Ces symboles étaient utilisés par les scribes pour abréger des
    séquences de lettres fréquentes.
    """
    abbrev_map = {
        '#': 'lt',  # # est une abréviation de "lt" dans les documents historiques
    }
    for old, new in abbrev_map.items():
        text = text.replace(old, new)
    return text


def normalize_full(text: str) -> str:
    """
    Applique toutes les normalisations:
    1. Normalisation des abréviations historiques
    2. Normalisation des caractères spéciaux
    3. Conversion en minuscules
    4. Suppression des accents
    5. Suppression de la ponctuation
    
    Retourne le texte "canonique" pour comparaison fondamentale.
    """
    text = normalize_historical_abbrev(text)  # D'abord les abréviations historiques
    text = normalize_special_chars(text)
    text = normalize_lowercase(text)
    text = normalize_no_accents(text)
    text = normalize_no_punctuation(text)
    return text


# Dictionnaire des normalisations pour itération
NORMALIZATIONS: Dict[str, Callable[[str], str]] = {
    'no_accents': normalize_no_accents,
    'lowercase': normalize_lowercase,
    'normalized_chars': normalize_special_chars,
    'no_punctuation': normalize_no_punctuation,
    'historical_abbrev': normalize_historical_abbrev,
    'normalized': normalize_full,
}

