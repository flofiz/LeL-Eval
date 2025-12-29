"""
Package d'évaluation pour les modèles HTR (Handwritten Text Recognition).

Ce package fournit des outils pour évaluer les modèles de transcription de texte manuscrit,
incluant le calcul de métriques (CER, IoU), la génération de rapports, et l'export LabelMe.

Modules:
    normalization: Fonctions de normalisation de texte
    models: Structures de données (BBox, etc.)
    metrics: Calculs de métriques d'évaluation
    tsv_parser: Parsing et manipulation de données TSV
    api_client: Client asynchrone pour l'API VLLM
    labelme_export: Export au format LabelMe
    report_generator: Génération de rapports Markdown
    html_report: Génération de rapports HTML interactifs
    data_loader: Chargement des données de test/entraînement
"""

# Normalization
from .normalization import (
    normalize_no_accents,
    normalize_lowercase,
    normalize_special_chars,
    normalize_no_punctuation,
    normalize_full,
    NORMALIZATIONS,
)

# Models
from .models import BBox

# Metrics
from .metrics import (
    calculate_cer,
    calculate_oriented_iou,
    evaluate_format,
    match_predictions_to_ground_truth,
    evaluate_sample,
    compute_per_document_stats,
    get_document_name,
)

# TSV Parser
from .tsv_parser import (
    parse_tsv_line,
    extract_tsv_from_response,
    bbox_to_4points,
)

# API Client
from .api_client import (
    call_vllm_api,
    process_sample,
    image_to_url,
)

# LabelMe Export
from .labelme_export import (
    get_image_dimensions,
    tsv_to_labelme,
)

# Report Generator
from .report_generator import generate_markdown_report
from .html_report import generate_html_report

# Data Loader
from .data_loader import (
    load_test_data,
    load_training_data,
    count_training_pages_per_document,
)

__all__ = [
    # Normalization
    'normalize_no_accents',
    'normalize_lowercase',
    'normalize_special_chars',
    'normalize_no_punctuation',
    'normalize_full',
    'NORMALIZATIONS',
    # Models
    'BBox',
    # Metrics
    'calculate_cer',
    'calculate_oriented_iou',
    'evaluate_format',
    'match_predictions_to_ground_truth',
    'evaluate_sample',
    'compute_per_document_stats',
    'get_document_name',
    # TSV Parser
    'parse_tsv_line',
    'extract_tsv_from_response',
    'bbox_to_4points',
    # API Client
    'call_vllm_api',
    'process_sample',
    'image_to_url',
    # LabelMe Export
    'get_image_dimensions',
    'tsv_to_labelme',
    # Report Generator
    'generate_markdown_report',
    'generate_html_report',
    # Data Loader
    'load_test_data',
    'load_training_data',
    'count_training_pages_per_document',
]
