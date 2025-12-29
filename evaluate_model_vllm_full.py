import json
import base64
import re
import numpy as np
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
from collections import defaultdict
import editdistance
from tqdm.asyncio import tqdm as async_tqdm
from Utils.promptes import Qwen2_5_SYSTEM_MESSAGE, PROMPTS, Qwen2_5_OLD_SYSTEM_MESSAGE
import os
import unicodedata
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les images

# =============================================================================
# Fonctions de normalisation de texte pour les variantes CER
# =============================================================================

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


def normalize_full(text: str) -> str:
    """
    Applique toutes les normalisations:
    1. Normalisation des caractères spéciaux
    2. Conversion en minuscules
    3. Suppression des accents
    4. Suppression de la ponctuation
    
    Retourne le texte "canonique" pour comparaison fondamentale.
    """
    text = normalize_special_chars(text)
    text = normalize_lowercase(text)
    text = normalize_no_accents(text)
    text = normalize_no_punctuation(text)
    return text


# Dictionnaire des normalisations pour itération
NORMALIZATIONS = {
    'no_accents': normalize_no_accents,
    'lowercase': normalize_lowercase,
    'normalized_chars': normalize_special_chars,
    'no_punctuation': normalize_no_punctuation,
    'normalized': normalize_full,
}


# Import du module d'analyse d'erreurs
from error_analysis import (
    LineErrorAnalysis, PageErrorAnalysis,
    analyze_line_errors, detect_line_fusion,
    aggregate_error_stats, page_analysis_to_dict
)


def get_document_name(page_name: str) -> str:
    """
    Extrait le nom du document à partir du nom de la page.
    
    Gère deux cas:
    - Si le nom se termine par 'D' ou 'G' (pages gauche/droite), 
      supprime les 2 derniers segments
    - Sinon, supprime juste le dernier segment (numéro de page)
    
    Exemples:
        FRAD021_C_03003_001_D -> FRAD021_C_03003
        FRAD021_C_03003_001_G -> FRAD021_C_03003
        FRAD021_C_03003_001 -> FRAD021_C_03003
    """
    file_name = page_name.split(".")[0]  # Supprimer l'extension
    if file_name[-1] == "D" or file_name[-1] == "G":
        document_name = "_".join(file_name.split("_")[:-2])
    else:
        document_name = "_".join(file_name.split("_")[:-1])
    return document_name


def compute_per_document_stats(all_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Regroupe les métriques par document et calcule les statistiques par document.
    Le CER est pondéré par le nombre de caractères de chaque page.
    
    Inclut les variantes CER:
    - avg_cer: CER original
    - avg_cer_no_accents, avg_cer_lowercase, avg_cer_normalized_chars,
      avg_cer_no_punctuation, avg_cer_normalized
    
    Args:
        all_metrics: Liste des métriques par page
        
    Returns:
        Dictionnaire avec les statistiques par document
    """
    # Regrouper les métriques par document
    per_document = defaultdict(list)
    for metrics in all_metrics:
        doc_name = get_document_name(metrics['name'])
        per_document[doc_name].append(metrics)
    
    # Calculer les statistiques par document
    document_stats = {}
    for doc_name, doc_metrics in per_document.items():
        # CER pondéré par caractères: somme des erreurs / somme des caractères
        total_edit_dist = sum(m['total_edit_distance'] for m in doc_metrics)
        total_chars = sum(m['total_gt_chars'] for m in doc_metrics)
        weighted_cer = total_edit_dist / total_chars if total_chars > 0 else 1.0
        
        doc_stat = {
            'avg_cer': float(weighted_cer),  # CER pondéré par caractères
            'total_gt_chars': total_chars,  # Pour pondération au niveau global
            'total_edit_distance': total_edit_dist,  # Pour recalcul de moyenne pondérée
            'avg_iou': float(np.mean([m['iou'] for m in doc_metrics])),
            'avg_format_score': float(np.mean([m['format_score'] for m in doc_metrics])),
            'avg_recall': float(np.mean([m['recall'] for m in doc_metrics])),
            'avg_precision': float(np.mean([m['precision'] for m in doc_metrics])),
            'num_pages': len(doc_metrics),
            'total_matched': sum(m['num_matched'] for m in doc_metrics),
            'total_predicted': sum(m['num_predicted'] for m in doc_metrics),
            'total_ground_truth': sum(m['num_ground_truth'] for m in doc_metrics),
            'pages': [m['name'] for m in doc_metrics]
        }
        
        # Ajouter les variantes CER
        for variant_name in NORMALIZATIONS.keys():
            edit_dist_key = f'total_edit_distance_{variant_name}'
            chars_key = f'total_gt_chars_{variant_name}'
            
            variant_edit_dist = sum(m[edit_dist_key] for m in doc_metrics)
            variant_chars = sum(m[chars_key] for m in doc_metrics)
            variant_cer = variant_edit_dist / variant_chars if variant_chars > 0 else 1.0
            
            doc_stat[f'avg_cer_{variant_name}'] = float(variant_cer)
            doc_stat[f'total_edit_distance_{variant_name}'] = variant_edit_dist
            doc_stat[f'total_gt_chars_{variant_name}'] = variant_chars
        
        document_stats[doc_name] = doc_stat
    
    return document_stats


@dataclass
class BBox:
    """Représente une boîte englobante orientée"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    angle: float
    text: str


def parse_tsv_line(line: str) -> BBox:
    """Parse une ligne TSV au format: transcription\txmin\tymin\txmax\tymax\tangle"""
    parts = line.strip().split('\t')
    if len(parts) != 6:
        raise ValueError(f"Format TSV invalide: {line}")
    
    text = parts[0]
    xmin, ymin, xmax, ymax, angle = map(float, parts[1:])
    
    return BBox(xmin, ymin, xmax, ymax, angle, text)


def bbox_to_4points(bbox: BBox) -> List[List[float]]:
    """
    Convertit une bounding box (2 points + angle) en 4 points.
    L'angle est en degrés, entre -90 et 90.
    Les coordonnées sont en pixels réels.
    
    Returns:
        Liste de 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dans l'ordre: top-left, top-right, bottom-right, bottom-left
    """
    # Centre de la boîte
    cx = (bbox.xmin + bbox.xmax) / 2
    cy = (bbox.ymin + bbox.ymax) / 2
    
    # Dimensions
    width = bbox.xmax - bbox.xmin
    height = bbox.ymax - bbox.ymin
    
    # Convertir l'angle en radians
    angle_rad = np.deg2rad(bbox.angle)
    
    # Cosinus et sinus pour la rotation
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Points du rectangle non-rotaté (centrés sur l'origine)
    # top-left, top-right, bottom-right, bottom-left
    half_w = width / 2
    half_h = height / 2
    
    points_local = [
        [-half_w, -half_h],  # top-left
        [half_w, -half_h],   # top-right
        [half_w, half_h],    # bottom-right
        [-half_w, half_h]    # bottom-left
    ]
    
    # Appliquer la rotation et translater au centre
    points_rotated = []
    for px, py in points_local:
        # Rotation
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        # Translation
        points_rotated.append([rx + cx, ry + cy])
    
    return points_rotated


def extract_tsv_from_response(response: str) -> str:
    """
    Extrait le contenu TSV entre les balises ```tsv et ```.
    
    Gère les sorties tronquées:
    - Si la réponse commence par ```tsv mais ne se termine pas par ```,
      on extrait quand même le contenu et on supprime la dernière ligne
      si elle est incomplète (pas parsable).
    """
    # Cas normal: balise ouvrante et fermante présentes
    pattern = r"```tsv\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Cas tronqué: commence par ```tsv mais pas de fermeture
    truncated_pattern = r"```tsv\s*(.*)"
    truncated_match = re.search(truncated_pattern, response, re.DOTALL)
    
    if truncated_match:
        content = truncated_match.group(1).strip()
        lines = content.split('\n')
        
        # Vérifier si la dernière ligne est parsable (6 colonnes)
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                parts = last_line.split('\t')
                # Une ligne valide doit avoir 6 colonnes (text + 5 coords)
                if len(parts) != 6:
                    # Dernière ligne incomplète, on la supprime
                    lines = lines[:-1]
                else:
                    # Vérifier que les 5 derniers éléments sont des nombres
                    try:
                        for i in range(1, 6):
                            float(parts[i])
                    except (ValueError, IndexError):
                        # Dernière ligne invalide, on la supprime
                        lines = lines[:-1]
        
        return '\n'.join(lines).strip()
    
    return ""


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calcule le Character Error Rate (CER)"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def calculate_oriented_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calcule l'IoU pour des boîtes orientées.
    Simplifié: utilise l'intersection des rectangles axis-aligned.
    Pour une implémentation complète avec rotation, utiliser shapely.
    """
    # Intersection
    x_left = max(bbox1.xmin, bbox2.xmin)
    y_top = max(bbox1.ymin, bbox2.ymin)
    x_right = min(bbox1.xmax, bbox2.xmax)
    y_bottom = min(bbox1.ymax, bbox2.ymax)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    bbox1_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    bbox2_area = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    # Pénalité pour la différence d'angle (en degrés)
    angle_diff = abs(bbox1.angle - bbox2.angle)
    angle_diff = min(angle_diff, 360 - angle_diff)  # Normaliser à [0, 180]
    angle_penalty = 1.0 - (angle_diff / 180.0) * 0.3  # Pénalité max de 30%
    
    iou = (intersection_area / union_area) * angle_penalty
    return iou


def evaluate_format(predicted_tsv: str, ground_truth_tsv: str) -> float:
    """
    Évalue la qualité du formatage de la réponse.
    Retourne un score entre 0 et 1.
    """
    if not predicted_tsv:
        return 0.0
    
    pred_lines = [l for l in predicted_tsv.strip().split('\n') if l.strip()]
    gt_lines = [l for l in ground_truth_tsv.strip().split('\n') if l.strip()]
    
    if len(gt_lines) == 0:
        return 1.0 if len(pred_lines) == 0 else 0.0
    
    score = 0.0
    
    # Score pour le nombre de lignes correctes
    num_lines_score = 1.0 - abs(len(pred_lines) - len(gt_lines)) / max(len(gt_lines), 1)
    score += num_lines_score * 0.4
    
    # Score pour le format de chaque ligne
    valid_lines = 0
    for line in pred_lines:
        parts = line.split('\t')
        if len(parts) == 6:
            try:
                # Vérifier que les 5 derniers éléments sont des nombres
                float(parts[1])
                float(parts[2])
                float(parts[3])
                float(parts[4])
                float(parts[5])
                valid_lines += 1
            except ValueError:
                pass
    
    if len(pred_lines) > 0:
        format_score = valid_lines / len(pred_lines)
        score += format_score * 0.6
    
    return score


def match_predictions_to_ground_truth(predictions: List[BBox], 
                                     ground_truths: List[BBox]) -> List[Tuple[BBox, BBox]]:
    """
    Apparie les prédictions aux vérités terrain basé sur l'IoU maximal.
    Retourne une liste de tuples (pred, gt).
    """
    matched_pairs = []
    used_gt_indices = set()
    
    # Pour chaque prédiction, trouver la meilleure vérité terrain
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt_indices:
                continue
            
            iou = calculate_oriented_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Appariement si IoU > seuil
        if best_iou > 0.1 and best_gt_idx != -1:
            matched_pairs.append((pred, ground_truths[best_gt_idx]))
            used_gt_indices.add(best_gt_idx)
    
    return matched_pairs


def evaluate_sample(predicted_tsv: str, ground_truth_tsv: str, page_name: str = "") -> Tuple[Dict, Optional[PageErrorAnalysis]]:
    """
    Évalue un échantillon et retourne les métriques + analyse d'erreurs détaillée.
    
    Calcule le CER pour plusieurs variantes de normalisation:
    - cer: CER original (sans normalisation)
    - cer_no_accents: CER sans accents
    - cer_lowercase: CER en minuscules
    - cer_normalized_chars: CER avec caractères normalisés
    - cer_no_punctuation: CER sans ponctuation
    - cer_normalized: CER avec toutes les normalisations
    
    Returns:
        Tuple (metrics_dict, page_error_analysis)
    """
    
    # Parse les TSV
    try:
        pred_bboxes = [parse_tsv_line(line) for line in predicted_tsv.strip().split('\n') 
                      if line.strip()]
    except Exception as e:
        print(f"Erreur lors du parsing de la prédiction: {e}")
        pred_bboxes = []
    
    try:
        gt_bboxes = [parse_tsv_line(line) for line in ground_truth_tsv.strip().split('\n') 
                    if line.strip()]
    except Exception as e:
        print(f"Erreur lors du parsing de la vérité terrain: {e}")
        gt_bboxes = []
    
    # Appariement des prédictions aux vérités terrain
    matched_pairs = match_predictions_to_ground_truth(pred_bboxes, gt_bboxes)
    
    # Calcul des métriques avec pondération par nombre de caractères
    # CER de base
    total_edit_distance = 0
    total_gt_chars = 0
    total_iou = 0.0
    
    # CER pour chaque variante de normalisation
    variant_edit_distances = {key: 0 for key in NORMALIZATIONS.keys()}
    variant_gt_chars = {key: 0 for key in NORMALIZATIONS.keys()}
    
    # Créer l'analyse d'erreurs
    page_analysis = PageErrorAnalysis(page_name=page_name, line_analyses=[])
    
    # Tracker les indices matchés
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for pred, gt in matched_pairs:
        # Calculer la distance d'édition brute (CER de base)
        edit_dist = editdistance.eval(gt.text, pred.text)
        gt_chars = len(gt.text)
        
        total_edit_distance += edit_dist
        total_gt_chars += gt_chars
        
        # Calculer les distances pour chaque variante normalisée
        for variant_name, normalize_func in NORMALIZATIONS.items():
            gt_normalized = normalize_func(gt.text)
            pred_normalized = normalize_func(pred.text)
            variant_edit_dist = editdistance.eval(gt_normalized, pred_normalized)
            variant_gt_len = len(gt_normalized)
            
            variant_edit_distances[variant_name] += variant_edit_dist
            variant_gt_chars[variant_name] += variant_gt_len
        
        iou = calculate_oriented_iou(pred, gt)
        total_iou += iou
        
        # Analyse détaillée de cette ligne
        line_analysis = analyze_line_errors(gt.text, pred.text)
        page_analysis.line_analyses.append(line_analysis)
        
        # Tracker les matchs
        for i, gt_box in enumerate(gt_bboxes):
            if gt_box.text == gt.text:
                matched_gt_indices.add(i)
                break
        for i, pred_box in enumerate(pred_bboxes):
            if pred_box.text == pred.text:
                matched_pred_indices.add(i)
                break
    
    # Lignes GT non matchées (manquantes)
    unmatched_gt_texts = []
    unmatched_gt_chars = 0
    unmatched_variant_chars = {key: 0 for key in NORMALIZATIONS.keys()}
    
    for i, gt_box in enumerate(gt_bboxes):
        if i not in matched_gt_indices:
            unmatched_gt_chars += len(gt_box.text)
            unmatched_gt_texts.append(gt_box.text)
            page_analysis.missing_lines.append(gt_box.text)
            
            # Compter les caractères non matchés pour chaque variante
            for variant_name, normalize_func in NORMALIZATIONS.items():
                unmatched_variant_chars[variant_name] += len(normalize_func(gt_box.text))
    
    # Lignes prédites non matchées (extras)
    extra_pred_texts = []
    for i, pred_box in enumerate(pred_bboxes):
        if i not in matched_pred_indices:
            extra_pred_texts.append(pred_box.text)
            page_analysis.extra_lines.append(pred_box.text)
    
    # Détecter les fusions de lignes
    # Une prédiction non matchée peut contenir plusieurs lignes GT non matchées
    for extra_text in extra_pred_texts:
        fused_gts = detect_line_fusion(extra_text, unmatched_gt_texts)
        if fused_gts:
            page_analysis.merged_lines.append((fused_gts, extra_text))
    
    # CER pondéré: (total_edit_distance + chars_non_matchées) / total_chars_gt
    total_gt_chars_all = total_gt_chars + unmatched_gt_chars
    total_errors = total_edit_distance + unmatched_gt_chars
    
    # Calculer le CER pondéré pour chaque variante
    variant_total_errors = {}
    variant_total_chars = {}
    variant_cer = {}
    
    for variant_name in NORMALIZATIONS.keys():
        total_chars = variant_gt_chars[variant_name] + unmatched_variant_chars[variant_name]
        total_errs = variant_edit_distances[variant_name] + unmatched_variant_chars[variant_name]
        
        variant_total_chars[variant_name] = total_chars
        variant_total_errors[variant_name] = total_errs
        variant_cer[variant_name] = total_errs / total_chars if total_chars > 0 else 1.0
    
    # Moyennes
    num_matched = len(matched_pairs)
    num_gt = len(gt_bboxes)
    
    # CER pondéré par caractères (inclut les GT non matchées comme erreurs)
    cer_weighted = total_errors / total_gt_chars_all if total_gt_chars_all > 0 else 1.0
    
    avg_iou = total_iou / num_matched if num_matched > 0 else 0.0
    
    # Recall (combien de vérités terrain ont été détectées)
    recall = num_matched / num_gt if num_gt > 0 else 0.0
    
    # Precision (combien de prédictions sont correctes)
    precision = num_matched / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    
    # Score de formatage
    format_score = evaluate_format(predicted_tsv, ground_truth_tsv)
    
    # Agréger les stats de la page
    page_analysis.total_insertions = sum(la.insertions for la in page_analysis.line_analyses)
    page_analysis.total_deletions = sum(la.deletions for la in page_analysis.line_analyses)
    page_analysis.total_substitutions = sum(la.substitutions for la in page_analysis.line_analyses)
    page_analysis.total_matches = sum(la.matches for la in page_analysis.line_analyses)
    page_analysis.total_gt_chars = total_gt_chars_all
    
    metrics = {
        'cer': cer_weighted,
        'total_gt_chars': total_gt_chars_all,
        'total_edit_distance': total_errors,
        'iou': avg_iou,
        'format_score': format_score,
        'recall': recall,
        'precision': precision,
        'num_matched': num_matched,
        'num_predicted': len(pred_bboxes),
        'num_ground_truth': num_gt,
    }
    
    # Ajouter les métriques pour chaque variante CER
    for variant_name in NORMALIZATIONS.keys():
        metrics[f'cer_{variant_name}'] = variant_cer[variant_name]
        metrics[f'total_edit_distance_{variant_name}'] = variant_total_errors[variant_name]
        metrics[f'total_gt_chars_{variant_name}'] = variant_total_chars[variant_name]
    
    return metrics, page_analysis


def load_test_data(json_path: str) -> List[Dict]:
    """Charge les données de test depuis le fichier JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["test"]


def load_training_data(json_path: str) -> List[Dict]:
    """Charge les données d'entraînement depuis le fichier JSON"""
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


def get_image_dimensions(base64_image: str) -> Tuple[int, int]:
    """Récupère les dimensions d'une image base64"""
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image.width, image.height


def tsv_to_labelme(tsv_content: str, image_base64: str, image_name: str) -> Dict:
    """
    Convertit un TSV au format LabelMe pour visualisation.
    
    Args:
        tsv_content: Contenu TSV avec les bounding boxes
        image_base64: Image encodée en base64
        image_name: Nom de l'image
    
    Returns:
        Dictionnaire au format LabelMe
    """
    # Récupérer les dimensions de l'image
    width, height = get_image_dimensions(image_base64)
    
    # Parser les bounding boxes
    shapes = []
    try:
        lines = [l for l in tsv_content.strip().split('\n') if l.strip()]
        for line in lines:
            bbox = parse_tsv_line(line)
            # Convertir en 4 points
            points = bbox_to_4points(bbox)
            
            shape = {
                "label": bbox.text,
                "points": points,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
    except Exception as e:
        print(f"Erreur lors de la conversion LabelMe pour {image_name}: {e}")
    
    # Format LabelMe
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": image_base64,
        "imageHeight": height,
        "imageWidth": width
    }
    
    return labelme_data


def image_to_url(base64_image: str) -> str:
    """Convertit une image base64 en URL data"""
    return f"data:image/jpeg;base64,{base64_image}"


async def call_vllm_api(session: aiohttp.ClientSession,
                       api_url: str,
                       system_prompt: str,
                       user_prompt: str,
                       image_base64: str,
                       temperature: float = 0.0,
                       max_tokens: int = 2048) -> str:
    """
    Envoie une requête asynchrone à l'API VLLM (format OpenAI)
    """
    
    # Construire le message avec l'image
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_url(image_base64)
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]
    
    payload = {
        "model": "Qwen3-VL-4B",  # Nom du modèle servi
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0
    }
    
    try:
        async with session.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                error_text = await response.text()
                print(f"Erreur API (status {response.status}): {error_text}")
                return ""
    except asyncio.TimeoutError:
        print("Timeout lors de l'appel API")
        return ""
    except Exception as e:
        print(f"Erreur lors de l'appel API: {e}")
        return ""


async def process_sample(session: aiohttp.ClientSession,
                        sample: Dict,
                        api_url: str,
                        system_prompt: str,
                        user_prompt: str,
                        semaphore: asyncio.Semaphore,
                        output_dir: str = None) -> Dict:
    """
    Traite un échantillon de manière asynchrone et sauvegarde au format LabelMe
    """
    async with semaphore:
        # Appeler l'API
        response_text = await call_vllm_api(
            session,
            api_url,
            system_prompt,
            user_prompt,
            sample['image']
        )
        
        # Extraire le TSV de la réponse
        predicted_tsv = extract_tsv_from_response(response_text)
        
        # Créer les fichiers LabelMe si output_dir est spécifié
        if output_dir:
            # Créer le sous-dossier labelme
            labelme_dir = os.path.join(output_dir, "labelme")
            os.makedirs(labelme_dir, exist_ok=True)
            
            # Prédiction
            pred_labelme = tsv_to_labelme(
                predicted_tsv,
                sample['image'],
                sample['name']
            )
            pred_filename = os.path.join(labelme_dir, f"{sample['name']}_pred.json")
            with open(pred_filename, 'w', encoding='utf-8') as f:
                json.dump(pred_labelme, f, indent=2, ensure_ascii=False)
            
            # Ground truth
            gt_labelme = tsv_to_labelme(
                sample['tsv'],
                sample['image'],
                sample['name']
            )
            gt_filename = os.path.join(labelme_dir, f"{sample['name']}_gt.json")
            with open(gt_filename, 'w', encoding='utf-8') as f:
                json.dump(gt_labelme, f, indent=2, ensure_ascii=False)
        
        return {
            'name': sample['name'],
            'predicted': predicted_tsv,
            'ground_truth': sample['tsv'],
            'full_response': response_text
        }


# =============================================================================
# Génération du rapport Markdown avec graphes
# =============================================================================

def generate_markdown_report(
    all_metrics: List[Dict],
    document_stats: Dict[str, Dict],
    error_stats: Optional[Dict],
    page_cer_variants: Dict[str, float],
    doc_cer_variants: Dict[str, float],
    training_pages_per_doc: Optional[Dict[str, int]] = None,
    output_dir: str = ".",
    report_name: str = "evaluation_report"
) -> str:
    """
    Génère un rapport Markdown avec des graphes de performance.
    
    Args:
        all_metrics: Métriques par page
        document_stats: Statistiques par document
        error_stats: Statistiques d'erreurs agrégées
        page_cer_variants: CER variants au niveau page
        doc_cer_variants: CER variants au niveau document
        training_pages_per_doc: Nombre de pages d'entraînement par document
        output_dir: Dossier de sortie pour les images
        report_name: Nom de base du rapport
    
    Returns:
        Chemin vers le fichier markdown généré
    """
    # Créer le dossier pour les images
    images_dir = os.path.join(output_dir, f"{report_name}_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Configuration matplotlib pour le style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    
    # Couleurs cohérentes
    colors = {
        'primary': '#2563eb',
        'secondary': '#7c3aed',
        'success': '#059669',
        'warning': '#d97706',
        'danger': '#dc2626',
        'info': '#0891b2',
        'gray': '#6b7280'
    }
    
    # =========================================================================
    # Graphe 1: Distribution du CER par page
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    cer_values = [m['cer'] * 100 for m in all_metrics]  # Convertir en %
    
    ax.hist(cer_values, bins=30, color=colors['primary'], alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(cer_values), color=colors['danger'], linestyle='--', 
               linewidth=2, label=f'Moyenne: {np.mean(cer_values):.1f}%')
    ax.axvline(np.median(cer_values), color=colors['success'], linestyle='--', 
               linewidth=2, label=f'Médiane: {np.median(cer_values):.1f}%')
    
    ax.set_xlabel('CER (%)', fontsize=12)
    ax.set_ylabel('Nombre de pages', fontsize=12)
    ax.set_title('Distribution du Character Error Rate (CER) par page', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    cer_dist_path = os.path.join(images_dir, 'cer_distribution.png')
    plt.savefig(cer_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 1b: Distribution du CER normalisé par page
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    cer_normalized_values = [m['cer_normalized'] * 100 for m in all_metrics]  # CER normalisé en %
    
    ax.hist(cer_normalized_values, bins=30, color=colors['secondary'], alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(cer_normalized_values), color=colors['danger'], linestyle='--', 
               linewidth=2, label=f'Moyenne: {np.mean(cer_normalized_values):.1f}%')
    ax.axvline(np.median(cer_normalized_values), color=colors['success'], linestyle='--', 
               linewidth=2, label=f'Médiane: {np.median(cer_normalized_values):.1f}%')
    
    ax.set_xlabel('CER normalisé (%)', fontsize=12)
    ax.set_ylabel('Nombre de pages', fontsize=12)
    ax.set_title('Distribution du CER Normalisé par page (sans accents, minuscules, sans ponctuation)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    cer_normalized_dist_path = os.path.join(images_dir, 'cer_normalized_distribution.png')
    plt.savefig(cer_normalized_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 2: Comparaison des variantes CER
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    variant_labels = {
        'base': 'Base',
        'no_accents': 'Sans accents',
        'lowercase': 'Minuscules',
        'normalized_chars': 'Chars normalisés',
        'no_punctuation': 'Sans ponctuation',
        'normalized': 'Normalisé complet'
    }
    
    # Calculer le CER de base
    total_edit = sum(m['total_edit_distance'] for m in all_metrics)
    total_chars = sum(m['total_gt_chars'] for m in all_metrics)
    base_cer = (total_edit / total_chars * 100) if total_chars > 0 else 100
    
    page_variants = {'base': base_cer}
    page_variants.update({k: v * 100 for k, v in page_cer_variants.items()})
    
    labels = [variant_labels.get(k, k) for k in page_variants.keys()]
    values = list(page_variants.values())
    
    bar_colors = [colors['primary'] if i == 0 else colors['info'] for i in range(len(values))]
    bars1 = ax1.bar(labels, values, color=bar_colors, edgecolor='white', alpha=0.8)
    ax1.set_ylabel('CER (%)', fontsize=12)
    ax1.set_title('CER par variante (niveau page)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Niveau document
    doc_total_edit = sum(d['total_edit_distance'] for d in document_stats.values())
    doc_total_chars = sum(d['total_gt_chars'] for d in document_stats.values())
    doc_base_cer = (doc_total_edit / doc_total_chars * 100) if doc_total_chars > 0 else 100
    
    doc_variants = {'base': doc_base_cer}
    doc_variants.update({k: v * 100 for k, v in doc_cer_variants.items()})
    
    values_doc = list(doc_variants.values())
    bars2 = ax2.bar(labels, values_doc, color=bar_colors, edgecolor='white', alpha=0.8)
    ax2.set_ylabel('CER (%)', fontsize=12)
    ax2.set_title('CER par variante (niveau document)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, values_doc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    cer_variants_path = os.path.join(images_dir, 'cer_variants.png')
    plt.savefig(cer_variants_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 3: Métriques de performance (Radar chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    metrics_names = ['Précision', 'Recall', 'IoU', 'Format Score', 'Accuracy (1-CER)']
    
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_format = np.mean([m['format_score'] for m in all_metrics])
    avg_accuracy = 1 - (total_edit / total_chars if total_chars > 0 else 1)
    
    values_radar = [avg_precision, avg_recall, avg_iou, avg_format, avg_accuracy]
    
    # Nombre d'axes
    N = len(metrics_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le polygone
    values_radar += values_radar[:1]
    
    ax.plot(angles, values_radar, 'o-', linewidth=2, color=colors['primary'], markersize=8)
    ax.fill(angles, values_radar, alpha=0.25, color=colors['primary'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Métriques de Performance Globales', fontsize=14, fontweight='bold', pad=20)
    
    # Ajouter les valeurs
    for angle, val in zip(angles[:-1], values_radar[:-1]):
        ax.annotate(f'{val:.2f}', xy=(angle, val), xytext=(angle, val + 0.08),
                   ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    radar_path = os.path.join(images_dir, 'performance_radar.png')
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 4: Top 10 documents les plus difficiles
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'], reverse=True)
    top_10_docs = sorted_docs[:10]
    
    doc_names = [name[:25] + '...' if len(name) > 25 else name for name, _ in top_10_docs]
    doc_cers = [stats['avg_cer'] * 100 for _, stats in top_10_docs]
    doc_pages = [stats['num_pages'] for _, stats in top_10_docs]
    
    bars = ax.barh(doc_names, doc_cers, color=colors['danger'], alpha=0.8, edgecolor='white')
    
    # Ajouter le nombre de pages comme annotation
    for bar, pages in zip(bars, doc_pages):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{pages} pages', va='center', fontsize=9, color=colors['gray'])
    
    ax.set_xlabel('CER (%)', fontsize=12)
    ax.set_title('Top 10 Documents les Plus Difficiles', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    worst_docs_path = os.path.join(images_dir, 'worst_documents.png')
    plt.savefig(worst_docs_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 5: CER moyen par document (tous les documents)
    # =========================================================================
    # Trier les documents par CER pour une meilleure lisibilité
    sorted_all_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'])
    
    # Calculer la hauteur dynamique en fonction du nombre de documents
    num_docs = len(sorted_all_docs)
    fig_height = max(8, num_docs * 0.3)  # 0.3 inch par document, minimum 8
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    all_doc_names = [name[:30] + '...' if len(name) > 30 else name for name, _ in sorted_all_docs]
    all_doc_cers = [stats['avg_cer'] * 100 for _, stats in sorted_all_docs]
    
    # Créer un gradient de couleurs du vert au rouge selon le CER
    norm = plt.Normalize(min(all_doc_cers), max(all_doc_cers))
    cmap = plt.cm.RdYlGn_r  # Rouge pour mauvais, vert pour bon
    bar_colors = [cmap(norm(cer)) for cer in all_doc_cers]
    
    bars = ax.barh(all_doc_names, all_doc_cers, color=bar_colors, alpha=0.85, edgecolor='white', height=0.7)
    
    # Ajouter une ligne verticale pour la moyenne
    mean_cer = np.mean(all_doc_cers)
    ax.axvline(mean_cer, color=colors['primary'], linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_cer:.1f}%')
    
    ax.set_xlabel('CER (%)', fontsize=12)
    ax.set_ylabel('Document', fontsize=12)
    ax.set_title(f'CER Moyen par Document ({num_docs} documents)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Ajuster les marges pour les noms longs
    plt.tight_layout()
    all_docs_cer_path = os.path.join(images_dir, 'cer_by_document.png')
    plt.savefig(all_docs_cer_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 6: Analyse des erreurs
    # =========================================================================
    if error_stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart des types d'erreurs
        totals = error_stats['totals']
        error_types = ['Insertions', 'Suppressions', 'Substitutions']
        error_values = [totals['insertions'], totals['deletions'], totals['substitutions']]
        error_colors = [colors['info'], colors['warning'], colors['danger']]
        
        wedges, texts, autotexts = ax1.pie(
            error_values, labels=error_types, colors=error_colors,
            autopct='%1.1f%%', startangle=90, explode=(0.02, 0.02, 0.02)
        )
        ax1.set_title('Répartition des Types d\'Erreurs', fontsize=12, fontweight='bold')
        
        # Bar chart des statistiques de lignes
        line_stats = error_stats['line_stats']
        line_categories = ['Lignes\nmanquantes', 'Lignes\nextras', 'Lignes\nfusionnées']
        line_values = [line_stats['missing_lines'], line_stats['extra_lines'], line_stats['merged_lines']]
        
        bars = ax2.bar(line_categories, line_values, 
                      color=[colors['warning'], colors['info'], colors['secondary']], 
                      edgecolor='white', alpha=0.8)
        ax2.set_ylabel('Nombre', fontsize=12)
        ax2.set_title('Statistiques des Lignes', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, line_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        errors_path = os.path.join(images_dir, 'error_analysis.png')
        plt.savefig(errors_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        errors_path = None
    
    # =========================================================================
    # Graphe 6: CER vs Nombre de lignes (scatter)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_lines = [m['num_ground_truth'] for m in all_metrics]
    cers = [m['cer'] * 100 for m in all_metrics]
    
    scatter = ax.scatter(num_lines, cers, alpha=0.6, c=cers, cmap='RdYlGn_r', 
                        edgecolors='white', s=50)
    
    # Ligne de tendance
    z = np.polyfit(num_lines, cers, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(num_lines), max(num_lines), 100)
    ax.plot(x_trend, p(x_trend), '--', color=colors['gray'], alpha=0.8, 
           label=f'Tendance linéaire')
    
    ax.set_xlabel('Nombre de lignes par page', fontsize=12)
    ax.set_ylabel('CER (%)', fontsize=12)
    ax.set_title('CER en fonction de la Complexité de la Page', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='CER (%)')
    plt.tight_layout()
    complexity_path = os.path.join(images_dir, 'cer_vs_complexity.png')
    plt.savefig(complexity_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 8: CER vs Nombre de pages d'entraînement par document
    # =========================================================================
    cer_vs_training_path = None
    # Préparer les données: pour chaque document testé, obtenir le CER et le nb de pages d'entraînement
    docs_with_training = []
    for doc_name, stats in document_stats.items():
        train_pages = training_pages_per_doc.get(doc_name, 0) if training_pages_per_doc else 0
        docs_with_training.append({
            'name': doc_name,
            'cer': stats['avg_cer'] * 100,
            'train_pages': train_pages
        })
    
    if docs_with_training and training_pages_per_doc:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_pages_list = [d['train_pages'] for d in docs_with_training]
        cer_list = [d['cer'] for d in docs_with_training]
        
        # Scatter plot avec couleur selon le CER
        scatter = ax.scatter(train_pages_list, cer_list, alpha=0.7, c=cer_list, 
                            cmap='RdYlGn_r', edgecolors='white', s=80)
        
        # Ligne de tendance si assez de points
        if len(train_pages_list) > 2:
            z = np.polyfit(train_pages_list, cer_list, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(train_pages_list), max(train_pages_list), 100)
            ax.plot(x_trend, p(x_trend), '--', color=colors['primary'], linewidth=2,
                   label=f'Tendance (pente: {z[0]:.2f})')
            
            # Calculer la corrélation
            correlation = np.corrcoef(train_pages_list, cer_list)[0, 1]
            ax.text(0.02, 0.98, f'Corrélation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Nombre de pages d\'entraînement', fontsize=12)
        ax.set_ylabel('CER moyen (%)', fontsize=12)
        ax.set_title('CER par Document vs Données d\'Entraînement', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.colorbar(scatter, ax=ax, label='CER (%)')
        plt.tight_layout()
        cer_vs_training_path = os.path.join(images_dir, 'cer_vs_training.png')
        plt.savefig(cer_vs_training_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # Génération du rapport Markdown
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculer les statistiques clés
    num_pages = len(all_metrics)
    num_documents = len(document_stats)
    total_lines_gt = sum(m['num_ground_truth'] for m in all_metrics)
    total_lines_pred = sum(m['num_predicted'] for m in all_metrics)
    total_matched = sum(m['num_matched'] for m in all_metrics)
    
    # Quartiles CER
    cer_array = np.array([m['cer'] * 100 for m in all_metrics])
    q1, q2, q3 = np.percentile(cer_array, [25, 50, 75])
    
    markdown_content = f"""# 📊 Rapport d'Évaluation du Modèle OCR

**Date de génération:** {timestamp}

---

## 📈 Résumé Exécutif

| Métrique | Valeur |
|----------|--------|
| **CER moyen** | {base_cer:.2f}% |
| **CER normalisé** | {page_cer_variants.get('normalized', 0)*100:.2f}% |
| **Précision** | {avg_precision*100:.1f}% |
| **Recall** | {avg_recall*100:.1f}% |
| **IoU moyen** | {avg_iou*100:.1f}% |
| **Score format** | {avg_format*100:.1f}% |
| **Pages évaluées** | {num_pages} |
| **Documents** | {num_documents} |

---

## 📉 Distribution du CER

Le graphe ci-dessous montre la distribution du Character Error Rate (CER) sur l'ensemble des pages évaluées.

![Distribution du CER]({os.path.relpath(cer_dist_path, output_dir)})

### Statistiques de distribution

| Statistique | Valeur |
|-------------|--------|
| Moyenne | {np.mean(cer_array):.2f}% |
| Médiane | {np.median(cer_array):.2f}% |
| Écart-type | {np.std(cer_array):.2f}% |
| Min | {np.min(cer_array):.2f}% |
| Max | {np.max(cer_array):.2f}% |
| Q1 (25%) | {q1:.2f}% |
| Q3 (75%) | {q3:.2f}% |

---

## � Distribution du CER Normalisé

Le graphe ci-dessous montre la distribution du CER après normalisation du texte (sans accents, minuscules, sans ponctuation).
Cette métrique montre la qualité de la transcription indépendamment des erreurs de casse et de ponctuation.

![Distribution du CER Normalisé]({os.path.relpath(cer_normalized_dist_path, output_dir)})

### Statistiques de distribution (CER normalisé)

| Statistique | Valeur |
|-------------|--------|
| Moyenne | {np.mean(cer_normalized_values):.2f}% |
| Médiane | {np.median(cer_normalized_values):.2f}% |
| Écart-type | {np.std(cer_normalized_values):.2f}% |
| Min | {np.min(cer_normalized_values):.2f}% |
| Max | {np.max(cer_normalized_values):.2f}% |
| Gain moyen vs CER base | {np.mean(cer_array) - np.mean(cer_normalized_values):.2f}pp |

---

## �🔄 Impact des Normalisations

Ce graphe compare le CER selon différentes stratégies de normalisation, permettant d'identifier si les erreurs sont principalement dues à la casse, aux accents ou à la ponctuation.

![Comparaison des variantes CER]({os.path.relpath(cer_variants_path, output_dir)})

### Interprétation des variantes

| Variante | Page CER | Document CER | Impact |
|----------|----------|--------------|--------|
| Base (original) | {base_cer:.2f}% | {doc_base_cer:.2f}% | Référence |
| Sans accents | {page_cer_variants.get('no_accents', 0)*100:.2f}% | {doc_cer_variants.get('no_accents', 0)*100:.2f}% | {(base_cer - page_cer_variants.get('no_accents', 0)*100):.2f}pp gain |
| Minuscules | {page_cer_variants.get('lowercase', 0)*100:.2f}% | {doc_cer_variants.get('lowercase', 0)*100:.2f}% | {(base_cer - page_cer_variants.get('lowercase', 0)*100):.2f}pp gain |
| Sans ponctuation | {page_cer_variants.get('no_punctuation', 0)*100:.2f}% | {doc_cer_variants.get('no_punctuation', 0)*100:.2f}% | {(base_cer - page_cer_variants.get('no_punctuation', 0)*100):.2f}pp gain |
| Normalisé complet | {page_cer_variants.get('normalized', 0)*100:.2f}% | {doc_cer_variants.get('normalized', 0)*100:.2f}% | {(base_cer - page_cer_variants.get('normalized', 0)*100):.2f}pp gain |

---

## 🎯 Performance Globale

Ce radar chart présente une vue d'ensemble des métriques de performance du modèle.

![Métriques de performance]({os.path.relpath(radar_path, output_dir)})

### Détail des métriques

| Métrique | Score | Description |
|----------|-------|-------------|
| **Précision** | {avg_precision:.3f} | % de lignes prédites correctement associées |
| **Recall** | {avg_recall:.3f} | % de lignes ground truth détectées |
| **IoU** | {avg_iou:.3f} | Intersection over Union des bounding boxes |
| **Format** | {avg_format:.3f} | Respect du format TSV attendu |
| **Accuracy** | {avg_accuracy:.3f} | 1 - CER (taux de caractères corrects) |

---

## ⚠️ Documents les Plus Difficiles

Les 10 documents avec le CER le plus élevé représentent les cas les plus difficiles pour le modèle.

![Documents les plus difficiles]({os.path.relpath(worst_docs_path, output_dir)})

### Détail des 10 pires documents

| Document | CER | IoU | Recall | Pages |
|----------|-----|-----|--------|-------|
"""
    
    # Ajouter le détail des 10 pires documents
    for name, stats in top_10_docs:
        short_name = name[:35] + '...' if len(name) > 35 else name
        markdown_content += f"| {short_name} | {stats['avg_cer']*100:.1f}% | {stats['avg_iou']:.2f} | {stats['avg_recall']:.2f} | {stats['num_pages']} |\n"
    
    markdown_content += f"""

---

## 📊 CER par Document

Vue d'ensemble du CER moyen pour chaque document, trié du meilleur au pire. Le gradient de couleur permet d'identifier rapidement les documents problématiques.

![CER par Document]({os.path.relpath(all_docs_cer_path, output_dir)})

---

## 📊 CER vs Complexité

Ce graphe montre la relation entre le nombre de lignes par page et le taux d'erreur.

![CER vs Complexité]({os.path.relpath(complexity_path, output_dir)})

### Observations sur la complexité

| Métrique | Valeur |
|----------|--------|
| Moyenne lignes/page | {np.mean(num_lines):.1f} |
| Corrélation CER/lignes | {np.corrcoef(num_lines, cers)[0,1]:.3f} |
| Lignes GT total | {total_lines_gt} |
| Lignes prédites total | {total_lines_pred} |
| Lignes matchées | {total_matched} |

"""
    
    # Ajouter la section CER vs Training si le graphe a été généré
    if cer_vs_training_path:
        # Calculer les stats pour le tableau
        train_pages_list = [d['train_pages'] for d in docs_with_training]
        cer_list = [d['cer'] for d in docs_with_training]
        correlation = np.corrcoef(train_pages_list, cer_list)[0, 1] if len(train_pages_list) > 2 else 0
        
        docs_with_zero_training = sum(1 for t in train_pages_list if t == 0)
        
        markdown_content += f"""---

## 📈 CER vs Pages d'Entraînement

Ce graphe montre la relation entre le nombre de pages d'entraînement par document et le CER obtenu en test.

![CER vs Training]({os.path.relpath(cer_vs_training_path, output_dir)})

### Statistiques d'entraînement

| Métrique | Valeur |
|----------|--------|
| Total pages d'entraînement | {sum(train_pages_list)} |
| Moyenne pages/document | {np.mean(train_pages_list):.1f} |
| Documents sans entraînement | {docs_with_zero_training} |
| Corrélation CER/entraînement | {correlation:.3f} |

"""
    
    # Ajouter la section analyse des erreurs si disponible
    if error_stats and errors_path:
        totals = error_stats['totals']
        rates = error_stats['rates']
        line_stats = error_stats['line_stats']
        
        markdown_content += f"""---

## 🔍 Analyse des Erreurs

### Répartition des types d'erreurs

![Analyse des erreurs]({os.path.relpath(errors_path, output_dir)})

### Statistiques détaillées

| Type d'erreur | Nombre | Pourcentage |
|---------------|--------|-------------|
| Insertions | {totals['insertions']} | {rates['insertion_rate']*100:.1f}% |
| Suppressions | {totals['deletions']} | {rates['deletion_rate']*100:.1f}% |
| Substitutions | {totals['substitutions']} | {rates['substitution_rate']*100:.1f}% |
| **Total erreurs** | **{totals['total_errors']}** | **100%** |

### Problèmes de détection de lignes

| Problème | Nombre |
|----------|--------|
| Lignes manquantes | {line_stats['missing_lines']} |
| Lignes en trop | {line_stats['extra_lines']} |
| Lignes fusionnées | {line_stats['merged_lines']} |

"""
        
        # Top confusions si disponibles
        if error_stats.get('top_confusions'):
            markdown_content += """### Top 10 Confusions de Caractères

| Ground Truth | Prédiction | Occurrences |
|--------------|------------|-------------|
"""
            for conf in error_stats['top_confusions'][:10]:
                gt_char = conf['gt'] if conf['gt'] else '∅'
                pred_char = conf['pred'] if conf['pred'] else '∅'
                markdown_content += f"| `{gt_char}` | `{pred_char}` | {conf['count']} |\n"
    
    # Section conclusion et limites
    markdown_content += f"""

---

## 📋 Limites et Points d'Attention

### Forces du modèle
"""
    
    # Identifier les forces
    if avg_recall > 0.9:
        markdown_content += "- ✅ **Excellent recall** : le modèle détecte la quasi-totalité des lignes\n"
    elif avg_recall > 0.8:
        markdown_content += "- ✅ **Bon recall** : la majorité des lignes sont détectées\n"
    
    if avg_precision > 0.9:
        markdown_content += "- ✅ **Excellente précision** : peu de faux positifs\n"
    elif avg_precision > 0.8:
        markdown_content += "- ✅ **Bonne précision** : relativement peu de prédictions incorrectes\n"
    
    if base_cer < 5:
        markdown_content += "- ✅ **CER très faible** : transcription de haute qualité\n"
    elif base_cer < 10:
        markdown_content += "- ✅ **CER acceptable** : quelques erreurs mais globalement fiable\n"
    
    markdown_content += """
### Limites identifiées
"""
    
    # Identifier les limites
    if avg_recall < 0.8:
        markdown_content += f"- ⚠️ **Recall limité ({avg_recall*100:.1f}%)** : certaines lignes ne sont pas détectées\n"
    
    if avg_precision < 0.8:
        markdown_content += f"- ⚠️ **Précision à améliorer ({avg_precision*100:.1f}%)** : présence de faux positifs\n"
    
    if base_cer > 10:
        markdown_content += f"- ⚠️ **CER élevé ({base_cer:.1f}%)** : la transcription contient des erreurs significatives\n"
    
    if error_stats:
        line_stats = error_stats['line_stats']
        if line_stats['merged_lines'] > 0:
            markdown_content += f"- ⚠️ **Fusions de lignes ({line_stats['merged_lines']})** : le modèle fusionne parfois plusieurs lignes\n"
        if line_stats['missing_lines'] > num_pages * 0.1:
            markdown_content += f"- ⚠️ **Lignes manquantes ({line_stats['missing_lines']})** : certaines lignes ne sont pas transcrites\n"
    
    # Variabilité
    if np.std(cer_array) > 10:
        markdown_content += f"- ⚠️ **Grande variabilité (σ={np.std(cer_array):.1f}%)** : performances inconsistantes selon les pages\n"
    
    markdown_content += f"""

### Recommandations

1. **Documents prioritaires** : Les 10 documents les plus difficiles devraient être analysés pour comprendre les échecs du modèle
2. **Normalisation** : Le CER normalisé complet ({page_cer_variants.get('normalized', 0)*100:.2f}%) montre le potentiel si les erreurs de casse/accents sont corrigées
3. **Post-traitement** : Un gain de {(base_cer - page_cer_variants.get('normalized', 0)*100):.2f}pp est possible avec des corrections automatiques

---

*Rapport généré automatiquement par evaluate_model_vllm_full.py*
"""
    
    # Sauvegarder le rapport
    report_path = os.path.join(output_dir, f"{report_name}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Rapport Markdown généré: {report_path}")
    print(f"Images sauvegardées dans: {images_dir}/")
    
    return report_path


async def run_evaluation_async(api_url: str,
                               test_json_path: str,
                               system_prompt: str,
                               user_prompt: str,
                               max_concurrent: int = 10,
                               output_dir: str = "labelme_outputs"):
    """
    Exécute l'évaluation en envoyant des requêtes asynchrones à l'API VLLM
    
    Args:
        api_url: URL du serveur VLLM (ex: http://localhost:8000)
        test_json_path: Chemin vers le fichier JSON de test
        system_prompt: Prompt système définissant le rôle
        user_prompt: Prompt utilisateur pour la tâche
        max_concurrent: Nombre maximum de requêtes simultanées
        output_dir: Dossier pour sauvegarder les fichiers LabelMe (None pour désactiver)
    """
    
    print("Chargement des données de test...")
    test_data = load_test_data(test_json_path)
    print(f"Nombre d'échantillons de test: {len(test_data)}")
    
    # Charger les données d'entraînement pour l'analyse CER vs training
    print("Chargement des données d'entraînement...")
    train_data = load_training_data(test_json_path)
    training_pages_per_doc = count_training_pages_per_document(train_data)
    print(f"Nombre d'échantillons d'entraînement: {len(train_data)}")
    print(f"Documents avec données d'entraînement: {len(training_pages_per_doc)}")
    
    # Créer le dossier de sortie si nécessaire
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Fichiers LabelMe seront sauvegardés dans: {output_dir}")
    
    # Créer un sémaphore pour limiter le nombre de requêtes simultanées
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"\nEnvoi de {len(test_data)} requêtes asynchrones (max {max_concurrent} simultanées)...")
    
    # Créer une session HTTP partagée
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Créer toutes les tâches
        tasks = [
            process_sample(session, sample, api_url, system_prompt, user_prompt, semaphore, output_dir)
            for sample in test_data
        ]
        
        # Exécuter toutes les tâches avec une barre de progression
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await coro
            results.append(result)
    
    print("\nCalcul des métriques et analyse des erreurs...")
    all_metrics = []
    all_page_analyses = []
    
    for result in results:
        metrics, page_analysis = evaluate_sample(
            result['predicted'], 
            result['ground_truth'],
            page_name=result['name']
        )
        metrics['name'] = result['name']
        all_metrics.append(metrics)
        if page_analysis:
            all_page_analyses.append(page_analysis)
    
    # Agrégation des erreurs
    print("Agrégation des statistiques d'erreurs...")
    error_stats = aggregate_error_stats(all_page_analyses) if all_page_analyses else None
    
    # Calcul des moyennes par page - CER pondéré par caractères
    total_edit_dist_all = sum(m['total_edit_distance'] for m in all_metrics)
    total_chars_all = sum(m['total_gt_chars'] for m in all_metrics)
    avg_cer = total_edit_dist_all / total_chars_all if total_chars_all > 0 else 1.0
    
    # Calculer les CER variants globaux (niveau page)
    page_cer_variants = {}
    for variant_name in NORMALIZATIONS.keys():
        edit_dist_key = f'total_edit_distance_{variant_name}'
        chars_key = f'total_gt_chars_{variant_name}'
        variant_edit_dist = sum(m[edit_dist_key] for m in all_metrics)
        variant_chars = sum(m[chars_key] for m in all_metrics)
        page_cer_variants[variant_name] = variant_edit_dist / variant_chars if variant_chars > 0 else 1.0
    
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_format = np.mean([m['format_score'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    
    # Calcul des statistiques par document
    document_stats = compute_per_document_stats(all_metrics)
    num_documents = len(document_stats)
    
    # Moyennes au niveau document - CER pondéré par caractères
    doc_total_edit_dist = sum(d['total_edit_distance'] for d in document_stats.values())
    doc_total_chars = sum(d['total_gt_chars'] for d in document_stats.values())
    doc_avg_cer = doc_total_edit_dist / doc_total_chars if doc_total_chars > 0 else 1.0
    
    # Calculer les CER variants globaux (niveau document)
    doc_cer_variants = {}
    for variant_name in NORMALIZATIONS.keys():
        edit_dist_key = f'total_edit_distance_{variant_name}'
        chars_key = f'total_gt_chars_{variant_name}'
        variant_edit_dist = sum(d[edit_dist_key] for d in document_stats.values())
        variant_chars = sum(d[chars_key] for d in document_stats.values())
        doc_cer_variants[variant_name] = variant_edit_dist / variant_chars if variant_chars > 0 else 1.0
    
    doc_avg_iou = np.mean([d['avg_iou'] for d in document_stats.values()])
    doc_avg_format = np.mean([d['avg_format_score'] for d in document_stats.values()])
    doc_avg_recall = np.mean([d['avg_recall'] for d in document_stats.values()])
    doc_avg_precision = np.mean([d['avg_precision'] for d in document_stats.values()])
    
    print("\n" + "="*70)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*70)
    print(f"\n--- STATISTIQUES PAR PAGE ({len(all_metrics)} pages) ---")
    print(f"CER moyen (base):       {avg_cer:.4f}")
    print(f"  ├─ sans accents:      {page_cer_variants['no_accents']:.4f}")
    print(f"  ├─ minuscules:        {page_cer_variants['lowercase']:.4f}")
    print(f"  ├─ chars normalisés:  {page_cer_variants['normalized_chars']:.4f}")
    print(f"  ├─ sans ponctuation:  {page_cer_variants['no_punctuation']:.4f}")
    print(f"  └─ normalisé complet: {page_cer_variants['normalized']:.4f}")
    print(f"IoU orienté moyen:      {avg_iou:.4f}")
    print(f"Score de formatage:     {avg_format:.4f}")
    print(f"Recall moyen:           {avg_recall:.4f}")
    print(f"Precision moyenne:      {avg_precision:.4f}")
    
    print(f"\n--- STATISTIQUES PAR DOCUMENT ({num_documents} documents) ---")
    print(f"CER moyen (base):       {doc_avg_cer:.4f}")
    print(f"  ├─ sans accents:      {doc_cer_variants['no_accents']:.4f}")
    print(f"  ├─ minuscules:        {doc_cer_variants['lowercase']:.4f}")
    print(f"  ├─ chars normalisés:  {doc_cer_variants['normalized_chars']:.4f}")
    print(f"  ├─ sans ponctuation:  {doc_cer_variants['no_punctuation']:.4f}")
    print(f"  └─ normalisé complet: {doc_cer_variants['normalized']:.4f}")
    print(f"IoU orienté moyen:      {doc_avg_iou:.4f}")
    print(f"Score de formatage:     {doc_avg_format:.4f}")
    print(f"Recall moyen:           {doc_avg_recall:.4f}")
    print(f"Precision moyenne:      {doc_avg_precision:.4f}")
    
    # Afficher le détail par document
    print(f"\n--- DÉTAIL PAR DOCUMENT ---")
    # Trier par CER décroissant pour voir les pires documents en premier
    sorted_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'], reverse=True)
    for doc_name, stats in sorted_docs[:10]:  # Afficher les 10 premiers
        print(f"  {doc_name}: CER={stats['avg_cer']:.4f}, IoU={stats['avg_iou']:.4f}, "
              f"Recall={stats['avg_recall']:.4f} ({stats['num_pages']} pages)")
    if len(sorted_docs) > 10:
        print(f"  ... et {len(sorted_docs) - 10} autres documents")
    
    # Afficher un résumé de l'analyse d'erreurs
    if error_stats:
        print(f"\n--- ANALYSE DES ERREURS ---")
        totals = error_stats['totals']
        rates = error_stats['rates']
        print(f"Total erreurs: {totals['total_errors']}")
        print(f"  Insertions:    {totals['insertions']} ({rates['insertion_rate']*100:.1f}%)")
        print(f"  Suppressions:  {totals['deletions']} ({rates['deletion_rate']*100:.1f}%)")
        print(f"  Substitutions: {totals['substitutions']} ({rates['substitution_rate']*100:.1f}%)")
        
        line_stats = error_stats['line_stats']
        print(f"\nLignes manquantes: {line_stats['missing_lines']}")
        print(f"Lignes extras:     {line_stats['extra_lines']}")
        print(f"Lignes fusionnées: {line_stats['merged_lines']}")
        
        if error_stats['top_confusions']:
            print(f"\nTop 5 confusions de caractères:")
            for conf in error_stats['top_confusions'][:5]:
                print(f"  '{conf['gt']}' → '{conf['pred']}': {conf['count']} fois")
    
    print("="*70)
    
    # Construire les dictionnaires de statistiques per_page et per_document pour le JSON
    per_page_summary = {
        'avg_cer': float(avg_cer),
        'avg_iou': float(avg_iou),
        'avg_format_score': float(avg_format),
        'avg_recall': float(avg_recall),
        'avg_precision': float(avg_precision),
        'num_pages': len(all_metrics)
    }
    # Ajouter les variantes CER
    for variant_name, variant_cer in page_cer_variants.items():
        per_page_summary[f'avg_cer_{variant_name}'] = float(variant_cer)
    
    per_document_summary = {
        'avg_cer': float(doc_avg_cer),
        'avg_iou': float(doc_avg_iou),
        'avg_format_score': float(doc_avg_format),
        'avg_recall': float(doc_avg_recall),
        'avg_precision': float(doc_avg_precision),
        'num_documents': num_documents
    }
    # Ajouter les variantes CER
    for variant_name, variant_cer in doc_cer_variants.items():
        per_document_summary[f'avg_cer_{variant_name}'] = float(variant_cer)
    
    # Sauvegarder les résultats détaillés
    output = {
        'summary': {
            'per_page': per_page_summary,
            'per_document': per_document_summary
        },
        'error_analysis': error_stats,
        'per_sample': all_metrics,
        'per_sample_error_details': [page_analysis_to_dict(pa) for pa in all_page_analyses],
        'per_document': document_stats,
        'predictions': results
    }
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\nRésultats détaillés sauvegardés dans 'evaluation_results.json'")
    
    # Générer le rapport Markdown avec graphes
    report_output_dir = output_dir if output_dir else "."
    report_path = generate_markdown_report(
        all_metrics=all_metrics,
        document_stats=document_stats,
        error_stats=error_stats,
        page_cer_variants=page_cer_variants,
        doc_cer_variants=doc_cer_variants,
        training_pages_per_doc=training_pages_per_doc,
        output_dir=report_output_dir,
        report_name="evaluation_report"
    )
    
    if output_dir:
        print(f"Fichiers LabelMe sauvegardés dans: {output_dir}/labelme/")
        print("  - *_pred.json : prédictions du modèle")
        print("  - *_gt.json   : vérités terrain")
    
    return output


def run_evaluation(api_url: str,
                  test_json_path: str,
                  system_prompt: str,
                  user_prompt: str,
                  max_concurrent: int = 10,
                  output_dir: str = "labelme_outputs"):
    """
    Wrapper synchrone pour l'évaluation asynchrone
    
    Args:
        api_url: URL du serveur VLLM
        test_json_path: Chemin vers les données de test
        system_prompt: Prompt système
        user_prompt: Prompt utilisateur
        max_concurrent: Nombre de requêtes simultanées
        output_dir: Dossier pour les fichiers LabelMe (None pour désactiver)
    """
    return asyncio.run(run_evaluation_async(
        api_url,
        test_json_path,
        system_prompt,
        user_prompt,
        max_concurrent,
        output_dir
    ))


if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"  # URL du serveur VLLM
    root_path = '/work/lead/ff379570/Lettres_En_Lumieres_tf/'
    TEST_JSON_PATH = "Data/Datasets/Generated/jsonraw.json"
    OUTPUT_DIR = "Data/labelme_outputs_generated"  
    # SYSTEM_PROMPT = Qwen2_5_SYSTEM_MESSAGE
    SYSTEM_PROMPT = Qwen2_5_OLD_SYSTEM_MESSAGE
    
    USER_PROMPT = PROMPTS["tsv"]
    
    # Lancer l'évaluation
    print("Assurez-vous que le serveur VLLM est lancé sur", API_URL)
    print("Pour lancer le serveur: ./start_vllm_server.sh\n")
    
    results = run_evaluation(
        api_url=API_URL,
        test_json_path=TEST_JSON_PATH,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        max_concurrent=256,  # Nombre de requêtes simultanées
        output_dir=OUTPUT_DIR  # Dossier pour sauvegarder les fichiers LabelMe
    )
    
    print(f"\n✓ Visualisation: Ouvrez les fichiers dans {OUTPUT_DIR}/ avec LabelMe")
    print("  Installation LabelMe: pip install labelme")
    print(f"  Commande: labelme {OUTPUT_DIR}/")