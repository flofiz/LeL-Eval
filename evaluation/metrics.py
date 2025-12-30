"""
Calculs de métriques d'évaluation pour HTR.

Ce module fournit les fonctions de calcul des métriques:
- CER (Character Error Rate)
- IoU (Intersection over Union)
- Score de formatage
- Statistiques par document
"""

import numpy as np
import editdistance
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from .models import BBox
from .normalization import NORMALIZATIONS
from .tsv_parser import parse_tsv_line

# Import depuis error_analysis (module existant)
import sys
import os
# Ajouter le répertoire parent au path pour importer error_analysis
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from error_analysis import (
    PageErrorAnalysis,
    analyze_line_errors,
    detect_line_fusion,
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


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Character Error Rate (CER).
    
    Args:
        reference: Texte de référence (ground truth)
        hypothesis: Texte prédit
        
    Returns:
        CER entre 0.0 et 1.0 (peut dépasser 1.0 si plus d'erreurs que de caractères)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def calculate_oriented_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calcule l'IoU pour des boîtes orientées.
    Simplifié: utilise l'intersection des rectangles axis-aligned.
    Pour une implémentation complète avec rotation, utiliser shapely.
    
    Args:
        bbox1: Première bounding box
        bbox2: Deuxième bounding box
        
    Returns:
        IoU entre 0.0 et 1.0
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
    
    Args:
        predicted_tsv: TSV prédit par le modèle
        ground_truth_tsv: TSV de référence
        
    Returns:
        Score entre 0 et 1
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


def match_predictions_to_ground_truth(
    predictions: List[BBox], 
    ground_truths: List[BBox],
    max_cer_threshold: float = 0.95
) -> Tuple[List[Tuple[BBox, BBox]], Dict]:
    """
    Apparie les prédictions aux vérités terrain en utilisant l'algorithme hongrois
    pour minimiser le CER total.
    
    Amélioration par rapport au matching greedy IoU:
    - Utilise une matrice de coûts basée sur le CER symétrique
    - Trouve le matching optimal global (algorithme hongrois)
    - Filtre les paires où intersection = 0
    - Applique un seuil de CER maximum pour éviter les matchs absurdes
    
    Args:
        predictions: Liste des bounding boxes prédites
        ground_truths: Liste des bounding boxes de référence
        max_cer_threshold: Seuil maximum de CER pour accepter un match (défaut: 0.95)
        
    Returns:
        Tuple (matched_pairs, matching_info) où:
        - matched_pairs: Liste de tuples (pred, gt) appariés
        - matching_info: Dict avec informations de débogage (matrice CER, IoU, etc.)
    """
    n_pred = len(predictions)
    n_gt = len(ground_truths)
    
    if n_pred == 0 or n_gt == 0:
        return [], {
            'n_predictions': n_pred,
            'n_ground_truths': n_gt,
            'cost_matrix': [],
            'iou_matrix': [],
            'matches': []
        }
    
    # 1. Calculer les matrices de coûts et IoU
    # Note: On utilise une grande valeur (1e10) au lieu de inf pour éviter les erreurs "infeasible"
    # de linear_sum_assignment. On filtrera les matchs invalides après.
    LARGE_COST = 1e10
    cost_matrix = np.full((n_pred, n_gt), LARGE_COST)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            # Calculer IoU pour filtrer les paires impossibles
            iou = calculate_oriented_iou(pred, gt)
            iou_matrix[i, j] = iou
            
            # Filtrer: intersection doit être > 0
            if iou > 0:
                # Calculer CER symétrique (normalisé par max des longueurs)
                edit_dist = editdistance.eval(gt.text, pred.text)
                max_len = max(len(gt.text), len(pred.text))
                
                if max_len > 0:
                    cer_symmetric = edit_dist / max_len
                    
                    # Appliquer le seuil de CER
                    if cer_symmetric <= max_cer_threshold:
                        cost_matrix[i, j] = cer_symmetric
    
    # 2. Appliquer l'algorithme hongrois (minimisation du coût total)
    # linear_sum_assignment trouve le matching optimal
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 3. Extraire les paires matchées valides (coût < LARGE_COST)
    matched_pairs = []
    match_details = []
    
    for pred_idx, gt_idx in zip(row_indices, col_indices):
        cost = cost_matrix[pred_idx, gt_idx]
        if cost < LARGE_COST:  # Match valide
            matched_pairs.append((predictions[pred_idx], ground_truths[gt_idx]))
            match_details.append({
                'pred_idx': int(pred_idx),
                'gt_idx': int(gt_idx),
                'pred_text': predictions[pred_idx].text,
                'gt_text': ground_truths[gt_idx].text,
                'cer_symmetric': float(cost),
                'iou': float(iou_matrix[pred_idx, gt_idx])
            })
    
    # 4. Informations de débogage
    matching_info = {
        'n_predictions': n_pred,
        'n_ground_truths': n_gt,
        'n_matched': len(matched_pairs),
        'cost_matrix': cost_matrix.tolist(),
        'iou_matrix': iou_matrix.tolist(),
        'matches': match_details,
        'total_cost': sum(cost_matrix[i, j] for i, j in zip(row_indices, col_indices) if cost_matrix[i, j] < LARGE_COST)
    }
    
    return matched_pairs, matching_info


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
    
    Args:
        predicted_tsv: TSV prédit par le modèle
        ground_truth_tsv: TSV de référence
        page_name: Nom de la page pour l'analyse
        
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
    matched_pairs, matching_info = match_predictions_to_ground_truth(pred_bboxes, gt_bboxes)
    
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
        'matching_info': matching_info,  # Logs de débogage pour LabelMe JSON
    }
    
    # Ajouter les métriques pour chaque variante CER
    for variant_name in NORMALIZATIONS.keys():
        metrics[f'cer_{variant_name}'] = variant_cer[variant_name]
        metrics[f'total_edit_distance_{variant_name}'] = variant_total_errors[variant_name]
        metrics[f'total_gt_chars_{variant_name}'] = variant_total_chars[variant_name]
    
    return metrics, page_analysis


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
        
        # Calculs pour les métriques de segmentation
        total_matched = sum(m['num_matched'] for m in doc_metrics)
        total_predicted = sum(m['num_predicted'] for m in doc_metrics)
        total_ground_truth = sum(m['num_ground_truth'] for m in doc_metrics)
        
        doc_stat = {
            'avg_cer': float(weighted_cer),  # CER pondéré par caractères
            'total_gt_chars': total_chars,  # Pour pondération au niveau global
            'total_edit_distance': total_edit_dist,  # Pour recalcul de moyenne pondérée
            'avg_iou': float(np.mean([m['iou'] for m in doc_metrics])),
            'avg_format_score': float(np.mean([m['format_score'] for m in doc_metrics])),
            'avg_recall': float(np.mean([m['recall'] for m in doc_metrics])),
            'avg_precision': float(np.mean([m['precision'] for m in doc_metrics])),
            'num_pages': len(doc_metrics),
            'total_matched': total_matched,
            'total_predicted': total_predicted,
            'total_ground_truth': total_ground_truth,
            # Métriques de segmentation pour analyse CER vs segmentation
            'missing_lines': total_ground_truth - total_matched,
            'extra_lines': total_predicted - total_matched,
            'segmentation_error_rate': 1.0 - (total_matched / total_ground_truth) if total_ground_truth > 0 else 0.0,
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
