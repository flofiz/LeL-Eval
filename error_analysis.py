"""
Module d'analyse détaillée des erreurs pour l'évaluation HTR.

Ce module fournit des fonctions pour analyser en profondeur les erreurs
de transcription: types d'opérations, matrice de confusion, clustering, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import unicodedata


@dataclass
class EditOperation:
    """Représente une opération d'édition"""
    op_type: str  # 'insert', 'delete', 'substitute', 'match'
    gt_char: Optional[str] = None
    pred_char: Optional[str] = None
    gt_pos: Optional[int] = None
    pred_pos: Optional[int] = None


@dataclass
class LineErrorAnalysis:
    """Analyse des erreurs pour une ligne"""
    gt_text: str
    pred_text: str
    operations: List[EditOperation]
    
    # Comptages
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    matches: int = 0
    
    # Positions des erreurs (indices dans GT)
    error_positions: List[int] = field(default_factory=list)
    
    # Clusters d'erreurs (plages consécutives)
    error_clusters: List[Tuple[int, int]] = field(default_factory=list)
    
    # Paires de substitution (gt_char, pred_char)
    substitution_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class PageErrorAnalysis:
    """Analyse des erreurs pour une page"""
    page_name: str
    line_analyses: List[LineErrorAnalysis]
    
    # Lignes manquantes (GT non matchées)
    missing_lines: List[str] = field(default_factory=list)
    
    # Lignes supplémentaires (prédictions sans match)
    extra_lines: List[str] = field(default_factory=list)
    
    # Lignes fusionnées: (liste des lignes GT, ligne prédite)
    merged_lines: List[Tuple[List[str], str]] = field(default_factory=list)
    
    # Totaux agrégés
    total_insertions: int = 0
    total_deletions: int = 0
    total_substitutions: int = 0
    total_matches: int = 0
    total_gt_chars: int = 0


def categorize_character(char: str) -> str:
    """
    Classifie un caractère dans une catégorie.
    
    Returns:
        Une des catégories: 'lowercase', 'uppercase', 'digit', 
        'punctuation', 'accent', 'space', 'other'
    """
    if char == ' ':
        return 'space'
    
    if char.isdigit():
        return 'digit'
    
    if char.isalpha():
        # Vérifier si c'est un caractère accentué
        normalized = unicodedata.normalize('NFD', char)
        if len(normalized) > 1:  # Caractère décomposé = accentué
            return 'accent'
        if char.islower():
            return 'lowercase'
        return 'uppercase'
    
    # Ponctuation et symboles
    if unicodedata.category(char).startswith('P'):
        return 'punctuation'
    
    return 'other'


def compute_edit_operations(gt: str, pred: str) -> List[EditOperation]:
    """
    Calcule les opérations d'édition entre deux chaînes en utilisant
    l'algorithme de Wagner-Fischer avec traceback.
    
    Returns:
        Liste des opérations pour transformer gt en pred
    """
    m, n = len(gt), len(pred)
    
    # Matrice de distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialisation
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Remplissage
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    # Traceback pour récupérer les opérations
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == pred[j-1]:
            operations.append(EditOperation(
                op_type='match',
                gt_char=gt[i-1],
                pred_char=pred[j-1],
                gt_pos=i-1,
                pred_pos=j-1
            ))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            operations.append(EditOperation(
                op_type='substitute',
                gt_char=gt[i-1],
                pred_char=pred[j-1],
                gt_pos=i-1,
                pred_pos=j-1
            ))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion (caractère ajouté dans pred)
            operations.append(EditOperation(
                op_type='insert',
                gt_char=None,
                pred_char=pred[j-1],
                gt_pos=i if i > 0 else 0,
                pred_pos=j-1
            ))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (caractère manquant dans pred)
            operations.append(EditOperation(
                op_type='delete',
                gt_char=gt[i-1],
                pred_char=None,
                gt_pos=i-1,
                pred_pos=j if j > 0 else 0
            ))
            i -= 1
        else:
            # Fallback pour éviter boucle infinie
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1
    
    # Inverser pour avoir l'ordre correct
    operations.reverse()
    return operations


def find_error_clusters(error_positions: List[int], max_gap: int = 1) -> List[Tuple[int, int]]:
    """
    Trouve les clusters d'erreurs consécutives.
    
    Args:
        error_positions: Liste triée des positions d'erreurs
        max_gap: Écart maximum entre deux erreurs pour être dans le même cluster
        
    Returns:
        Liste de tuples (start, end) représentant les clusters
    """
    if not error_positions:
        return []
    
    clusters = []
    start = error_positions[0]
    end = error_positions[0]
    
    for pos in error_positions[1:]:
        if pos <= end + max_gap + 1:
            end = pos
        else:
            clusters.append((start, end))
            start = pos
            end = pos
    
    clusters.append((start, end))
    return clusters


def analyze_line_errors(gt_text: str, pred_text: str) -> LineErrorAnalysis:
    """
    Analyse détaillée des erreurs entre une ligne GT et une prédiction.
    """
    operations = compute_edit_operations(gt_text, pred_text)
    
    analysis = LineErrorAnalysis(
        gt_text=gt_text,
        pred_text=pred_text,
        operations=operations
    )
    
    error_positions = []
    
    for op in operations:
        if op.op_type == 'match':
            analysis.matches += 1
        elif op.op_type == 'insert':
            analysis.insertions += 1
            # L'insertion n'a pas de position GT directe, on note la position adjacente
            if op.gt_pos is not None:
                error_positions.append(op.gt_pos)
        elif op.op_type == 'delete':
            analysis.deletions += 1
            if op.gt_pos is not None:
                error_positions.append(op.gt_pos)
        elif op.op_type == 'substitute':
            analysis.substitutions += 1
            if op.gt_pos is not None:
                error_positions.append(op.gt_pos)
            if op.gt_char and op.pred_char:
                analysis.substitution_pairs.append((op.gt_char, op.pred_char))
    
    # Supprimer les doublons et trier
    analysis.error_positions = sorted(set(error_positions))
    
    # Trouver les clusters
    analysis.error_clusters = find_error_clusters(analysis.error_positions)
    
    return analysis


def detect_line_fusion(
    pred_text: str,
    unmatched_gt_texts: List[str],
    similarity_threshold: float = 0.7
) -> Optional[List[str]]:
    """
    Détecte si une prédiction est la fusion de plusieurs lignes GT.
    
    Args:
        pred_text: Texte de la prédiction
        unmatched_gt_texts: Textes GT non matchés
        similarity_threshold: Seuil de similarité pour considérer une inclusion
        
    Returns:
        Liste des textes GT fusionnés, ou None si pas de fusion détectée
    """
    if not unmatched_gt_texts or not pred_text:
        return None
    
    pred_lower = pred_text.lower().replace(' ', '')
    fused_gts = []
    
    for gt_text in unmatched_gt_texts:
        gt_lower = gt_text.lower().replace(' ', '')
        # Vérifier si le texte GT est inclus dans la prédiction
        if gt_lower and gt_lower in pred_lower:
            fused_gts.append(gt_text)
    
    # Une fusion nécessite au moins 2 lignes GT
    if len(fused_gts) >= 2:
        return fused_gts
    
    return None


def aggregate_error_stats(page_analyses: List[PageErrorAnalysis]) -> Dict:
    """
    Agrège les statistiques d'erreurs sur toutes les pages.
    
    Returns:
        Dictionnaire avec les statistiques globales
    """
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    total_matches = 0
    total_gt_chars = 0
    
    total_missing_lines = 0
    total_extra_lines = 0
    total_merged_lines = 0
    
    # Matrice de confusion: gt_char -> pred_char -> count
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Erreurs par catégorie de caractère
    errors_by_category = defaultdict(lambda: {'errors': 0, 'total': 0})
    
    # Statistiques de clustering
    all_cluster_sizes = []
    total_clustered_errors = 0
    total_errors_with_pos = 0
    
    for page in page_analyses:
        total_insertions += page.total_insertions
        total_deletions += page.total_deletions
        total_substitutions += page.total_substitutions
        total_matches += page.total_matches
        total_gt_chars += page.total_gt_chars
        
        total_missing_lines += len(page.missing_lines)
        total_extra_lines += len(page.extra_lines)
        total_merged_lines += len(page.merged_lines)
        
        for line_analysis in page.line_analyses:
            # Matrice de confusion
            for gt_char, pred_char in line_analysis.substitution_pairs:
                confusion_matrix[gt_char][pred_char] += 1
            
            # Erreurs par catégorie (basé sur les opérations)
            for op in line_analysis.operations:
                if op.gt_char:
                    cat = categorize_character(op.gt_char)
                    errors_by_category[cat]['total'] += 1
                    if op.op_type != 'match':
                        errors_by_category[cat]['errors'] += 1
            
            # Clustering
            for start, end in line_analysis.error_clusters:
                cluster_size = end - start + 1
                all_cluster_sizes.append(cluster_size)
                total_clustered_errors += cluster_size
            
            total_errors_with_pos += len(line_analysis.error_positions)
    
    # Calculer les totaux et ratios
    total_errors = total_insertions + total_deletions + total_substitutions
    
    # Top confusions (triées par fréquence)
    top_confusions = []
    for gt_char, pred_dict in confusion_matrix.items():
        for pred_char, count in pred_dict.items():
            top_confusions.append({
                'gt': gt_char,
                'pred': pred_char,
                'count': count
            })
    top_confusions.sort(key=lambda x: x['count'], reverse=True)
    
    # Convertir confusion_matrix en dict normal
    confusion_dict = {k: dict(v) for k, v in confusion_matrix.items()}
    
    # Erreurs par catégorie avec ratios
    category_stats = {}
    for cat, stats in errors_by_category.items():
        rate = stats['errors'] / stats['total'] if stats['total'] > 0 else 0
        category_stats[cat] = {
            'errors': stats['errors'],
            'total': stats['total'],
            'rate': round(rate, 4)
        }
    
    # Stats de clustering
    avg_cluster_size = sum(all_cluster_sizes) / len(all_cluster_sizes) if all_cluster_sizes else 0
    clustered_ratio = total_clustered_errors / total_errors_with_pos if total_errors_with_pos > 0 else 0
    
    return {
        'totals': {
            'insertions': total_insertions,
            'deletions': total_deletions,
            'substitutions': total_substitutions,
            'matches': total_matches,
            'total_errors': total_errors,
            'total_gt_chars': total_gt_chars
        },
        'rates': {
            'insertion_rate': round(total_insertions / total_errors, 4) if total_errors > 0 else 0,
            'deletion_rate': round(total_deletions / total_errors, 4) if total_errors > 0 else 0,
            'substitution_rate': round(total_substitutions / total_errors, 4) if total_errors > 0 else 0
        },
        'confusion_matrix': confusion_dict,
        'top_confusions': top_confusions[:30],  # Top 30
        'by_category': category_stats,
        'clustering': {
            'avg_cluster_size': round(avg_cluster_size, 2),
            'clustered_error_ratio': round(clustered_ratio, 4),
            'total_clusters': len(all_cluster_sizes)
        },
        'line_stats': {
            'missing_lines': total_missing_lines,
            'extra_lines': total_extra_lines,
            'merged_lines': total_merged_lines
        }
    }


def line_analysis_to_dict(analysis: LineErrorAnalysis) -> Dict:
    """Convertit une LineErrorAnalysis en dictionnaire pour JSON."""
    return {
        'gt_text': analysis.gt_text,
        'pred_text': analysis.pred_text,
        'insertions': analysis.insertions,
        'deletions': analysis.deletions,
        'substitutions': analysis.substitutions,
        'matches': analysis.matches,
        'error_positions': analysis.error_positions,
        'error_clusters': analysis.error_clusters,
        'substitution_pairs': analysis.substitution_pairs
    }


def page_analysis_to_dict(analysis: PageErrorAnalysis) -> Dict:
    """Convertit une PageErrorAnalysis en dictionnaire pour JSON."""
    return {
        'page_name': analysis.page_name,
        'lines': [line_analysis_to_dict(la) for la in analysis.line_analyses],
        'missing_lines': analysis.missing_lines,
        'extra_lines': analysis.extra_lines,
        'merged_lines': [(gts, pred) for gts, pred in analysis.merged_lines],
        'totals': {
            'insertions': analysis.total_insertions,
            'deletions': analysis.total_deletions,
            'substitutions': analysis.total_substitutions,
            'matches': analysis.total_matches,
            'total_gt_chars': analysis.total_gt_chars
        }
    }
