"""
Génération du rapport Markdown avec graphes.

Ce module fournit la fonction de génération du rapport d'évaluation
complet avec visualisations matplotlib.
"""

import os
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les images

from .normalization import NORMALIZATIONS


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
    # Graphe 1c: Distribution de l'IoU par page
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    iou_values = [m['iou'] * 100 for m in all_metrics]  # IoU en %
    
    ax.hist(iou_values, bins=30, color=colors['success'], alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(iou_values), color=colors['danger'], linestyle='--', 
               linewidth=2, label=f'Moyenne: {np.mean(iou_values):.1f}%')
    ax.axvline(np.median(iou_values), color=colors['primary'], linestyle='--', 
               linewidth=2, label=f'Médiane: {np.median(iou_values):.1f}%')
    
    ax.set_xlabel('IoU (%)', fontsize=12)
    ax.set_ylabel('Nombre de pages', fontsize=12)
    ax.set_title('Distribution de l\'IoU (Intersection over Union) par page', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    iou_dist_path = os.path.join(images_dir, 'iou_distribution.png')
    plt.savefig(iou_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Graphe 1d: IoU moyen par document (tous les documents)
    # =========================================================================
    sorted_docs_by_iou = sorted(document_stats.items(), key=lambda x: x[1]['avg_iou'], reverse=True)
    
    num_docs_iou = len(sorted_docs_by_iou)
    fig_height_iou = max(8, num_docs_iou * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height_iou))
    
    doc_names_iou = [name[:30] + '...' if len(name) > 30 else name for name, _ in sorted_docs_by_iou]
    doc_ious = [stats['avg_iou'] * 100 for _, stats in sorted_docs_by_iou]
    
    # Créer un gradient de couleurs du rouge au vert selon l'IoU (inversé par rapport au CER)
    norm_iou = plt.Normalize(min(doc_ious), max(doc_ious))
    cmap_iou = plt.cm.RdYlGn  # Vert pour bon (haut IoU), rouge pour mauvais
    bar_colors_iou = [cmap_iou(norm_iou(iou)) for iou in doc_ious]
    
    bars = ax.barh(doc_names_iou, doc_ious, color=bar_colors_iou, alpha=0.85, edgecolor='white', height=0.7)
    
    # Ajouter une ligne verticale pour la moyenne
    mean_iou = np.mean(doc_ious)
    ax.axvline(mean_iou, color=colors['primary'], linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_iou:.1f}%')
    
    ax.set_xlabel('IoU (%)', fontsize=12)
    ax.set_ylabel('Document', fontsize=12)
    ax.set_title(f'IoU Moyen par Document ({num_docs_iou} documents)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    iou_by_document_path = os.path.join(images_dir, 'iou_by_document.png')
    plt.savefig(iou_by_document_path, dpi=150, bbox_inches='tight')
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
    # Graphe 7: CER vs Nombre de lignes (scatter)
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
    # Graphe 9: CER vs Erreurs de Segmentation par document
    # Ce graphe permet de distinguer les problèmes de transcription des problèmes de segmentation
    # =========================================================================
    cer_vs_segmentation_path = None
    
    # Préparer les données de segmentation par document
    docs_segmentation_data = []
    for doc_name, stats in document_stats.items():
        seg_error_rate = stats.get('segmentation_error_rate', 0) * 100  # en %
        cer = stats['avg_cer'] * 100
        missing = stats.get('missing_lines', 0)
        extra = stats.get('extra_lines', 0)
        total_gt = stats['total_ground_truth']
        
        docs_segmentation_data.append({
            'name': doc_name,
            'cer': cer,
            'seg_error_rate': seg_error_rate,
            'missing_lines': missing,
            'extra_lines': extra,
            'total_gt': total_gt
        })
    
    if docs_segmentation_data and len(docs_segmentation_data) >= 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        seg_rates = [d['seg_error_rate'] for d in docs_segmentation_data]
        seg_doc_cers = [d['cer'] for d in docs_segmentation_data]
        sizes = [max(20, min(200, d['total_gt'] * 3)) for d in docs_segmentation_data]  # Taille proportionnelle au nb de lignes
        
        # Scatter plot avec couleur selon le CER
        scatter = ax.scatter(seg_rates, seg_doc_cers, alpha=0.7, c=seg_doc_cers, 
                            cmap='RdYlGn_r', edgecolors='white', s=sizes)
        
        # Calculer les médianes pour les quadrants
        median_seg = np.median(seg_rates)
        median_cer = np.median(seg_doc_cers)
        
        # Tracer les lignes de quadrants
        ax.axvline(median_seg, color=colors['gray'], linestyle=':', alpha=0.7)
        ax.axhline(median_cer, color=colors['gray'], linestyle=':', alpha=0.7)
        
        # Label les quadrants
        ax.text(0.02, 0.98, 'Bonne segmentation\nBonne transcription', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               color=colors['success'], fontweight='bold', alpha=0.8)
        ax.text(0.98, 0.98, 'Mauvaise segmentation\nBonne/Moyenne transcription', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
               color=colors['warning'], fontweight='bold', alpha=0.8)
        ax.text(0.02, 0.02, 'Bonne segmentation\nMauvaise transcription', 
               transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
               color=colors['info'], fontweight='bold', alpha=0.8)
        ax.text(0.98, 0.02, 'Mauvaise segmentation\nMauvaise transcription', 
               transform=ax.transAxes, fontsize=9, verticalalignment='bottom', ha='right',
               color=colors['danger'], fontweight='bold', alpha=0.8)
        
        # Ligne de tendance
        if len(seg_rates) > 2:
            z = np.polyfit(seg_rates, seg_doc_cers, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(seg_rates), max(seg_rates), 100)
            ax.plot(x_trend, p(x_trend), '--', color=colors['primary'], linewidth=2,
                   label=f'Tendance (pente: {z[0]:.2f})')
            
            # Calculer la corrélation
            correlation = np.corrcoef(seg_rates, seg_doc_cers)[0, 1]
            ax.text(0.5, 0.02, f'Corrélation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                   ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Taux d\'erreur de segmentation (%)\n(lignes non matchées / lignes GT)', fontsize=12)
        ax.set_ylabel('CER moyen (%)', fontsize=12)
        ax.set_title('CER vs Erreurs de Segmentation par Document\n(Taille des points = nombre de lignes GT)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        plt.colorbar(scatter, ax=ax, label='CER (%)')
        plt.tight_layout()
        cer_vs_segmentation_path = os.path.join(images_dir, 'cer_vs_segmentation.png')
        plt.savefig(cer_vs_segmentation_path, dpi=150, bbox_inches='tight')
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
    
    # Calculer micro-CER (pondéré par caractères - niveau page)
    total_edit_dist = sum(m['total_edit_distance'] for m in all_metrics)
    total_gt_chars = sum(m['total_gt_chars'] for m in all_metrics)
    micro_cer = (total_edit_dist / total_gt_chars * 100) if total_gt_chars > 0 else 100.0
    
    # Calculer macro-CER (moyenne des CER par document)
    doc_cers = [stats['avg_cer'] * 100 for stats in document_stats.values()]
    macro_cer = np.mean(doc_cers) if doc_cers else 100.0
    
    # Calculer le biais (écart entre micro et macro)
    cer_bias = micro_cer - macro_cer
    
    # Quartiles CER
    cer_array = np.array([m['cer'] * 100 for m in all_metrics])
    q1, q2, q3 = np.percentile(cer_array, [25, 50, 75])
    
    markdown_content = f"""# 📊 Rapport d'Évaluation du Modèle OCR

**Date de génération:** {timestamp}

---

## 📈 Résumé Exécutif

### Métriques de Performance (CER)

| Métrique | Valeur | Description |
|----------|--------|-------------|
| **Micro-CER** | {micro_cer:.2f}% | Pondéré par caractères (niveau page) |
| **Macro-CER** | {macro_cer:.2f}% | Moyenne par document (non biaisé) |
| **Biais** | {cer_bias:+.2f}pp | Écart micro-macro |
| **CER normalisé** | {page_cer_variants.get('normalized', 0)*100:.2f}% | Sans accents/casse/ponctuation |

> **Interprétation du biais:** {"⚠️ Les gros documents tirent le CER vers le haut" if cer_bias > 1 else ("⚠️ Les gros documents ont de meilleurs scores" if cer_bias < -1 else "✅ Corpus équilibré")}

### Autres Métriques

| Métrique | Valeur |
|----------|--------|
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

## 📉 Distribution du CER Normalisé

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

## 📐 Distribution de l'IoU par page

Le graphe ci-dessous montre la distribution de l'Intersection over Union (IoU) des bounding boxes sur l'ensemble des pages.
Un IoU élevé indique que les bounding boxes prédites correspondent bien aux bounding boxes de référence.

![Distribution de l'IoU]({os.path.relpath(iou_dist_path, output_dir)})

### Statistiques de distribution (IoU)

| Statistique | Valeur |
|-------------|--------|
| Moyenne | {np.mean(iou_values):.2f}% |
| Médiane | {np.median(iou_values):.2f}% |
| Écart-type | {np.std(iou_values):.2f}% |
| Min | {np.min(iou_values):.2f}% |
| Max | {np.max(iou_values):.2f}% |

---

## 📐 IoU par Document

Vue d'ensemble de l'IoU moyen pour chaque document. Le gradient de couleur permet d'identifier rapidement les documents avec de bons ou mauvais alignements.

![IoU par Document]({os.path.relpath(iou_by_document_path, output_dir)})

---

## 🔄 Impact des Normalisations

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
    
    # Ajouter la section CER vs Segmentation si le graphe a été généré
    if cer_vs_segmentation_path:
        seg_rates = [d['seg_error_rate'] for d in docs_segmentation_data]
        seg_cers = [d['cer'] for d in docs_segmentation_data]
        seg_correlation = np.corrcoef(seg_rates, seg_cers)[0, 1] if len(seg_rates) > 2 else 0
        
        # Compter les documents par quadrant
        median_seg = np.median(seg_rates)
        median_cer_seg = np.median(seg_cers)
        
        q1_count = sum(1 for s, c in zip(seg_rates, seg_cers) if s < median_seg and c < median_cer_seg)  # Bonne seg, bonne transcription
        q2_count = sum(1 for s, c in zip(seg_rates, seg_cers) if s >= median_seg and c < median_cer_seg)  # Mauvaise seg, bonne transcription
        q3_count = sum(1 for s, c in zip(seg_rates, seg_cers) if s < median_seg and c >= median_cer_seg)  # Bonne seg, mauvaise transcription
        q4_count = sum(1 for s, c in zip(seg_rates, seg_cers) if s >= median_seg and c >= median_cer_seg)  # Mauvaise seg, mauvaise transcription
        
        total_missing = sum(d['missing_lines'] for d in docs_segmentation_data)
        total_extra = sum(d['extra_lines'] for d in docs_segmentation_data)
        
        markdown_content += f"""---

## 🎯 CER vs Erreurs de Segmentation

Ce graphe permet d'identifier si les erreurs sont principalement dues à la **transcription** ou à la **segmentation** (lignes oubliées, fusionnées, ou splittées).

![CER vs Segmentation]({os.path.relpath(cer_vs_segmentation_path, output_dir)})

### Interprétation des quadrants

| Quadrant | Description | Documents |
|----------|-------------|-----------|
| 🟢 Bas-Gauche | Bonne segmentation + Bonne transcription | {q1_count} |
| 🟡 Bas-Droite | Mauvaise segmentation + Transcription OK | {q2_count} |
| 🔵 Haut-Gauche | Bonne segmentation + Mauvaise transcription | {q3_count} |
| 🔴 Haut-Droite | Mauvaise segmentation + Mauvaise transcription | {q4_count} |

### Statistiques de segmentation

| Métrique | Valeur |
|----------|--------|
| Corrélation Segmentation/CER | {seg_correlation:.3f} |
| Total lignes manquantes | {total_missing} |
| Total lignes en trop | {total_extra} |
| Médiane taux erreur seg. | {median_seg:.1f}% |
| Médiane CER | {median_cer_seg:.1f}% |

### Diagnostic

"""
        # Ajouter des diagnostics automatiques
        if seg_correlation > 0.5:
            markdown_content += f"- ⚠️ **Forte corrélation ({seg_correlation:.2f})** : Les erreurs de CER sont largement dues aux problèmes de segmentation\n"
        elif seg_correlation > 0.3:
            markdown_content += f"- ℹ️ **Corrélation modérée ({seg_correlation:.2f})** : La segmentation contribue partiellement au CER\n"
        else:
            markdown_content += f"- ✅ **Faible corrélation ({seg_correlation:.2f})** : Les erreurs de CER sont principalement dues à la transcription, pas à la segmentation\n"
        
        if q2_count > q3_count:
            markdown_content += f"- 📊 **{q2_count} documents** ont des problèmes de segmentation mais une transcription correcte → améliorer la détection de lignes\n"
        if q3_count > q2_count:
            markdown_content += f"- 📊 **{q3_count} documents** ont une bonne segmentation mais des erreurs de transcription → améliorer le modèle de transcription\n"
        if q4_count > 0:
            markdown_content += f"- 🔴 **{q4_count} documents critiques** combinent problèmes de segmentation ET transcription\n"

        markdown_content += "\n"
    
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
        
        # Top confusions normalisées (vraies erreurs hors casse/accents)
        if error_stats.get('top_confusions_normalized'):
            markdown_content += """
### Top 10 Confusions Normalisées

Ces erreurs persistent après normalisation (minuscules, sans accents). Ce sont les **vraies erreurs de transcription**, pas les erreurs de casse ou d'accentuation.

| Ground Truth | Prédiction | Occurrences |
|--------------|------------|-------------|
"""
            for conf in error_stats['top_confusions_normalized'][:10]:
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
