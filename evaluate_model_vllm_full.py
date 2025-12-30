"""
Script principal d'évaluation de modèles HTR via VLLM.

Ce script est l'orchestrateur qui utilise les modules du package `evaluation`
pour évaluer un modèle de transcription de texte manuscrit.

Usage:
    python evaluate_model_vllm_full.py

Configuration:
    Modifier les variables dans la section if __name__ == "__main__"
"""

import os
import json
import asyncio
import numpy as np
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

# Import des prompts
from Utils.promptes import Qwen2_5_SYSTEM_MESSAGE, PROMPTS, Qwen2_5_OLD_SYSTEM_MESSAGE

# Imports depuis le package evaluation
from evaluation import (
    # Normalization
    NORMALIZATIONS,
    # Metrics
    evaluate_sample,
    compute_per_document_stats,
    # API Client
    process_sample,
    # Report Generator
    generate_markdown_report,
    generate_html_report,
    # Data Loader
    load_test_data,
    load_training_data,
    count_training_pages_per_document,
)

# Import de l'analyse d'erreurs (module existant)
from error_analysis import aggregate_error_stats, page_analysis_to_dict


async def run_evaluation_async(api_url: str,
                               test_json_path: str,
                               system_prompt: str,
                               user_prompt: str,
                               max_concurrent: int = 10,
                               output_dir: str = "labelme_outputs"):
    """
    Exécute l'évaluation en envoyant des requêtes asynchrones à l'API VLLM.
    
    Args:
        api_url: URL du serveur VLLM (ex: http://localhost:8000)
        test_json_path: Chemin vers le fichier JSON de test
        system_prompt: Prompt système définissant le rôle
        user_prompt: Prompt utilisateur pour la tâche
        max_concurrent: Nombre maximum de requêtes simultanées
        output_dir: Dossier pour sauvegarder les fichiers LabelMe (None pour désactiver)
    
    Returns:
        Dictionnaire avec les résultats complets de l'évaluation
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
        metrics['perplexity'] = result.get('perplexity')  # Ajouter la perplexité
        all_metrics.append(metrics)
        if page_analysis:
            all_page_analyses.append(page_analysis)
    
    # Agrégation des erreurs
    print("Agrégation des statistiques d'erreurs...")
    error_stats = aggregate_error_stats(all_page_analyses) if all_page_analyses else None
    
    # =========================================================================
    # CALCUL DES MÉTRIQUES AVEC PONDÉRATION APPROPRIÉE
    # =========================================================================
    # 
    # NIVEAU PAGE (micro-average):
    #   CER = somme(edit_distance) / somme(gt_chars)
    #   Pondéré par le nombre de caractères pour éviter qu'une page courte
    #   biaise le résultat.
    #
    # NIVEAU DOCUMENT (micro-average par doc, puis macro-average sur corpus):
    #   CER_doc = somme(edit_distance_pages) / somme(gt_chars_pages)
    #   CER_corpus = moyenne(CER_doc pour chaque document)
    #   Chaque document a le même poids, évite le biais des documents
    #   sur-représentés (avec beaucoup de pages).
    # =========================================================================
    
    # --- NIVEAU PAGE (micro-average) ---
    # CER pondéré par caractères (une page avec plus de texte a plus de poids)
    total_edit_dist_all = sum(m['total_edit_distance'] for m in all_metrics)
    total_chars_all = sum(m['total_gt_chars'] for m in all_metrics)
    micro_avg_cer = total_edit_dist_all / total_chars_all if total_chars_all > 0 else 1.0
    
    # CER variants (micro-average niveau page)
    page_cer_variants = {}
    for variant_name in NORMALIZATIONS.keys():
        edit_dist_key = f'total_edit_distance_{variant_name}'
        chars_key = f'total_gt_chars_{variant_name}'
        variant_edit_dist = sum(m[edit_dist_key] for m in all_metrics)
        variant_chars = sum(m[chars_key] for m in all_metrics)
        page_cer_variants[variant_name] = variant_edit_dist / variant_chars if variant_chars > 0 else 1.0
    
    # Autres métriques page (moyenne simple ok car déjà normalisées entre 0-1)
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_format = np.mean([m['format_score'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    
    # --- NIVEAU DOCUMENT ---
    document_stats = compute_per_document_stats(all_metrics)
    num_documents = len(document_stats)
    
    # CER corpus (macro-average): moyenne des CER par document
    # Chaque document compte égal, peu importe sa taille
    doc_cers = [d['avg_cer'] for d in document_stats.values()]
    macro_avg_cer = np.mean(doc_cers) if doc_cers else 1.0
    
    # CER variants (macro-average sur documents)
    doc_cer_variants = {}
    for variant_name in NORMALIZATIONS.keys():
        variant_key = f'avg_cer_{variant_name}'
        doc_variant_cers = [d[variant_key] for d in document_stats.values()]
        doc_cer_variants[variant_name] = np.mean(doc_variant_cers) if doc_variant_cers else 1.0
    
    # Autres métriques document (macro-average)
    doc_avg_iou = np.mean([d['avg_iou'] for d in document_stats.values()])
    doc_avg_format = np.mean([d['avg_format_score'] for d in document_stats.values()])
    doc_avg_recall = np.mean([d['avg_recall'] for d in document_stats.values()])
    doc_avg_precision = np.mean([d['avg_precision'] for d in document_stats.values()])
    
    # Affichage des résultats
    _print_results(
        all_metrics, micro_avg_cer, page_cer_variants, avg_iou, avg_format, avg_recall, avg_precision,
        document_stats, num_documents, macro_avg_cer, doc_cer_variants, doc_avg_iou,
        doc_avg_format, doc_avg_recall, doc_avg_precision, error_stats
    )
    
    # Construire les dictionnaires de statistiques
    per_page_summary = _build_page_summary(
        micro_avg_cer, avg_iou, avg_format, avg_recall, avg_precision,
        len(all_metrics), page_cer_variants
    )
    
    per_document_summary = _build_document_summary(
        macro_avg_cer, doc_avg_iou, doc_avg_format, doc_avg_recall, doc_avg_precision,
        num_documents, doc_cer_variants
    )
    
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
    
    # Générer le rapport HTML interactif
    html_report_path = generate_html_report(
        all_metrics=all_metrics,
        document_stats=document_stats,
        error_stats=error_stats,
        page_cer_variants=page_cer_variants,
        doc_cer_variants=doc_cer_variants,
        output_dir=report_output_dir,
        report_name="evaluation_report"
    )
    
    print(f"\n📄 Rapports générés:")
    print(f"   - Markdown: {report_path} (pour VSCode)")
    print(f"   - HTML:     {html_report_path} (pour navigateur)")
    
    if output_dir:
        print(f"\n📁 Fichiers LabelMe sauvegardés dans: {output_dir}/labelme/")
        print("   - *_pred.json : prédictions du modèle")
        print("   - *_gt.json   : vérités terrain")
    
    return output


def _print_results(
    all_metrics, micro_avg_cer, page_cer_variants, avg_iou, avg_format, avg_recall, avg_precision,
    document_stats, num_documents, macro_avg_cer, doc_cer_variants, doc_avg_iou,
    doc_avg_format, doc_avg_recall, doc_avg_precision, error_stats
):
    """Affiche les résultats de l'évaluation dans la console."""
    print("\n" + "="*70)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*70)
    
    # Niveau page (micro-average: pondéré par caractères)
    print(f"\n--- NIVEAU PAGE - Micro-Average ({len(all_metrics)} pages) ---")
    print(f"CER (pondéré par caractères): {micro_avg_cer:.4f}")
    print(f"  ├─ sans accents:            {page_cer_variants['no_accents']:.4f}")
    print(f"  ├─ minuscules:              {page_cer_variants['lowercase']:.4f}")
    print(f"  ├─ chars normalisés:        {page_cer_variants['normalized_chars']:.4f}")
    print(f"  ├─ sans ponctuation:        {page_cer_variants['no_punctuation']:.4f}")
    print(f"  └─ normalisé complet:       {page_cer_variants['normalized']:.4f}")
    print(f"IoU moyen:                    {avg_iou:.4f}")
    print(f"Recall moyen:                 {avg_recall:.4f}")
    print(f"Precision moyenne:            {avg_precision:.4f}")
    
    # Niveau corpus (macro-average: moyenne des CER par document)
    print(f"\n--- NIVEAU CORPUS - Macro-Average ({num_documents} documents) ---")
    print(f"CER (moyenne par document):   {macro_avg_cer:.4f}")
    print(f"  ├─ sans accents:            {doc_cer_variants['no_accents']:.4f}")
    print(f"  ├─ minuscules:              {doc_cer_variants['lowercase']:.4f}")
    print(f"  ├─ chars normalisés:        {doc_cer_variants['normalized_chars']:.4f}")
    print(f"  ├─ sans ponctuation:        {doc_cer_variants['no_punctuation']:.4f}")
    print(f"  └─ normalisé complet:       {doc_cer_variants['normalized']:.4f}")
    print(f"IoU moyen:                    {doc_avg_iou:.4f}")
    print(f"Recall moyen:                 {doc_avg_recall:.4f}")
    print(f"Precision moyenne:            {doc_avg_precision:.4f}")
    
    # Afficher le détail par document
    print(f"\n--- DÉTAIL PAR DOCUMENT ---")
    sorted_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'], reverse=True)
    for doc_name, stats in sorted_docs[:10]:
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


def _build_page_summary(avg_cer, avg_iou, avg_format, avg_recall, avg_precision, num_pages, page_cer_variants):
    """Construit le dictionnaire de résumé par page."""
    summary = {
        'avg_cer': float(avg_cer),
        'avg_iou': float(avg_iou),
        'avg_format_score': float(avg_format),
        'avg_recall': float(avg_recall),
        'avg_precision': float(avg_precision),
        'num_pages': num_pages
    }
    for variant_name, variant_cer in page_cer_variants.items():
        summary[f'avg_cer_{variant_name}'] = float(variant_cer)
    return summary


def _build_document_summary(doc_avg_cer, doc_avg_iou, doc_avg_format, doc_avg_recall, doc_avg_precision, num_documents, doc_cer_variants):
    """Construit le dictionnaire de résumé par document."""
    summary = {
        'avg_cer': float(doc_avg_cer),
        'avg_iou': float(doc_avg_iou),
        'avg_format_score': float(doc_avg_format),
        'avg_recall': float(doc_avg_recall),
        'avg_precision': float(doc_avg_precision),
        'num_documents': num_documents
    }
    for variant_name, variant_cer in doc_cer_variants.items():
        summary[f'avg_cer_{variant_name}'] = float(variant_cer)
    return summary


def run_evaluation(api_url: str,
                  test_json_path: str,
                  system_prompt: str,
                  user_prompt: str,
                  max_concurrent: int = 10,
                  output_dir: str = "labelme_outputs"):
    """
    Wrapper synchrone pour l'évaluation asynchrone.
    
    Args:
        api_url: URL du serveur VLLM
        test_json_path: Chemin vers les données de test
        system_prompt: Prompt système
        user_prompt: Prompt utilisateur
        max_concurrent: Nombre de requêtes simultanées
        output_dir: Dossier pour les fichiers LabelMe (None pour désactiver)
    
    Returns:
        Dictionnaire avec les résultats complets de l'évaluation
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
    TEST_JSON_PATH = "../Qwen3/Data/Datasets/Generated/jsonraw.json"
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