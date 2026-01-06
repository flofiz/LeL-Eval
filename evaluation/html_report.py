"""
Générateur de rapport HTML interactif avec Plotly.

Ce module génère un rapport HTML autonome avec:
- Graphiques interactifs (zoom, hover, pan)
- Onglets pour organiser le contenu
- Sections pliables (accordéons)
- Design moderne avec Bootstrap
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

# Template HTML de base avec Bootstrap et Plotly
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation OCR - {timestamp}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary-color: #4F46E5;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --danger-color: #EF4444;
            --bg-dark: #1F2937;
        }}
        body {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            min-height: 100vh;
        }}
        .navbar {{
            background: var(--bg-dark) !important;
        }}
        .card {{
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}
        .card-header {{
            background: linear-gradient(135deg, var(--primary-color), #7C3AED);
            color: white;
            border-radius: 12px 12px 0 0 !important;
            font-weight: 600;
        }}
        .metric-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: white;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }}
        .metric-label {{
            color: #6B7280;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .nav-tabs .nav-link {{
            border: none;
            color: #6B7280;
            font-weight: 500;
        }}
        .nav-tabs .nav-link.active {{
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            background: transparent;
        }}
        .accordion-button:not(.collapsed) {{
            background-color: rgba(79, 70, 229, 0.1);
            color: var(--primary-color);
        }}
        .badge-success {{ background-color: var(--success-color); }}
        .badge-warning {{ background-color: var(--warning-color); }}
        .badge-danger {{ background-color: var(--danger-color); }}
        .plotly-graph {{
            width: 100%;
            height: 350px;
        }}
        .table-hover tbody tr:hover {{
            background-color: rgba(79, 70, 229, 0.05);
        }}
        .bias-indicator {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .bias-ok {{ background: #D1FAE5; color: #065F46; }}
        .bias-warning {{ background: #FEF3C7; color: #92400E; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark mb-4">
        <div class="container d-flex justify-content-between align-items-center">
            <span class="navbar-brand mb-0 h1">📊 Rapport d'Évaluation OCR</span>
            <div class="d-flex align-items-center gap-3">
                <div class="d-flex align-items-center">
                    <label class="text-light me-2 small">Document:</label>
                    <select id="docSelector" class="form-select form-select-sm" style="width: auto;" onchange="refreshDashboard()">
                        <option value="all">Tous les documents</option>
                        {doc_options}
                    </select>
                </div>
                <div class="d-flex align-items-center">
                    <label class="text-light me-2 small">Normalisation:</label>
                    <select id="normSelector" class="form-select form-select-sm" style="width: auto;" onchange="refreshDashboard(); refreshFilterTab()">
                        <option value="base">Base (aucune)</option>
                        <option value="no_accents">Sans accents</option>
                        <option value="lowercase">Minuscules</option>
                        <option value="normalized_chars">Chars normalisés</option>
                        <option value="no_punctuation">Sans ponctuation</option>
                        <option value="historical_abbrev">Abbr. historiques</option>
                        <option value="normalized">Normalisé (tout)</option>
                    </select>
                </div>
                <span class="text-light small">{timestamp}</span>
            </div>
        </div>
    </nav>

    <div class="container">
        <!-- Métriques principales -->
        <div class="row g-4 mb-4">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="metric-micro-cer">{micro_cer:.1f}%</div>
                    <div class="metric-label">Micro-CER</div>
                    <small class="text-muted">Pondéré par caractères</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="metric-macro-cer">{macro_cer:.1f}%</div>
                    <div class="metric-label">Macro-CER</div>
                    <small class="text-muted">Moyenne par document</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{avg_iou:.1f}%</div>
                    <div class="metric-label">IoU Moyen</div>
                    <small class="text-muted">Qualité segmentation</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="metric-bias" style="color: {bias_color};">{cer_bias:+.1f}pp</div>
                    <div class="metric-label">Biais Corpus</div>
                    <small class="text-muted" id="metric-bias-text">{bias_interpretation}</small>
                </div>
            </div>
        </div>

        <!-- Onglets principaux -->
        <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
            <li class="nav-item">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview">📈 Vue d'ensemble</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#documents">📄 Par Document</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#errors">🔍 Analyse Erreurs</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#advanced">⚙️ Avancé</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#filter-analysis">🎯 Filtrage Perplexité</button>
            </li>
        </ul>

        <div class="tab-content">
            <!-- Onglet Vue d'ensemble -->
            <div class="tab-pane fade show active" id="overview">
                <div class="row">
                    <div class="col-12 col-lg-6">
                        <div class="card">
                            <div class="card-header">Distribution du CER par Page</div>
                            <div class="card-body">
                                <div id="cer-distribution" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-12 col-lg-6">
                        <div class="card">
                            <div class="card-header">Distribution de l'IoU</div>
                            <div class="card-body">
                                <div id="iou-distribution" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>CER vs Complexité (Nombre de lignes)</span>
                                <button class="btn btn-sm btn-outline-secondary" onclick="toggleScale('cer-complexity')">📊 Échelle Log/Lin</button>
                            </div>
                            <div class="card-body">
                                <div id="cer-complexity" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>CER vs Perplexité</span>
                                <div class="d-flex gap-2">
                                    <button class="btn btn-sm btn-outline-primary" onclick="cyclePerpType('cer-perplexity')">🔀 Global/Trans/Seg</button>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="toggleScale('cer-perplexity')">📊 Log/Lin</button>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="cer-perplexity" class="plotly-graph"></div>
                                <small class="text-muted d-block text-center mt-2" id="cer-perplexity-label">Perplexité: Globale</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Training Pages Analysis -->
                <div class="row">
                    <div class="col-12 col-lg-6">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>Impact Données d'Entraînement (Par Document)</span>
                                <button class="btn btn-sm btn-outline-primary" onclick="cyclePerpType('perp-vs-training-doc')">🔀 Full/Trans/Seg</button>
                            </div>
                            <div class="card-body">
                                <div id="perp-vs-training-doc" class="plotly-graph"></div>
                                <small class="text-muted d-block text-center mt-2" id="perp-vs-training-doc-label">Perplexité: Globale</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-12 col-lg-6">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>Impact Données d'Entraînement (Par Page)</span>
                                <button class="btn btn-sm btn-outline-primary" onclick="cyclePerpType('perp-vs-training-page')">🔀 Full/Trans/Seg</button>
                            </div>
                            <div class="card-body">
                                <div id="perp-vs-training-page" class="plotly-graph"></div>
                                <small class="text-muted d-block text-center mt-2" id="perp-vs-training-page-label">Perplexité: Globale</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- IoU vs Perplexity -->
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>IoU vs Perplexité</span>
                                <div class="d-flex gap-2">
                                    <button class="btn btn-sm btn-outline-primary" onclick="togglePerpType('iou-perplexity')">🔀 Global/Segmentation</button>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="toggleScale('iou-perplexity')">📊 Log/Lin</button>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="iou-perplexity" class="plotly-graph"></div>
                                <small class="text-muted d-block text-center mt-2" id="iou-perplexity-label">Perplexité: Globale</small>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>CER vs Erreurs de Segmentation (par page)</span>
                                <button class="btn btn-sm btn-outline-secondary" onclick="toggleScale('cer-seg-page')">📊 Échelle Log/Lin</button>
                            </div>
                            <div class="card-body">
                                <div id="cer-seg-page" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">CER par Variante de Normalisation</div>
                            <div class="card-body">
                                <div id="cer-variants" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Onglet Par Document -->
            <div class="tab-pane fade" id="documents">
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">CER par Document</div>
                            <div class="card-body">
                                <div id="cer-by-document" style="width:100%; height:600px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">IoU par Document</div>
                            <div class="card-body">
                                <div id="iou-by-document" style="width:100%; height:500px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">CER vs Erreurs de Segmentation</div>
                            <div class="card-body">
                                <div id="cer-vs-segmentation" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Tableau des documents -->
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>📋 Détail par Document</span>
                        <input type="text" class="form-control form-control-sm w-25" id="docSearch" placeholder="Rechercher...">
                    </div>
                    <div class="card-body">
                        <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                            <table class="table table-hover table-sm" id="docTable">
                                <thead class="sticky-top bg-white">
                                    <tr>
                                        <th onclick="sortTable(0)">Document ↕</th>
                                        <th onclick="sortTable(1)">CER ↕</th>
                                        <th onclick="sortTable(2)">IoU ↕</th>
                                        <th onclick="sortTable(3)">Recall ↕</th>
                                        <th onclick="sortTable(4)">Pages ↕</th>
                                        <th onclick="sortTable(5)">Training Pages ↕</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {doc_table_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Onglet Analyse Erreurs -->
            <div class="tab-pane fade" id="errors">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Répartition des Types d'Erreurs</div>
                            <div class="card-body">
                                <div id="error-types" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Statistiques des Erreurs</div>
                            <div class="card-body">
                                {error_stats_html}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Accordéons pour les confusions -->
                <div class="accordion" id="confusionAccordion">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#confusionsRaw">
                                Top 20 Confusions de Caractères (brutes)
                            </button>
                        </h2>
                        <div id="confusionsRaw" class="accordion-collapse collapse show">
                            <div class="accordion-body">
                                {confusions_raw_html}
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#confusionsNorm">
                                Top 20 Confusions Normalisées (vraies erreurs)
                            </button>
                        </h2>
                        <div id="confusionsNorm" class="accordion-collapse collapse">
                            <div class="accordion-body">
                                {confusions_norm_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Onglet Avancé -->
            <div class="tab-pane fade" id="advanced">
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">CER par Document et Normalisation</div>
                            <div class="card-body">
                                <div id="cer-doc-norm-comparison" style="width:100%; height:800px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span>CER par Document (normalisation sélectionnée)</span>
                                <button class="btn btn-sm btn-outline-secondary" onclick="toggleScale('cer-doc-selected-norm')">📊 Échelle Log/Lin</button>
                            </div>
                            <div class="card-body">
                                <div id="cer-doc-selected-norm" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">Variantes CER - Niveau Page vs Document</div>
                            <div class="card-body">
                                <div id="cer-variants-comparison" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12 col-lg-6 mx-auto">
                        <div class="card">
                            <div class="card-header">Métriques Radar</div>
                            <div class="card-body">
                                <div id="radar-chart" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Données brutes -->
                <div class="card">
                    <div class="card-header">
                        <button class="btn btn-sm btn-outline-light" onclick="downloadJSON()">
                            📥 Télécharger les données JSON
                        </button>
                    </div>
                    <div class="card-body">
                        <pre style="max-height: 300px; overflow: auto; font-size: 0.75rem;" id="rawData"></pre>
                    </div>
                </div>
            </div>
            
            <!-- Onglet Filtrage Perplexité -->
            <div class="tab-pane fade" id="filter-analysis">
                <!-- Contrôles des seuils -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">🎛️ Paramètres de Filtrage</div>
                            <div class="card-body">
                                <div class="row g-4">
                                    <div class="col-md-4">
                                        <label class="form-label">Seuil Perplexité Globale: <span id="thresh-global-val">1.30</span></label>
                                        <input type="range" class="form-range" id="thresh-global" min="0.5" max="3.0" step="0.05" value="1.3" oninput="updateFilterThresholds()">
                                    </div>
                                    <div class="col-md-4">
                                        <label class="form-label">Seuil Perplexité Transcription: <span id="thresh-trans-val">1.20</span></label>
                                        <input type="range" class="form-range" id="thresh-trans" min="0.5" max="3.0" step="0.05" value="1.2" oninput="updateFilterThresholds()">
                                    </div>
                                    <div class="col-md-4">
                                        <label class="form-label">Seuil CER "Bonne Page" (%): <span id="thresh-cer-val">10</span>%</label>
                                        <input type="range" class="form-range" id="thresh-cer" min="1" max="30" step="1" value="10" oninput="updateFilterThresholds()">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Métriques de filtrage -->
                <div class="row g-4 mb-4">
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-precision">-</div>
                            <div class="metric-label">Précision</div>
                            <small class="text-muted">TP / (TP + FP)</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-recall">-</div>
                            <div class="metric-label">Rappel</div>
                            <small class="text-muted">TP / (TP + FN)</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-f1">-</div>
                            <div class="metric-label">F1 Score</div>
                            <small class="text-muted">Harmonic mean</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-accept-rate">-</div>
                            <div class="metric-label">Taux Acceptation</div>
                            <small class="text-muted">Acceptées / Total</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-accepted-cer">-</div>
                            <div class="metric-label">CER Acceptées</div>
                            <small class="text-muted">Micro-CER</small>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="metric-card">
                            <div class="metric-value" id="filter-rejected-cer">-</div>
                            <div class="metric-label">CER Rejetées</div>
                            <small class="text-muted">Micro-CER</small>
                        </div>
                    </div>
                </div>
                
                <!-- Graphique acceptance par document -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">Taux d'Acceptation par Document</div>
                            <div class="card-body">
                                <div id="filter-doc-acceptance" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Sous-onglets Acceptées/Rejetées -->
                <ul class="nav nav-pills mb-3">
                    <li class="nav-item">
                        <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#accepted-pages">✅ Pages Acceptées</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="pill" data-bs-target="#rejected-pages">❌ Pages Rejetées</button>
                    </li>
                </ul>
                
                <div class="tab-content">
                    <!-- Sous-onglet Pages Acceptées -->
                    <div class="tab-pane fade show active" id="accepted-pages">
                        <div class="row g-4 mb-4">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="accepted-micro-cer">-</div>
                                    <div class="metric-label">Micro-CER</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="accepted-macro-cer">-</div>
                                    <div class="metric-label">Macro-CER</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="accepted-count">-</div>
                                    <div class="metric-label">Nb Pages</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="accepted-avg-iou">-</div>
                                    <div class="metric-label">IoU Moyen</div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12 col-lg-6">
                                <div class="card">
                                    <div class="card-header">Distribution CER (Acceptées)</div>
                                    <div class="card-body">
                                        <div id="accepted-cer-dist" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-12 col-lg-6">
                                <div class="card">
                                    <div class="card-header">Distribution IoU (Acceptées)</div>
                                    <div class="card-body">
                                        <div id="accepted-iou-dist" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">CER vs Complexité (Acceptées)</div>
                                    <div class="card-body">
                                        <div id="accepted-cer-complexity" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Sous-onglet Pages Rejetées -->
                    <div class="tab-pane fade" id="rejected-pages">
                        <div class="row g-4 mb-4">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="rejected-micro-cer">-</div>
                                    <div class="metric-label">Micro-CER</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="rejected-macro-cer">-</div>
                                    <div class="metric-label">Macro-CER</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="rejected-count">-</div>
                                    <div class="metric-label">Nb Pages</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value" id="rejected-avg-iou">-</div>
                                    <div class="metric-label">IoU Moyen</div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12 col-lg-6">
                                <div class="card">
                                    <div class="card-header">Distribution CER (Rejetées)</div>
                                    <div class="card-body">
                                        <div id="rejected-cer-dist" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-12 col-lg-6">
                                <div class="card">
                                    <div class="card-header">Distribution IoU (Rejetées)</div>
                                    <div class="card-body">
                                        <div id="rejected-iou-dist" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">CER vs Complexité (Rejetées)</div>
                                    <div class="card-body">
                                        <div id="rejected-cer-complexity" class="plotly-graph"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <div id="js-error-alert" class="alert alert-danger d-none container mt-4" role="alert">
        <h4 class="alert-heading">Erreur JavaScript</h4>
        <p>Une erreur est survenue lors de l'affichage des graphiques :</p>
        <pre id="js-error-message"></pre>
    </div>
    <script>
    try {{
        // Données embarquées
        var reportData = {json_data};
        
        // Afficher les données brutes
        document.getElementById('rawData').textContent = JSON.stringify(reportData.summary, null, 2);
        
        // Distribution CER
        Plotly.newPlot('cer-distribution', [{{
            x: reportData.page_cers,
            type: 'histogram',
            marker: {{ color: '#4F46E5', opacity: 0.7 }},
            nbinsx: 30
        }}], {{
            title: '',
            xaxis: {{ title: 'CER (%)' }},
            yaxis: {{ title: 'Nombre de pages' }},
            shapes: [
                {{ type: 'line', x0: {mean_cer}, x1: {mean_cer}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: {median_cer}, x1: {median_cer}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
            ],
            annotations: [
                {{ x: {mean_cer}, y: 1, yref: 'paper', text: 'Moyenne: {mean_cer:.1f}%', showarrow: false, yanchor: 'bottom', font: {{ color: '#EF4444' }} }},
                {{ x: {median_cer}, y: 0.9, yref: 'paper', text: 'Médiane: {median_cer:.1f}%', showarrow: false, yanchor: 'bottom', font: {{ color: '#10B981' }} }}
            ]
        }}, {{ responsive: true }});
        
        // Distribution IoU
        Plotly.newPlot('iou-distribution', [{{
            x: reportData.page_ious,
            type: 'histogram',
            marker: {{ color: '#10B981', opacity: 0.7 }},
            nbinsx: 30
        }}], {{
            xaxis: {{ title: 'IoU (%)' }},
            yaxis: {{ title: 'Nombre de pages' }},
            shapes: [
                {{ type: 'line', x0: {avg_iou}, x1: {avg_iou}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: {median_iou}, x1: {median_iou}, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#4F46E5', width: 2, dash: 'dot' }} }}
            ],
            annotations: [
                {{ x: {avg_iou}, y: 1, yref: 'paper', text: 'Moyenne: {avg_iou:.1f}%', showarrow: false, yanchor: 'bottom', font: {{ color: '#EF4444' }} }},
                {{ x: {median_iou}, y: 0.9, yref: 'paper', text: 'Médiane: {median_iou:.1f}%', showarrow: false, yanchor: 'bottom', font: {{ color: '#4F46E5' }} }}
            ]
        }}, {{ responsive: true }});
        
        // Variantes CER
        Plotly.newPlot('cer-variants', [{{
            x: {cer_variant_labels},
            y: {cer_variant_values},
            type: 'bar',
            marker: {{ color: ['#4F46E5', '#7C3AED', '#A855F7', '#C084FC', '#E879F9', '#F0ABFC'] }}
        }}], {{
            yaxis: {{ title: 'CER (%)' }}
        }}, {{ responsive: true }});
        
        // CER vs Complexité
        Plotly.newPlot('cer-complexity', [{{
            x: reportData.page_lines,
            y: reportData.page_cers,
            mode: 'markers',
            type: 'scatter',
            name: 'Pages',
            marker: {{ 
                color: reportData.page_cers, 
                colorscale: 'RdYlGn_r',
                size: 8,
                opacity: 0.6
            }},
            text: reportData.page_names,
            hovertemplate: '<b>%{{text}}</b><br>Lignes: %{{x}}<br>CER: %{{y:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'Nombre de lignes', range: [0, Math.max(...reportData.page_lines) * 1.1] }},
            yaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
            margin: {{ l: 60, r: 30, t: 30, b: 50 }},
            shapes: [
                {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
            ]
        }}, {{ responsive: true }});
        
        // Ajouter ligne de tendance CER vs Complexité
        (function() {{
            const x = reportData.page_lines;
            const y = reportData.page_cers;
            const n = x.length;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
            const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            const intercept = (sumY - slope * sumX) / n;
            const xMin = Math.min(...x);
            const xMax = Math.max(...x);
            const correlation = (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - sumX * sumX) * (n * y.reduce((a, b) => a + b*b, 0) - sumY * sumY));
            Plotly.addTraces('cer-complexity', {{
                x: [xMin, xMax],
                y: [slope * xMin + intercept, slope * xMax + intercept],
                mode: 'lines',
                type: 'scatter',
                name: 'Tendance (r=' + correlation.toFixed(2) + ')',
                line: {{ color: '#6B7280', width: 2, dash: 'dash' }}
            }});
        }})();
        
        // CER vs Perplexité
        if (reportData.page_perplexities && reportData.page_perplexities.some(p => p !== null)) {{
            const validIndices = reportData.page_perplexities.map((p, i) => p !== null ? i : -1).filter(i => i >= 0);
            const validPerp = validIndices.map(i => reportData.page_perplexities[i]);
            const validCers = validIndices.map(i => reportData.page_cers[i]);
            const validNames = validIndices.map(i => reportData.page_names[i]);
            
            Plotly.newPlot('cer-perplexity', [{{
                x: validPerp,
                y: validCers,
                mode: 'markers',
                type: 'scatter',
                name: 'Pages',
                marker: {{ 
                    color: validCers, 
                    colorscale: 'RdYlGn_r',
                    size: 8,
                    opacity: 0.6
                }},
                text: validNames,
                hovertemplate: '<b>%{{text}}</b><br>Perplexité: %{{x:.1f}}<br>CER: %{{y:.2f}}%<extra></extra>'
            }}], {{
                xaxis: {{ title: 'Perplexité (+ bas = + confiant)', range: [0, Math.max(...validPerp) * 1.1] }},
                yaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
                margin: {{ l: 60, r: 30, t: 30, b: 50 }},
                shapes: [
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                    {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
                ]
            }}, {{ responsive: true }});
            
            // Trendline CER vs Perplexité
            (function() {{
                const x = validPerp;
                const y = validCers;
                const n = x.length;
                if (n < 2) return;
                const sumX = x.reduce((a, b) => a + b, 0);
                const sumY = y.reduce((a, b) => a + b, 0);
                const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
                const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
                const denom = n * sumX2 - sumX * sumX;
                if (Math.abs(denom) < 0.001) return;
                const slope = (n * sumXY - sumX * sumY) / denom;
                const intercept = (sumY - slope * sumX) / n;
                const xMin = Math.min(...x);
                const xMax = Math.max(...x);
                const sumY2 = y.reduce((a, b) => a + b*b, 0);
                const corrDenom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
                const correlation = corrDenom > 0 ? (n * sumXY - sumX * sumY) / corrDenom : 0;
                Plotly.addTraces('cer-perplexity', {{
                    x: [xMin, xMax],
                    y: [slope * xMin + intercept, slope * xMax + intercept],
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Tendance (r=' + correlation.toFixed(2) + ')',
                    line: {{ color: '#6B7280', width: 2, dash: 'dash' }}
                }});
            }})();
        }} else {{
            document.getElementById('cer-perplexity').innerHTML = '<p class="text-muted text-center">Données de perplexité non disponibles</p>';
        }}
        
        // --- Perplexity vs Training Pages (Document Level) ---
        if (reportData.doc_perplexities && reportData.doc_training_count && reportData.doc_perplexities.some(p => p !== null)) {{
            const validDocIndices = reportData.doc_perplexities.map((p, i) => p !== null ? i : -1).filter(i => i >= 0);
            const validDocPerp = validDocIndices.map(i => reportData.doc_perplexities[i]);
            const validDocTrain = validDocIndices.map(i => reportData.doc_training_count[i]);
            const validDocCers = validDocIndices.map(i => reportData.doc_cers[i]);
            const validDocNames = validDocIndices.map(i => reportData.doc_names[i]);

            Plotly.newPlot('perp-vs-training-doc', [{{
                x: validDocTrain,
                y: validDocPerp,
                mode: 'markers',
                type: 'scatter',
                name: 'Documents',
                marker: {{
                    color: validDocCers,
                    colorscale: 'RdYlGn_r',
                    size: 10,
                    colorbar: {{ title: 'CER (%)', len: 0.5 }},
                    line: {{ color: 'white', width: 1 }}
                }},
                text: validDocNames,
                hovertemplate: '<b>%{{text}}</b><br>Training Pages: %{{x}}<br>Avg Perplexity: %{{y:.2f}}<br>Avg CER: %{{marker.color:.2f}}%<extra></extra>'
            }}], {{
                xaxis: {{ title: "Pages d'entraînement" }},
                yaxis: {{ title: 'Perplexité Moyenne' }},
                margin: {{ l: 50, r: 20, t: 20, b: 40 }}
            }}, {{ responsive: true }});
        }} else {{
             document.getElementById('perp-vs-training-doc').innerHTML = '<p class="text-muted text-center">Données non disponibles</p>';
        }}

        // --- Perplexity vs Training Pages (Page Level) ---
        if (reportData.page_perplexities && reportData.page_training_count && reportData.page_perplexities.some(p => p !== null)) {{
             const validPageIndices = reportData.page_perplexities.map((p, i) => p !== null ? i : -1).filter(i => i >= 0);
             const validPagePerp = validPageIndices.map(i => reportData.page_perplexities[i]);
             const validPageTrain = validPageIndices.map(i => reportData.page_training_count[i]);
             const validPageCers = validPageIndices.map(i => reportData.page_cers[i]);
             const validPageNames = validPageIndices.map(i => reportData.page_names[i]);

             Plotly.newPlot('perp-vs-training-page', [{{
                x: validPageTrain,
                y: validPagePerp,
                mode: 'markers',
                type: 'scatter',
                name: 'Pages',
                marker: {{
                    color: validPageCers,
                    colorscale: 'RdYlGn_r',
                    size: 6,
                    opacity: 0.6,
                     colorbar: {{ title: 'CER (%)', len: 0.5 }}
                }},
                text: validPageNames,
                hovertemplate: '<b>%{{text}}</b><br>Training Pages (Doc): %{{x}}<br>Perplexity: %{{y:.2f}}<br>CER: %{{marker.color:.2f}}%<extra></extra>'
            }}], {{
                xaxis: {{ title: "Pages d'entraînement (Document)" }},
                yaxis: {{ title: 'Perplexité Page' }},
                margin: {{ l: 50, r: 20, t: 20, b: 40 }}
            }}, {{ responsive: true }});
        }} else {{
             document.getElementById('perp-vs-training-page').innerHTML = '<p class="text-muted text-center">Données non disponibles</p>';
        }}
        
        // CER vs Erreurs de Segmentation (par page)
        Plotly.newPlot('cer-seg-page', [{{
            x: reportData.page_seg_errors,
            y: reportData.page_cers,
            mode: 'markers',
            type: 'scatter',
            name: 'Pages',
            marker: {{ 
                color: reportData.page_cers,
                colorscale: 'RdYlGn',
                size: 8,
                opacity: 0.6
            }},
            text: reportData.page_names,
            hovertemplate: '<b>%{{text}}</b><br>Seg Error: %{{x:.1f}}%<br>CER: %{{y:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'Taux erreur segmentation (%)' }},
            yaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
            margin: {{ l: 60, r: 30, t: 30, b: 50 }},
            shapes: [
                {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
            ]
        }}, {{ responsive: true }});
        
        // CER par document (bar chart horizontal)
        Plotly.newPlot('cer-by-document', [{{
            y: reportData.doc_names,
            x: reportData.doc_cers,
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: reportData.doc_cers,
                colorscale: 'RdYlGn_r'
            }},
            hovertemplate: '<b>%{{y}}</b><br>CER: %{{x:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
            margin: {{ l: 180, r: 30, t: 20, b: 40 }},
            height: Math.max(450, reportData.doc_names.length * 30),
            autosize: true,
            shapes: [
                {{ type: 'line', x0: reportData.doc_cers.reduce((a,b) => a+b, 0) / reportData.doc_cers.length, x1: reportData.doc_cers.reduce((a,b) => a+b, 0) / reportData.doc_cers.length, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: [...reportData.doc_cers].sort((a,b) => a-b)[Math.floor(reportData.doc_cers.length/2)], x1: [...reportData.doc_cers].sort((a,b) => a-b)[Math.floor(reportData.doc_cers.length/2)], y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
            ]
        }}, {{ responsive: true }});
        
        // IoU par document
        Plotly.newPlot('iou-by-document', [{{
            y: reportData.doc_names,
            x: reportData.doc_ious,
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: reportData.doc_ious,
                colorscale: 'RdYlGn_r'
            }}
        }}], {{
            xaxis: {{ title: 'IoU (%)', range: [0, 100] }},
            margin: {{ l: 180, r: 30, t: 20, b: 40 }},
            height: 450,
            autosize: true
        }}, {{ responsive: true }});
        
        // CER vs Segmentation
        Plotly.newPlot('cer-vs-segmentation', [{{
            x: reportData.doc_seg_errors,
            y: reportData.doc_cers,
            mode: 'markers',
            type: 'scatter',
            name: 'Documents',
            marker: {{ 
                color: reportData.doc_cers, 
                colorscale: 'RdYlGn_r',
                size: reportData.doc_sizes,
                sizemode: 'area',
                sizeref: 2.*Math.max(...reportData.doc_sizes)/(40.**2),
                sizemin: 4
            }},
            text: reportData.doc_names,
            hovertemplate: '<b>%{{text}}</b><br>Seg Error: %{{x:.1f}}%<br>CER: %{{y:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'Taux erreur segmentation (%)', range: [0, 100] }},
            yaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
            margin: {{ l: 60, r: 30, t: 30, b: 50 }},
            height: 350,
            autosize: true
        }}, {{ responsive: true }});
        
        // Ajouter ligne de tendance CER vs Segmentation
        (function() {{
            const x = reportData.doc_seg_errors;
            const y = reportData.doc_cers;
            const n = x.length;
            if (n < 2) return;
            const sumX = x.reduce((a, b) => a + b, 0);
            const sumY = y.reduce((a, b) => a + b, 0);
            const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
            const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
            const denom = n * sumX2 - sumX * sumX;
            if (Math.abs(denom) < 0.001) return;
            const slope = (n * sumXY - sumX * sumY) / denom;
            const intercept = (sumY - slope * sumX) / n;
            const xMin = Math.min(...x);
            const xMax = Math.max(...x);
            const sumY2 = y.reduce((a, b) => a + b*b, 0);
            const corrDenom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            const correlation = corrDenom > 0 ? (n * sumXY - sumX * sumY) / corrDenom : 0;
            Plotly.addTraces('cer-vs-segmentation', {{
                x: [xMin, xMax],
                y: [slope * xMin + intercept, slope * xMax + intercept],
                mode: 'lines',
                type: 'scatter',
                name: 'Tendance (r=' + correlation.toFixed(2) + ')',
                line: {{ color: '#6B7280', width: 2, dash: 'dash' }}
            }});
        }})();
        
        // Types d'erreurs (pie chart)
        if (reportData.error_totals) {{
            Plotly.newPlot('error-types', [{{
                values: [reportData.error_totals.insertions, reportData.error_totals.deletions, reportData.error_totals.substitutions],
                labels: ['Insertions', 'Suppressions', 'Substitutions'],
                type: 'pie',
                marker: {{ colors: ['#4F46E5', '#EF4444', '#F59E0B'] }},
                hole: 0.4,
                textposition: 'inside'
            }}], {{
                margin: {{ l: 20, r: 20, t: 20, b: 20 }},
                height: 350,
                autosize: true,
                showlegend: true,
                legend: {{ orientation: 'h', y: -0.1 }}
            }}, {{ responsive: true }});
        }}
        
        // CER par Document et Normalisation (grouped bar chart)
        const normColors = {{
            'base': '#3B82F6',           // Bleu
            'no_accents': '#10B981',     // Vert
            'lowercase': '#8B5CF6',       // Violet
            'normalized_chars': '#F59E0B', // Orange
            'no_punctuation': '#EF4444',   // Rouge
            'historical_abbrev': '#EC4899', // Rose
            'normalized': '#6B7280'        // Gris
        }};
        const normLabels = {{
            'base': 'Base',
            'no_accents': 'Sans accents',
            'lowercase': 'Minuscules',
            'normalized_chars': 'Chars norm.',
            'no_punctuation': 'Sans ponct.',
            'historical_abbrev': 'Abbr. hist.',
            'normalized': 'Normalisé'
        }};
        
        const docNormTraces = [];
        for (const [normKey, cers] of Object.entries(reportData.doc_cers_by_norm)) {{
            docNormTraces.push({{
                y: reportData.doc_names,
                x: cers,
                type: 'bar',
                orientation: 'h',
                name: normLabels[normKey] || normKey,
                marker: {{ color: normColors[normKey] || '#999' }},
                hovertemplate: '<b>%{{y}}</b><br>' + (normLabels[normKey] || normKey) + ': %{{x:.2f}}%<extra></extra>'
            }});
        }}
        
        Plotly.newPlot('cer-doc-norm-comparison', docNormTraces, {{
            barmode: 'group',
            xaxis: {{ title: 'CER (%)' }},
            yaxis: {{ automargin: true }},
            margin: {{ l: 180, r: 30, t: 30, b: 50 }},
            legend: {{ orientation: 'h', y: 1.1 }},
            height: Math.max(600, reportData.doc_names.length * 40)
        }}, {{ responsive: true }});
        
        // CER par Document (normalisation sélectionnée) - initial render avec base
        Plotly.newPlot('cer-doc-selected-norm', [{{
            y: reportData.doc_names,
            x: reportData.doc_cers,
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: reportData.doc_cers,
                colorscale: 'RdYlGn'
            }},
            hovertemplate: '<b>%{{y}}</b><br>CER: %{{x:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'CER (%)', type: 'log', range: [0, 2] }},
            margin: {{ l: 180, r: 30, t: 20, b: 40 }},
            height: Math.max(500, reportData.doc_names.length * 35),
            autosize: true,
            shapes: [
                {{ type: 'line', x0: reportData.doc_cers.reduce((a,b) => a+b, 0) / reportData.doc_cers.length, x1: reportData.doc_cers.reduce((a,b) => a+b, 0) / reportData.doc_cers.length, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                {{ type: 'line', x0: [...reportData.doc_cers].sort((a,b) => a-b)[Math.floor(reportData.doc_cers.length/2)], x1: [...reportData.doc_cers].sort((a,b) => a-b)[Math.floor(reportData.doc_cers.length/2)], y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
            ]
        }}, {{ responsive: true }});
        
        // Radar chart
        Plotly.newPlot('radar-chart', [{{
            type: 'scatterpolar',
            r: [{avg_precision}, {avg_recall}, {avg_iou}, {avg_format}, {accuracy}],
            theta: ['Précision', 'Recall', 'IoU', 'Format', 'Accuracy'],
            fill: 'toself',
            marker: {{ color: '#4F46E5' }}
        }}], {{
            polar: {{ radialaxis: {{ visible: true, range: [0, 100] }} }},
            margin: {{ l: 60, r: 60, t: 40, b: 40 }},
            height: 350,
            autosize: true
        }}, {{ responsive: true }});
        
        // Comparaison variantes Page vs Document
        Plotly.newPlot('cer-variants-comparison', [{{
            x: {cer_variant_labels},
            y: {cer_variant_values},
            name: 'Niveau Page',
            type: 'bar'
        }}, {{
            x: {cer_variant_labels},
            y: {doc_cer_variant_values},
            name: 'Niveau Document',
            type: 'bar'
        }}], {{
            barmode: 'group',
            yaxis: {{ title: 'CER (%)' }},
            margin: {{ l: 50, r: 20, t: 20, b: 80 }},
            height: 350,
            autosize: true,
            legend: {{ orientation: 'h', y: -0.2 }}
        }}, {{ responsive: true }});
    }} catch (e) {{
        console.error("Error initializing charts:", e);
        document.getElementById('js-error-alert').classList.remove('d-none');
        document.getElementById('js-error-message').innerText += "\\nInit Error: " + e.message;
    }}
        
        // Filtrage tableau
        document.getElementById('docSearch').addEventListener('input', function(e) {{
            const filter = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#docTable tbody tr');
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? '' : 'none';
            }});
        }});
        
        // Tri tableau
        function sortTable(n) {{
            const table = document.getElementById('docTable');
            let switching = true, dir = 'asc', switchcount = 0;
            while (switching) {{
                switching = false;
                const rows = table.rows;
                for (let i = 1; i < rows.length - 1; i++) {{
                    let x = rows[i].getElementsByTagName('TD')[n];
                    let y = rows[i + 1].getElementsByTagName('TD')[n];
                    let xVal = n === 0 ? x.textContent : parseFloat(x.textContent);
                    let yVal = n === 0 ? y.textContent : parseFloat(y.textContent);
                    if ((dir === 'asc' && xVal > yVal) || (dir === 'desc' && xVal < yVal)) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                        break;
                    }}
                }}
                if (!switching && switchcount === 0 && dir === 'asc') {{ dir = 'desc'; switching = true; }}
            }}
        }}
        
        // Toggle échelle logarithmique/linéaire
        const scaleStates = {{}}; // Tracker l'état de chaque graphique (true = log, false = lin)
        function toggleScale(graphId) {{
            // Initialiser l'état si première utilisation
            if (scaleStates[graphId] === undefined) {{
                scaleStates[graphId] = true; // Par défaut on est en log
            }}
            
            // Basculer l'état
            scaleStates[graphId] = !scaleStates[graphId];
            const isLog = scaleStates[graphId];
            
            // Déterminer l'axe à modifier selon le graphique
            let update = {{}};
            if (graphId === 'cer-by-document') {{
                update = {{
                    'xaxis.type': isLog ? 'log' : 'linear',
                    'xaxis.range': isLog ? [0, 2] : [0, 100]
                }};
            }} else {{
                update = {{
                    'yaxis.type': isLog ? 'log' : 'linear',
                    'yaxis.range': isLog ? [0, 2] : [0, 100]
                }};
            }}
            
            Plotly.relayout(graphId, update);
        }}
        
        // Download JSON
        function downloadJSON() {{
            const blob = new Blob([JSON.stringify(reportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'evaluation_data.json';
            a.click();
        }}
        
        // Perplexity type state (0=global, 1=transcription, 2=segmentation)
        var perpTypeStates = {{
            'cer-perplexity': 0,      // Global ↔ Transcription (binary toggle)
            'iou-perplexity': 0,      // Global ↔ Segmentation (binary toggle)
            'perp-vs-training-doc': 0,  // Full/Trans/Seg (3-way cycle)
            'perp-vs-training-page': 0  // Full/Trans/Seg (3-way cycle)
        }};
        
        // Binary toggle for perplexity type (Global ↔ specific type)
        function togglePerpType(graphId) {{
            if (graphId === 'iou-perplexity') {{
                // Toggle between 0 (global) and 2 (segmentation)
                perpTypeStates[graphId] = perpTypeStates[graphId] === 0 ? 2 : 0;
                const label = perpTypeStates[graphId] === 0 ? 'Globale' : 'Segmentation';
                document.getElementById('iou-perplexity-label').textContent = 'Perplexité: ' + label;
            }}
            refreshDashboard();
        }}
        
        // 3-way cycle for training impact graphs
        function cyclePerpType(graphId) {{
            perpTypeStates[graphId] = (perpTypeStates[graphId] + 1) % 3;
            const labels = ['Globale', 'Transcription', 'Segmentation'];
            const label = labels[perpTypeStates[graphId]];
            document.getElementById(graphId + '-label').textContent = 'Perplexité: ' + label;
            refreshDashboard();
        }}
        
        // Filtrage par perplexité - mise à jour des seuils
        function updateFilterThresholds() {{
            const threshGlobal = parseFloat(document.getElementById('thresh-global').value);
            const threshTrans = parseFloat(document.getElementById('thresh-trans').value);
            const threshCer = parseFloat(document.getElementById('thresh-cer').value);
            
            document.getElementById('thresh-global-val').textContent = threshGlobal.toFixed(2);
            document.getElementById('thresh-trans-val').textContent = threshTrans.toFixed(2);
            document.getElementById('thresh-cer-val').textContent = threshCer;
            
            refreshFilterTab();
        }}
        
        // Rafraîchir l'onglet filtrage
        function refreshFilterTab() {{
            try {{
                const normKey = document.getElementById('normSelector').value;
                const threshGlobal = parseFloat(document.getElementById('thresh-global').value);
                const threshTrans = parseFloat(document.getElementById('thresh-trans').value);
                const threshCer = parseFloat(document.getElementById('thresh-cer').value);
                
                // Utiliser CER normalisé pour le ground truth
                const cerNormalized = reportData.page_cers_by_norm['normalized'] || reportData.page_cers;
                
                // Classifier les pages
                const accepted = [];
                const rejected = [];
                
                for (let i = 0; i < reportData.page_names.length; i++) {{
                    const perpGlobal = reportData.page_perplexities[i];
                    const perpTrans = reportData.page_perplexities_trans[i];
                    
                    // Une page est acceptée si perp global < seuil ET perp trans < seuil
                    const isAccepted = perpGlobal !== null && perpTrans !== null &&
                                       perpGlobal < threshGlobal && perpTrans < threshTrans;
                    
                    if (isAccepted) {{
                        accepted.push(i);
                    }} else {{
                        rejected.push(i);
                    }}
                }}
                
                // Calculer TP, FP, FN, TN
                let tp = 0, fp = 0, fn = 0, tn = 0;
                for (let i of accepted) {{
                    if (cerNormalized[i] < threshCer) {{
                        tp++; // Bonne page correctement acceptée
                    }} else {{
                        fp++; // Mauvaise page incorrectement acceptée
                    }}
                }}
                for (let i of rejected) {{
                    if (cerNormalized[i] < threshCer) {{
                        fn++; // Bonne page incorrectement rejetée
                    }} else {{
                        tn++; // Mauvaise page correctement rejetée
                    }}
                }}
                
                // Métriques
                const precision = (tp + fp) > 0 ? (tp / (tp + fp) * 100) : 0;
                const recall = (tp + fn) > 0 ? (tp / (tp + fn) * 100) : 0;
                const f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0;
                const acceptRate = (accepted.length / reportData.page_names.length * 100);
                
                document.getElementById('filter-precision').textContent = precision.toFixed(1) + '%';
                document.getElementById('filter-recall').textContent = recall.toFixed(1) + '%';
                document.getElementById('filter-f1').textContent = f1.toFixed(1) + '%';
                document.getElementById('filter-accept-rate').textContent = acceptRate.toFixed(1) + '%';
                
                // CER pour acceptées et rejetées (utiliser la normalisation sélectionnée)
                const rawPageCers = reportData.page_cers_by_norm[normKey] || reportData.page_cers;
                
                const acceptedCers = accepted.map(i => rawPageCers[i]);
                const rejectedCers = rejected.map(i => rawPageCers[i]);
                
                const acceptedMicroCer = acceptedCers.length > 0 ? (acceptedCers.reduce((a,b) => a+b, 0) / acceptedCers.length) : 0;
                const rejectedMicroCer = rejectedCers.length > 0 ? (rejectedCers.reduce((a,b) => a+b, 0) / rejectedCers.length) : 0;
                
                document.getElementById('filter-accepted-cer').textContent = acceptedMicroCer.toFixed(1) + '%';
                document.getElementById('filter-rejected-cer').textContent = rejectedMicroCer.toFixed(1) + '%';
                
                // Graphique taux acceptation par document
                const docAcceptCounts = {{}};
                const docTotalCounts = {{}};
                
                for (let i = 0; i < reportData.page_names.length; i++) {{
                    const doc = reportData.page_to_doc[i];
                    if (!docTotalCounts[doc]) {{
                        docTotalCounts[doc] = 0;
                        docAcceptCounts[doc] = 0;
                    }}
                    docTotalCounts[doc]++;
                    if (accepted.includes(i)) {{
                        docAcceptCounts[doc]++;
                    }}
                }}
                
                const docNamesList = Object.keys(docTotalCounts).sort();
                const docAccepted = docNamesList.map(d => docAcceptCounts[d]);
                const docRejected = docNamesList.map(d => docTotalCounts[d] - docAcceptCounts[d]);
                
                Plotly.react('filter-doc-acceptance', [
                    {{
                        x: docNamesList,
                        y: docAccepted,
                        type: 'bar',
                        name: 'Acceptées',
                        marker: {{ color: '#10B981' }}
                    }},
                    {{
                        x: docNamesList,
                        y: docRejected,
                        type: 'bar',
                        name: 'Rejetées',
                        marker: {{ color: '#EF4444' }}
                    }}
                ], {{
                    barmode: 'stack',
                    xaxis: {{ title: 'Document', tickangle: -45 }},
                    yaxis: {{ title: 'Nombre de pages' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 120 }}
                }}, {{ responsive: true }});
                
                // --- Sous-onglet Acceptées ---
                const acceptedIous = accepted.map(i => reportData.page_ious[i]);
                const acceptedLines = accepted.map(i => reportData.page_lines[i]);
                const acceptedNames = accepted.map(i => reportData.page_names[i]);
                
                document.getElementById('accepted-count').textContent = accepted.length;
                document.getElementById('accepted-micro-cer').textContent = acceptedMicroCer.toFixed(1) + '%';
                document.getElementById('accepted-macro-cer').textContent = acceptedMicroCer.toFixed(1) + '%';
                document.getElementById('accepted-avg-iou').textContent = acceptedIous.length > 0 ? 
                    (acceptedIous.reduce((a,b) => a+b, 0) / acceptedIous.length).toFixed(1) + '%' : '-';
                
                // Distribution CER Acceptées
                Plotly.react('accepted-cer-dist', [{{
                    x: acceptedCers,
                    type: 'histogram',
                    marker: {{ color: '#10B981', opacity: 0.7 }},
                    nbinsx: 20
                }}], {{
                    xaxis: {{ title: 'CER (%)' }},
                    yaxis: {{ title: 'Nombre de pages' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
                // Distribution IoU Acceptées
                Plotly.react('accepted-iou-dist', [{{
                    x: acceptedIous,
                    type: 'histogram',
                    marker: {{ color: '#10B981', opacity: 0.7 }},
                    nbinsx: 20
                }}], {{
                    xaxis: {{ title: 'IoU (%)' }},
                    yaxis: {{ title: 'Nombre de pages' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
                // CER vs Complexité Acceptées
                Plotly.react('accepted-cer-complexity', [{{
                    x: acceptedLines,
                    y: acceptedCers,
                    mode: 'markers',
                    type: 'scatter',
                    text: acceptedNames,
                    marker: {{ color: acceptedCers, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                    hovertemplate: '<b>%{{text}}</b><br>Lignes: %{{x}}<br>CER: %{{y:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'Nombre de lignes' }},
                    yaxis: {{ title: 'CER (%)' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
                // --- Sous-onglet Rejetées ---
                const rejectedIous = rejected.map(i => reportData.page_ious[i]);
                const rejectedLines = rejected.map(i => reportData.page_lines[i]);
                const rejectedNames = rejected.map(i => reportData.page_names[i]);
                
                document.getElementById('rejected-count').textContent = rejected.length;
                document.getElementById('rejected-micro-cer').textContent = rejectedMicroCer.toFixed(1) + '%';
                document.getElementById('rejected-macro-cer').textContent = rejectedMicroCer.toFixed(1) + '%';
                document.getElementById('rejected-avg-iou').textContent = rejectedIous.length > 0 ? 
                    (rejectedIous.reduce((a,b) => a+b, 0) / rejectedIous.length).toFixed(1) + '%' : '-';
                
                // Distribution CER Rejetées
                Plotly.react('rejected-cer-dist', [{{
                    x: rejectedCers,
                    type: 'histogram',
                    marker: {{ color: '#EF4444', opacity: 0.7 }},
                    nbinsx: 20
                }}], {{
                    xaxis: {{ title: 'CER (%)' }},
                    yaxis: {{ title: 'Nombre de pages' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
                // Distribution IoU Rejetées
                Plotly.react('rejected-iou-dist', [{{
                    x: rejectedIous,
                    type: 'histogram',
                    marker: {{ color: '#EF4444', opacity: 0.7 }},
                    nbinsx: 20
                }}], {{
                    xaxis: {{ title: 'IoU (%)' }},
                    yaxis: {{ title: 'Nombre de pages' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
                // CER vs Complexité Rejetées
                Plotly.react('rejected-cer-complexity', [{{
                    x: rejectedLines,
                    y: rejectedCers,
                    mode: 'markers',
                    type: 'scatter',
                    text: rejectedNames,
                    marker: {{ color: rejectedCers, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                    hovertemplate: '<b>%{{text}}</b><br>Lignes: %{{x}}<br>CER: %{{y:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'Nombre de lignes' }},
                    yaxis: {{ title: 'CER (%)' }},
                    margin: {{ l: 50, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
                
            }} catch (error) {{
                console.error("Erreur lors du rafraîchissement du filtrage:", error);
            }}
        }}
        
        // Mise à jour dynamique selon la normalisation et le document
        function refreshDashboard() {{
            try {{
                const normKey = document.getElementById('normSelector').value;
                const docFilter = document.getElementById('docSelector').value;
                
                // --- Définitions globales pour la fonction ---
                // CER par document (utilisé par plusieurs graphiques)
                const docCers = reportData.doc_cers_by_norm[normKey] || reportData.doc_cers;
                
                // 1. Filtrer les données par document
                let filteredIndices = [];
                for (let i = 0; i < reportData.page_names.length; i++) {{
                    if (docFilter === 'all' || reportData.page_to_doc[i] === docFilter) {{
                        filteredIndices.push(i);
                    }}
                }}
                
                // 2. Extraire les CER pour la normalisation choisie
                const rawPageCers = reportData.page_cers_by_norm[normKey] || reportData.page_cers;
                const pageCers = filteredIndices.map(i => rawPageCers[i]);
                const pageLines = filteredIndices.map(i => reportData.page_lines[i]);
                const pageNames = filteredIndices.map(i => reportData.page_names[i]);
                
                // 3. Calculer les métriques du bandeau sur les données filtrées
                const totalErrors = filteredIndices.reduce((sum, idx) => {{
                    return sum + (rawPageCers[idx]/100 * reportData.page_lines[idx]);
                }}, 0);
                const totalChars = filteredIndices.reduce((sum, idx) => sum + reportData.page_lines[idx], 0);
                const microCer = totalChars > 0 ? (totalErrors / totalChars * 100) : 0;
                
                // Macro-CER (moyenne simple du CER des pages filtrées)
                const macroCer = pageCers.length > 0 ? pageCers.reduce((a, b) => a + b, 0) / pageCers.length : 0;
                
                // Biais
                const bias = microCer - macroCer;
                const biasColor = Math.abs(bias) <= 1 ? '#10B981' : (Math.abs(bias) <= 3 ? '#F59E0B' : '#EF4444');
                const biasText = Math.abs(bias) <= 1 ? 'Corpus équilibré' : (Math.abs(bias) <= 3 ? 'Biais modéré' : 'Biais important');
                
                // Mettre à jour les métriques du bandeau
                document.getElementById('metric-micro-cer').textContent = microCer.toFixed(1) + '%';
                document.getElementById('metric-macro-cer').textContent = macroCer.toFixed(1) + '%';
                document.getElementById('metric-bias').textContent = (bias >= 0 ? '+' : '') + bias.toFixed(1) + 'pp';
                document.getElementById('metric-bias').style.color = biasColor;
                document.getElementById('metric-bias-text').textContent = biasText;
                
                // 4. Mettre à jour les graphiques
                
                // Distribution CER
                if (pageCers.length > 0) {{
                    const meanCer = pageCers.reduce((a, b) => a + b, 0) / pageCers.length;
                    const sortedCers = [...pageCers].sort((a, b) => a - b);
                    const medianCer = sortedCers.length % 2 === 0 
                        ? (sortedCers[sortedCers.length/2-1] + sortedCers[sortedCers.length/2]) / 2 
                        : sortedCers[Math.floor(sortedCers.length/2)];
                    Plotly.react('cer-distribution', [{{
                        x: pageCers,
                        type: 'histogram',
                        marker: {{ color: '#4F46E5', opacity: 0.7 }},
                        nbinsx: 30
                    }}], {{
                        xaxis: {{ title: 'CER (%)' }},
                        yaxis: {{ title: 'Nombre de pages' }},
                        shapes: [
                            {{ type: 'line', x0: meanCer, x1: meanCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                            {{ type: 'line', x0: medianCer, x1: medianCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
                        ],
                        annotations: [
                            {{ x: meanCer, y: 1, yref: 'paper', text: 'Moyenne: ' + meanCer.toFixed(1) + '%', showarrow: false, yanchor: 'bottom', font: {{ color: '#EF4444' }} }},
                            {{ x: medianCer, y: 0.9, yref: 'paper', text: 'Médiane: ' + medianCer.toFixed(1) + '%', showarrow: false, yanchor: 'bottom', font: {{ color: '#10B981' }} }}
                        ]
                    }}, {{ responsive: true }});
                }}
                
                // CER vs Complexité avec trendline
                const n = pageLines.length;
                if (n >= 2) {{
                    const sumX = pageLines.reduce((a, b) => a + b, 0);
                    const sumY = pageCers.reduce((a, b) => a + b, 0);
                    const sumXY = pageLines.reduce((acc, xi, i) => acc + xi * pageCers[i], 0);
                    const sumX2 = pageLines.reduce((acc, xi) => acc + xi * xi, 0);
                    const sumY2 = pageCers.reduce((acc, yi) => acc + yi * yi, 0);
                    const denom = n * sumX2 - sumX * sumX;
                    const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
                    const intercept = (sumY - slope * sumX) / n;
                    const xMin = Math.min(...pageLines);
                    const xMax = Math.max(...pageLines);
                    const corrDenom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
                    const correlation = corrDenom > 0 ? (n * sumXY - sumX * sumY) / corrDenom : 0;
                    
                    Plotly.react('cer-complexity', [
                        {{
                            x: pageLines,
                            y: pageCers,
                            mode: 'markers',
                            type: 'scatter',
                            name: 'Pages',
                            marker: {{ color: pageCers, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                            text: pageNames,
                            hovertemplate: '<b>%{{text}}</b><br>Lignes: %{{x}}<br>CER: %{{y:.2f}}%<extra></extra>'
                        }},
                        {{
                            x: [xMin, xMax],
                            y: [slope * xMin + intercept, slope * xMax + intercept],
                            mode: 'lines',
                            type: 'scatter',
                            name: 'Tendance (r=' + correlation.toFixed(2) + ')',
                            line: {{ color: '#6B7280', width: 2, dash: 'dash' }}
                        }}
                    ], {{
                        xaxis: {{ title: 'Nombre de lignes' }},
                        yaxis: {{ title: 'CER (%)', type: scaleStates['cer-complexity'] === false ? 'linear' : 'log', range: scaleStates['cer-complexity'] === false ? [0, 100] : [0, 2] }},
                        margin: {{ l: 60, r: 30, t: 30, b: 50 }},
                        shapes: [
                            {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                            {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
                        ]
                    }}, {{ responsive: true }});
                }}
                
                // CER vs Perplexité (utilise perpTypeStates pour choisir le type)
                const cerPerpType = perpTypeStates['cer-perplexity'];
                const pagePerpsForCer = cerPerpType === 0 
                    ? filteredIndices.map(i => reportData.page_perplexities[i])
                    : (cerPerpType === 1 
                        ? filteredIndices.map(i => reportData.page_perplexities_trans[i])
                        : filteredIndices.map(i => reportData.page_perplexities_seg[i]));
                
                // Collecter toutes les perplexités pour le tooltip détaillé
                const allPagePerps = filteredIndices.map(i => reportData.page_perplexities[i]);
                const allPagePerpsTrans = filteredIndices.map(i => reportData.page_perplexities_trans[i]);
                const allPagePerpsSeg = filteredIndices.map(i => reportData.page_perplexities_seg[i]);
                
                const validPerp = [];
                const validCersPerp = [];
                const validNamesPerp = [];
                const validPerpGlobal = [];
                const validPerpTrans = [];
                const validPerpSeg = [];
                
                for (let i = 0; i < pagePerpsForCer.length; i++) {{
                    if (pagePerpsForCer[i] !== null && pagePerpsForCer[i] !== undefined) {{
                        validPerp.push(pagePerpsForCer[i]);
                        validCersPerp.push(pageCers[i]);
                        validNamesPerp.push(pageNames[i]);
                        validPerpGlobal.push(allPagePerps[i]);
                        validPerpTrans.push(allPagePerpsTrans[i]);
                        validPerpSeg.push(allPagePerpsSeg[i]);
                    }}
                }}
                
                if (validPerp.length >= 2) {{
                    // Calcul trendline pour perplexité
                    const nP = validPerp.length;
                    const sumXP = validPerp.reduce((a, b) => a + b, 0);
                    const sumYP = validCersPerp.reduce((a, b) => a + b, 0);
                    const sumXYP = validPerp.reduce((acc, xi, i) => acc + xi * validCersPerp[i], 0);
                    const sumX2P = validPerp.reduce((acc, xi) => acc + xi * xi, 0);
                    const sumY2P = validCersPerp.reduce((acc, yi) => acc + yi * yi, 0);
                    const denomP = nP * sumX2P - sumXP * sumXP;
                    const slopeP = denomP !== 0 ? (nP * sumXYP - sumXP * sumYP) / denomP : 0;
                    const interceptP = (sumYP - slopeP * sumXP) / nP;
                    const xMinP = Math.min(...validPerp);
                    const xMaxP = Math.max(...validPerp);
                    const corrDenomP = Math.sqrt((nP * sumX2P - sumXP * sumXP) * (nP * sumY2P - sumYP * sumYP));
                    const correlationP = corrDenomP > 0 ? (nP * sumXYP - sumXP * sumYP) / corrDenomP : 0;
                    const perpLabels = ['Perplexité Globale', 'Perplexité Transcription', 'Perplexité Segmentation'];
                    const perpLabel = perpLabels[cerPerpType];
                    
                    // Préparer customdata pour le tooltip détaillé
                    const customData = validPerpGlobal.map((g, idx) => [
                        g !== null ? g.toFixed(2) : 'N/A',
                        validPerpTrans[idx] !== null ? validPerpTrans[idx].toFixed(2) : 'N/A',
                        validPerpSeg[idx] !== null ? validPerpSeg[idx].toFixed(2) : 'N/A'
                    ]);
                    
                    Plotly.react('cer-perplexity', [
                        {{
                            x: validPerp,
                            y: validCersPerp,
                            mode: 'markers',
                            type: 'scatter',
                            name: 'Pages',
                            marker: {{ color: validCersPerp, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                            text: validNamesPerp,
                            customdata: customData,
                            hovertemplate: '<b>%{{text}}</b><br>CER: %{{y:.2f}}%<br>───────────<br>Perp Globale: %{{customdata[0]}}<br>Perp Trans: %{{customdata[1]}}<br>Perp Seg: %{{customdata[2]}}<extra></extra>'
                        }},
                        {{
                            x: [xMinP, xMaxP],
                            y: [slopeP * xMinP + interceptP, slopeP * xMaxP + interceptP],
                            mode: 'lines',
                            type: 'scatter',
                            name: 'Tendance (r=' + correlationP.toFixed(2) + ')',
                            line: {{ color: '#6B7280', width: 2, dash: 'dash' }}
                        }}
                    ], {{
                        xaxis: {{ title: perpLabel }},
                        yaxis: {{ title: 'CER (%)', type: scaleStates['cer-perplexity'] === false ? 'linear' : 'log', range: scaleStates['cer-perplexity'] === false ? [0, 100] : [0, 2] }},
                        margin: {{ l: 60, r: 30, t: 30, b: 50 }},
                        shapes: [
                            {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                            {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
                        ]
                    }}, {{ responsive: true }});
                }}
                
                // Perplexity vs Training Pages (Page Level) - avec toggle 3-way
                const pagePerpType = perpTypeStates['perp-vs-training-page'];
                const pagePerpsGlobal = filteredIndices.map(i => reportData.page_perplexities[i]);
                const pagePerpsTrainSel = pagePerpType === 0 
                    ? pagePerpsGlobal
                    : (pagePerpType === 1 
                        ? filteredIndices.map(i => reportData.page_perplexities_trans[i])
                        : filteredIndices.map(i => reportData.page_perplexities_seg[i]));
                
                const pageTrainCount = filteredIndices.map(i => reportData.page_training_count[i]);
                const validTrainPageIndices = [];
                for(let i=0; i<pagePerpsTrainSel.length; i++) {{
                    if(pagePerpsTrainSel[i] !== null && pagePerpsTrainSel[i] !== undefined) {{
                        validTrainPageIndices.push(i);
                    }}
                }}
                
                if (validTrainPageIndices.length > 0) {{
                    const x = validTrainPageIndices.map(i => pageTrainCount[i]);
                    const y = validTrainPageIndices.map(i => pagePerpsTrainSel[i]);
                    const c = validTrainPageIndices.map(i => pageCers[i]);
                    const t = validTrainPageIndices.map(i => pageNames[i]);
                    const perpLabels = ['Globale', 'Transcription', 'Segmentation'];
                    const yLabel = 'Perplexité ' + perpLabels[pagePerpType];
                    
                    Plotly.react('perp-vs-training-page', [{{
                        x: x,
                        y: y,
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Pages',
                        marker: {{
                            color: c,
                            colorscale: 'RdYlGn_r',
                            size: 6,
                            opacity: 0.6,
                            colorbar: {{ title: 'CER (%)', len: 0.5 }}
                        }},
                        text: t,
                        hovertemplate: '<b>%{{text}}</b><br>Training Pages: %{{x}}<br>' + yLabel + ': %{{y:.2f}}<br>CER: %{{marker.color:.2f}}%<extra></extra>'
                    }}], {{
                        xaxis: {{ title: "Pages d'entraînement (Document)" }},
                        yaxis: {{ title: yLabel }},
                        margin: {{ l: 50, r: 20, t: 20, b: 40 }}
                    }}, {{ responsive: true }});
                }}

                // Perplexity vs Training Pages (Document Level) - avec toggle 3-way
                const docPerpType = perpTypeStates['perp-vs-training-doc'];
                const docPerpsGlobal = reportData.doc_perplexities;
                const docPerpsSel = docPerpType === 0 
                    ? docPerpsGlobal
                    : (docPerpType === 1 
                        ? reportData.doc_perplexities_trans
                        : reportData.doc_perplexities_seg);
                const docTrain = reportData.doc_training_count;
                
                const validDocTrainIndices = [];
                for (let i=0; i<docPerpsSel.length; i++) {{
                    if (docPerpsSel[i] !== null && docPerpsSel[i] !== undefined) {{
                        validDocTrainIndices.push(i);
                    }}
                }}
                
                if (validDocTrainIndices.length > 0) {{
                    const x = validDocTrainIndices.map(i => docTrain[i]);
                    const y = validDocTrainIndices.map(i => docPerpsSel[i]);
                    const c = validDocTrainIndices.map(i => docCers[i]);
                    const t = validDocTrainIndices.map(i => reportData.doc_names[i]);
                    const perpLabels = ['Globale', 'Transcription', 'Segmentation'];
                    const yLabel = 'Perplexité ' + perpLabels[docPerpType];

                    Plotly.react('perp-vs-training-doc', [{{
                        x: x,
                        y: y,
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Documents',
                        marker: {{
                            color: c,
                            colorscale: 'RdYlGn_r',
                            size: 10,
                            colorbar: {{ title: 'CER (%)', len: 0.5 }},
                            line: {{ color: 'white', width: 1 }}
                        }},
                        text: t,
                        hovertemplate: '<b>%{{text}}</b><br>Training Pages: %{{x}}<br>' + yLabel + ': %{{y:.2f}}<br>Avg CER: %{{marker.color:.2f}}%<extra></extra>'
                    }}], {{
                        xaxis: {{ title: "Pages d'entraînement" }},
                        yaxis: {{ title: yLabel }},
                        margin: {{ l: 50, r: 20, t: 20, b: 40 }}
                    }}, {{ responsive: true }});
                }}

                // IoU vs Perplexité (avec toggle Global/Segmentation)
                const iouPerpType = perpTypeStates['iou-perplexity'];
                const pagePerpsForIou = iouPerpType === 0 
                    ? filteredIndices.map(i => reportData.page_perplexities[i])
                    : filteredIndices.map(i => reportData.page_perplexities_seg[i]);
                const pageIous = filteredIndices.map(i => reportData.page_ious[i]);
                
                const validIouPerps = [];
                const validIous = [];
                const validIouNames = [];
                for (let i = 0; i < pagePerpsForIou.length; i++) {{
                    if (pagePerpsForIou[i] !== null && pagePerpsForIou[i] !== undefined) {{
                        validIouPerps.push(pagePerpsForIou[i]);
                        validIous.push(pageIous[i]);
                        validIouNames.push(pageNames[i]);
                    }}
                }}
                
                if (validIouPerps.length >= 2) {{
                    const iouPerpLabel = iouPerpType === 0 ? 'Perplexité Globale' : 'Perplexité Segmentation';
                    
                    Plotly.react('iou-perplexity', [{{
                        x: validIouPerps,
                        y: validIous,
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Pages',
                        marker: {{ color: validIous, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                        text: validIouNames,
                        hovertemplate: '<b>%{{text}}</b><br>' + iouPerpLabel + ': %{{x:.2f}}<br>IoU: %{{y:.2f}}%<extra></extra>'
                    }}], {{
                        xaxis: {{ title: iouPerpLabel }},
                        yaxis: {{ title: 'IoU (%)', type: scaleStates['iou-perplexity'] === false ? 'linear' : 'log', range: scaleStates['iou-perplexity'] === false ? [0, 100] : [1, 2.1] }},
                        margin: {{ l: 60, r: 30, t: 30, b: 50 }}
                    }}, {{ responsive: true }});
                }}

                
                // CER vs Erreurs de Segmentation (par page)
                const pageSegErrors = filteredIndices.map(i => reportData.page_seg_errors[i]);
                Plotly.react('cer-seg-page', [{{
                    x: pageSegErrors,
                    y: pageCers,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Pages',
                    marker: {{ color: pageCers, colorscale: 'RdYlGn', size: 8, opacity: 0.6 }},
                    text: pageNames,
                    hovertemplate: '<b>%{{text}}</b><br>Seg Error: %{{x:.1f}}%<br>CER: %{{y:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'Taux erreur segmentation (%)' }},
                    yaxis: {{ title: 'CER (%)', type: scaleStates['cer-seg-page'] === false ? 'linear' : 'log', range: scaleStates['cer-seg-page'] === false ? [0, 100] : [0, 2] }},
                    margin: {{ l: 60, r: 30, t: 30, b: 50 }},
                    shapes: [
                        {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: {{ color: '#10B981', width: 2, dash: 'dash' }} }},
                        {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10, y1: 10, line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }}
                    ]
                }}, {{ responsive: true }});
                
                // CER par document
                const meanDocCer = docCers.reduce((a, b) => a + b, 0) / docCers.length;
                const sortedDocCers = [...docCers].sort((a, b) => a - b);
                const medianDocCer = sortedDocCers[Math.floor(sortedDocCers.length / 2)];
                
                Plotly.react('cer-by-document', [{{
                    y: reportData.doc_names,
                    x: docCers,
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: docCers, colorscale: 'RdYlGn' }},
                    hovertemplate: '<b>%{{y}}</b><br>CER: %{{x:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'CER (%)', type: scaleStates['cer-by-document'] === false ? 'linear' : 'log', range: scaleStates['cer-by-document'] === false ? [0, 100] : [0, 2] }},
                    margin: {{ l: 180, r: 30, t: 20, b: 40 }},
                    height: Math.max(450, reportData.doc_names.length * 30),
                    autosize: true,
                    shapes: [
                        {{ type: 'line', x0: meanDocCer, x1: meanDocCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                        {{ type: 'line', x0: medianDocCer, x1: medianDocCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
                    ]
                }}, {{ responsive: true }});
                
                // CER vs Segmentation
                Plotly.react('cer-vs-segmentation', [{{
                    x: reportData.doc_seg_errors,
                    y: docCers,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Documents',
                    marker: {{ 
                        color: docCers, colorscale: 'RdYlGn',
                        size: reportData.doc_sizes, sizemode: 'area',
                        sizeref: 2.*Math.max(...reportData.doc_sizes)/(40.**2), sizemin: 4
                    }},
                    text: reportData.doc_names,
                    hovertemplate: '<b>%{{text}}</b><br>Seg Error: %{{x:.1f}}%<br>CER: %{{y:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'Taux erreur segmentation (%)' }},
                    yaxis: {{ title: 'CER (%)', type: scaleStates['cer-vs-segmentation'] === false ? 'linear' : 'log', range: scaleStates['cer-vs-segmentation'] === false ? [0, 100] : [0, 2] }},
                    margin: {{ l: 60, r: 30, t: 30, b: 50 }},
                    height: 350, autosize: true
                }}, {{ responsive: true }});
                
                // CER par Document (normalisation sélectionnée)
                Plotly.react('cer-doc-selected-norm', [{{
                    y: reportData.doc_names,
                    x: docCers,
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: docCers, colorscale: 'RdYlGn' }},
                    hovertemplate: '<b>%{{y}}</b><br>CER: %{{x:.2f}}%<extra></extra>'
                }}], {{
                    xaxis: {{ title: 'CER (%)', type: scaleStates['cer-doc-selected-norm'] === false ? 'linear' : 'log', range: scaleStates['cer-doc-selected-norm'] === false ? [0, 100] : [0, 2] }},
                    margin: {{ l: 180, r: 30, t: 20, b: 40 }},
                    height: Math.max(500, reportData.doc_names.length * 35),
                    autosize: true,
                    shapes: [
                        {{ type: 'line', x0: meanDocCer, x1: meanDocCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                        {{ type: 'line', x0: medianDocCer, x1: medianDocCer, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#10B981', width: 2, dash: 'dot' }} }}
                    ]
                }}, {{ responsive: true }});
            
            }} catch (error) {{
                console.error("Erreur lors du rafraîchissement du dashboard:", error);
                document.getElementById('js-error-alert').classList.remove('d-none');
                document.getElementById('js-error-message').textContent = "Update Error: " + error.name + ": " + error.message;
            }}
        }}
    </script>
</body>
</html>
"""


def generate_html_report(
    all_metrics: List[Dict],
    document_stats: Dict[str, Dict],
    error_stats: Optional[Dict],
    page_cer_variants: Dict[str, float],
    doc_cer_variants: Dict[str, float],
    training_pages_per_doc: Dict[str, int] = None,
    output_dir: str = ".",
    report_name: str = "evaluation_report"
) -> str:
    """
    Génère un rapport HTML interactif avec Plotly.
    
    Returns:
        Chemin vers le fichier HTML généré
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_pages_per_doc = training_pages_per_doc or {}
    
    # Calculer les métriques
    total_edit_dist = sum(m['total_edit_distance'] for m in all_metrics)
    total_gt_chars = sum(m['total_gt_chars'] for m in all_metrics)
    micro_cer = (total_edit_dist / total_gt_chars * 100) if total_gt_chars > 0 else 100.0
    
    doc_cers_list = [stats['avg_cer'] * 100 for stats in document_stats.values()]
    macro_cer = np.mean(doc_cers_list) if doc_cers_list else 100.0
    
    cer_bias = micro_cer - macro_cer
    bias_color = "#10B981" if abs(cer_bias) <= 1 else ("#F59E0B" if abs(cer_bias) <= 3 else "#EF4444")
    bias_interpretation = "Corpus équilibré" if abs(cer_bias) <= 1 else ("Biais modéré" if abs(cer_bias) <= 3 else "Biais important")
    
    avg_iou = np.mean([m['iou'] * 100 for m in all_metrics])
    median_iou = np.median([m['iou'] * 100 for m in all_metrics])
    avg_precision = np.mean([m['precision'] * 100 for m in all_metrics])
    avg_recall = np.mean([m['recall'] * 100 for m in all_metrics])
    avg_format = np.mean([m['format_score'] * 100 for m in all_metrics])
    mean_cer = np.mean([m['cer'] * 100 for m in all_metrics])
    median_cer = np.median([m['cer'] * 100 for m in all_metrics])  # Micro-median
    macro_median_cer = np.median(doc_cers_list) if doc_cers_list else 100.0  # Macro-median
    
    # Préparer les données pour les graphiques
    # Importer get_document_name pour mapping
    from .metrics import get_document_name
    
    page_cers = [m['cer'] * 100 for m in all_metrics]
    page_ious = [m['iou'] * 100 for m in all_metrics]
    page_lines = [m['num_ground_truth'] for m in all_metrics]
    page_names = [m['name'] for m in all_metrics]
    page_doc_names = [get_document_name(m['name']) for m in all_metrics]
    page_perplexities = [m.get('perplexity') for m in all_metrics]  # Peut contenir des None
    page_perplexities_trans = [m.get('perplexity_transcription') for m in all_metrics]
    page_perplexities_seg = [m.get('perplexity_segmentation') for m in all_metrics]
    page_seg_errors = [(1 - m['recall']) * 100 for m in all_metrics]  # Erreurs de segmentation par page
    page_training_count = [training_pages_per_doc.get(dname, 0) for dname in page_doc_names]
    
    # Données par document (triées par CER)
    sorted_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'])
    doc_names = [name for name, _ in sorted_docs]
    doc_cers = [stats['avg_cer'] * 100 for _, stats in sorted_docs]
    doc_ious = [stats['avg_iou'] * 100 for _, stats in sorted_docs]
    doc_seg_errors = [stats.get('segmentation_error_rate', 0) * 100 for _, stats in sorted_docs]
    doc_sizes = [stats['total_ground_truth'] for _, stats in sorted_docs]
    doc_training_count = [training_pages_per_doc.get(doc, 0) for doc in doc_names]
    
    # Doc perplexity (moyenne des pages) - global, transcription, segmentation
    doc_perplexities = []
    doc_perplexities_trans = []
    doc_perplexities_seg = []
    for doc in doc_names:
        # Global perplexity
        doc_pages_perp = [m.get('perplexity') for m in all_metrics if get_document_name(m['name']) == doc and m.get('perplexity') is not None]
        if doc_pages_perp:
            doc_perplexities.append(sum(doc_pages_perp) / len(doc_pages_perp))
        else:
            doc_perplexities.append(None)
        
        # Transcription perplexity
        doc_pages_trans = [m.get('perplexity_transcription') for m in all_metrics if get_document_name(m['name']) == doc and m.get('perplexity_transcription') is not None]
        if doc_pages_trans:
            doc_perplexities_trans.append(sum(doc_pages_trans) / len(doc_pages_trans))
        else:
            doc_perplexities_trans.append(None)
        
        # Segmentation perplexity
        doc_pages_seg = [m.get('perplexity_segmentation') for m in all_metrics if get_document_name(m['name']) == doc and m.get('perplexity_segmentation') is not None]
        if doc_pages_seg:
            doc_perplexities_seg.append(sum(doc_pages_seg) / len(doc_pages_seg))
        else:
            doc_perplexities_seg.append(None)

    # Variantes CER
    variant_labels = ['Base', 'Sans accents', 'Minuscules', 'Chars norm.', 'Sans ponct.', 'Abbr. hist.', 'Normalisé']
    variant_keys = ['no_accents', 'lowercase', 'normalized_chars', 'no_punctuation', 'historical_abbrev', 'normalized']
    cer_variant_values = [micro_cer] + [page_cer_variants.get(k, 0) * 100 for k in variant_keys]
    doc_cer_variant_values = [macro_cer] + [doc_cer_variants.get(k, 0) * 100 for k in variant_keys]
    
    # Tableau des documents
    doc_table_rows = ""
    for doc_name, stats in sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'], reverse=True):
        cer = stats['avg_cer'] * 100
        cer_class = 'text-success' if cer < 5 else ('text-warning' if cer < 15 else 'text-danger')
        train_pages = training_pages_per_doc.get(doc_name, 0)
        doc_table_rows += f"""<tr>
            <td>{doc_name}</td>
            <td class="{cer_class}">{cer:.2f}%</td>
            <td>{stats['avg_iou']*100:.1f}%</td>
            <td>{stats['avg_recall']*100:.1f}%</td>
            <td>{stats['num_pages']}</td>
            <td>{train_pages}</td>
        </tr>"""
    
    # Statistiques d'erreurs
    error_stats_html = "<p class='text-muted'>Pas de données d'erreurs disponibles</p>"
    confusions_raw_html = ""
    confusions_norm_html = ""
    error_totals = None
    
    if error_stats:
        totals = error_stats['totals']
        error_totals = totals
        line_stats = error_stats['line_stats']
        error_stats_html = f"""
        <table class="table table-sm">
            <tr><td>Total erreurs</td><td><strong>{totals['total_errors']}</strong></td></tr>
            <tr><td>Insertions</td><td>{totals['insertions']} ({totals['insertions']/totals['total_errors']*100:.1f}%)</td></tr>
            <tr><td>Suppressions</td><td>{totals['deletions']} ({totals['deletions']/totals['total_errors']*100:.1f}%)</td></tr>
            <tr><td>Substitutions</td><td>{totals['substitutions']} ({totals['substitutions']/totals['total_errors']*100:.1f}%)</td></tr>
            <tr><td>Lignes manquantes</td><td>{line_stats['missing_lines']}</td></tr>
            <tr><td>Lignes extras</td><td>{line_stats['extra_lines']}</td></tr>
        </table>
        """
        
        # Confusions brutes
        if error_stats.get('top_confusions'):
            confusions_raw_html = "<table class='table table-sm'><thead><tr><th>GT</th><th>→</th><th>Pred</th><th>Count</th></tr></thead><tbody>"
            for conf in error_stats['top_confusions'][:20]:
                confusions_raw_html += f"<tr><td><code>{conf['gt']}</code></td><td>→</td><td><code>{conf['pred']}</code></td><td>{conf['count']}</td></tr>"
            confusions_raw_html += "</tbody></table>"
        
        # Confusions normalisées
        if error_stats.get('top_confusions_normalized'):
            confusions_norm_html = "<table class='table table-sm'><thead><tr><th>GT</th><th>→</th><th>Pred</th><th>Count</th></tr></thead><tbody>"
            for conf in error_stats['top_confusions_normalized'][:20]:
                confusions_norm_html += f"<tr><td><code>{conf['gt']}</code></td><td>→</td><td><code>{conf['pred']}</code></td><td>{conf['count']}</td></tr>"
            confusions_norm_html += "</tbody></table>"
    
    # Données JSON embarquées avec CER par niveau de normalisation
    norm_keys = ['base', 'no_accents', 'lowercase', 'normalized_chars', 'no_punctuation', 'historical_abbrev', 'normalized']
    
    # CER par page pour chaque normalisation
    page_cers_by_norm = {'base': page_cers}
    for norm_key in norm_keys[1:]:  # Skip 'base'
        cer_key = f'cer_{norm_key}'
        page_cers_by_norm[norm_key] = [m.get(cer_key, m['cer']) * 100 for m in all_metrics]
    
    # Recalculer doc_cers pour chaque normalisation
    doc_cers_by_norm = {}
    for norm_key in norm_keys:
        cer_key = 'cer' if norm_key == 'base' else f'cer_{norm_key}'
        # Calculer CER moyen par document pour cette normalisation
        doc_stats_for_norm = {}
        for m in all_metrics:
            doc_name = get_document_name(m['name'])  # Utiliser la même fonction que metrics.py
            if doc_name not in doc_stats_for_norm:
                doc_stats_for_norm[doc_name] = {'total_edit': 0, 'total_chars': 0}
            edit_key = 'total_edit_distance' if norm_key == 'base' else f'total_edit_distance_{norm_key}'
            chars_key = 'total_gt_chars' if norm_key == 'base' else f'total_gt_chars_{norm_key}'
            doc_stats_for_norm[doc_name]['total_edit'] += m.get(edit_key, m['total_edit_distance'])
            doc_stats_for_norm[doc_name]['total_chars'] += m.get(chars_key, m['total_gt_chars'])
        
        doc_cers_by_norm[norm_key] = [
            (doc_stats_for_norm[name]['total_edit'] / doc_stats_for_norm[name]['total_chars'] * 100)
            if name in doc_stats_for_norm and doc_stats_for_norm[name]['total_chars'] > 0 else 100.0
            for name in doc_names
        ]
    
    # Mapping page vers document pour filtrage JS
    page_to_doc = [get_document_name(m['name']) for m in all_metrics]
    
    # Options pour le sélecteur de document
    doc_options = ""
    for name in sorted(doc_names):
        doc_options += f'<option value="{name}">{name}</option>\n'
    
    json_data = {
        'summary': {
            'micro_cer': micro_cer,
            'macro_cer': macro_cer,
            'avg_iou': avg_iou,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'num_pages': len(all_metrics),
            'num_documents': len(document_stats)
        },
        'page_to_doc': page_to_doc,
        'page_cers': page_cers,
        'page_cers_by_norm': page_cers_by_norm,
        'doc_cers_by_norm': doc_cers_by_norm,
        'page_ious': page_ious,
        'page_lines': page_lines,
        'page_names': page_names,
        'page_perplexities': page_perplexities,
        'page_perplexities_trans': page_perplexities_trans,
        'page_perplexities_seg': page_perplexities_seg,
        'page_seg_errors': page_seg_errors,
        'page_training_count': page_training_count,
        'doc_names': doc_names,
        'doc_cers': doc_cers,
        'doc_ious': doc_ious,
        'doc_seg_errors': doc_seg_errors,
        'doc_sizes': doc_sizes,
        'doc_training_count': doc_training_count,
        'doc_perplexities': doc_perplexities,
        'doc_perplexities_trans': doc_perplexities_trans,
        'doc_perplexities_seg': doc_perplexities_seg,
        'error_totals': error_totals
    }
    
    # Générer le HTML
    accuracy = 100 - micro_cer  # Pré-calculer pour éviter problème de format
    html_content = HTML_TEMPLATE.format(
        timestamp=timestamp,
        micro_cer=micro_cer,
        macro_cer=macro_cer,
        cer_bias=cer_bias,
        bias_color=bias_color,
        bias_interpretation=bias_interpretation,
        avg_iou=avg_iou,
        median_iou=median_iou,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_format=avg_format,
        mean_cer=mean_cer,
        median_cer=median_cer,
        accuracy=accuracy,
        doc_table_rows=doc_table_rows,
        error_stats_html=error_stats_html,
        doc_options=doc_options,
        confusions_raw_html=confusions_raw_html,
        confusions_norm_html=confusions_norm_html,
        json_data=json.dumps(json_data),
        cer_variant_labels=json.dumps(variant_labels),
        cer_variant_values=json.dumps(cer_variant_values),
        doc_cer_variant_values=json.dumps(doc_cer_variant_values)
    )
    
    # Sauvegarder
    html_path = os.path.join(output_dir, f"{report_name}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Rapport HTML interactif généré: {html_path}")
    return html_path
