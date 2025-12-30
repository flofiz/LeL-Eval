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
            max-height: 400px;
            overflow: hidden;
        }}
        .card-body {{
            overflow: hidden;
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
        <div class="container">
            <span class="navbar-brand mb-0 h1">📊 Rapport d'Évaluation OCR</span>
            <span class="text-light">{timestamp}</span>
        </div>
    </nav>

    <div class="container">
        <!-- Métriques principales -->
        <div class="row g-4 mb-4">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{micro_cer:.1f}%</div>
                    <div class="metric-label">Micro-CER</div>
                    <small class="text-muted">Pondéré par caractères</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{macro_cer:.1f}%</div>
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
                    <div class="metric-value" style="color: {bias_color};">{cer_bias:+.1f}pp</div>
                    <div class="metric-label">Biais Corpus</div>
                    <small class="text-muted">{bias_interpretation}</small>
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
        </ul>

        <div class="tab-content">
            <!-- Onglet Vue d'ensemble -->
            <div class="tab-pane fade show active" id="overview">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Distribution du CER par Page</div>
                            <div class="card-body">
                                <div id="cer-distribution" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">CER par Variante de Normalisation</div>
                            <div class="card-body">
                                <div id="cer-variants" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Distribution de l'IoU</div>
                            <div class="card-body">
                                <div id="iou-distribution" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">CER vs Complexité (nb lignes)</div>
                            <div class="card-body">
                                <div id="cer-complexity" class="plotly-graph"></div>
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
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">IoU par Document</div>
                            <div class="card-body">
                                <div id="iou-by-document" style="width:100%; height:500px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
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
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Variantes CER - Niveau Page vs Document</div>
                            <div class="card-body">
                                <div id="cer-variants-comparison" class="plotly-graph"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
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
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Données embarquées
        const reportData = {json_data};
        
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
            shapes: [{{
                type: 'line', x0: {mean_cer}, x1: {mean_cer}, y0: 0, y1: 1, yref: 'paper',
                line: {{ color: '#EF4444', width: 2, dash: 'dash' }}
            }}],
            annotations: [{{
                x: {mean_cer}, y: 1, yref: 'paper', text: 'Moyenne: {mean_cer:.1f}%',
                showarrow: false, yanchor: 'bottom'
            }}]
        }}, {{ responsive: true }});
        
        // Distribution IoU
        Plotly.newPlot('iou-distribution', [{{
            x: reportData.page_ious,
            type: 'histogram',
            marker: {{ color: '#10B981', opacity: 0.7 }},
            nbinsx: 30
        }}], {{
            xaxis: {{ title: 'IoU (%)' }},
            yaxis: {{ title: 'Nombre de pages' }}
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
            marker: {{ 
                color: reportData.page_cers, 
                colorscale: 'RdYlGn', 
                reversescale: true,
                size: 8,
                opacity: 0.6
            }},
            text: reportData.page_names,
            hovertemplate: '<b>%{{text}}</b><br>Lignes: %{{x}}<br>CER: %{{y:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'Nombre de lignes' }},
            yaxis: {{ title: 'CER (%)' }}
        }}, {{ responsive: true }});
        
        // CER par document (bar chart horizontal)
        Plotly.newPlot('cer-by-document', [{{
            y: reportData.doc_names,
            x: reportData.doc_cers,
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: reportData.doc_cers,
                colorscale: 'RdYlGn',
                reversescale: true
            }},
            hovertemplate: '<b>%{{y}}</b><br>CER: %{{x:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'CER (%)' }},
            margin: {{ l: 200 }}
        }}, {{ responsive: true }});
        
        // IoU par document
        Plotly.newPlot('iou-by-document', [{{
            y: reportData.doc_names,
            x: reportData.doc_ious,
            type: 'bar',
            orientation: 'h',
            marker: {{ 
                color: reportData.doc_ious,
                colorscale: 'RdYlGn'
            }}
        }}], {{
            xaxis: {{ title: 'IoU (%)' }},
            margin: {{ l: 150, r: 20, t: 20, b: 40 }},
            height: 450,
            autosize: true
        }}, {{ responsive: true }});
        
        // CER vs Segmentation
        Plotly.newPlot('cer-vs-segmentation', [{{
            x: reportData.doc_seg_errors,
            y: reportData.doc_cers,
            mode: 'markers',
            type: 'scatter',
            marker: {{ 
                color: reportData.doc_cers, 
                colorscale: 'RdYlGn', 
                reversescale: true,
                size: reportData.doc_sizes,
                sizemode: 'area',
                sizeref: 2.*Math.max(...reportData.doc_sizes)/(40.**2),
                sizemin: 4
            }},
            text: reportData.doc_names,
            hovertemplate: '<b>%{{text}}</b><br>Seg Error: %{{x:.1f}}%<br>CER: %{{y:.2f}}%<extra></extra>'
        }}], {{
            xaxis: {{ title: 'Taux erreur segmentation (%)' }},
            yaxis: {{ title: 'CER (%)' }},
            margin: {{ l: 60, r: 20, t: 20, b: 50 }},
            height: 350,
            autosize: true
        }}, {{ responsive: true }});
        
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
        
        // Download JSON
        function downloadJSON() {{
            const blob = new Blob([JSON.stringify(reportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'evaluation_data.json';
            a.click();
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
    output_dir: str = ".",
    report_name: str = "evaluation_report"
) -> str:
    """
    Génère un rapport HTML interactif avec Plotly.
    
    Returns:
        Chemin vers le fichier HTML généré
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
    avg_precision = np.mean([m['precision'] * 100 for m in all_metrics])
    avg_recall = np.mean([m['recall'] * 100 for m in all_metrics])
    avg_format = np.mean([m['format_score'] * 100 for m in all_metrics])
    mean_cer = np.mean([m['cer'] * 100 for m in all_metrics])
    
    # Préparer les données pour les graphiques
    page_cers = [m['cer'] * 100 for m in all_metrics]
    page_ious = [m['iou'] * 100 for m in all_metrics]
    page_lines = [m['num_ground_truth'] for m in all_metrics]
    page_names = [m['name'] for m in all_metrics]
    
    # Données par document (triées par CER)
    sorted_docs = sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'])
    doc_names = [name for name, _ in sorted_docs]
    doc_cers = [stats['avg_cer'] * 100 for _, stats in sorted_docs]
    doc_ious = [stats['avg_iou'] * 100 for _, stats in sorted_docs]
    doc_seg_errors = [stats.get('segmentation_error_rate', 0) * 100 for _, stats in sorted_docs]
    doc_sizes = [stats['total_ground_truth'] for _, stats in sorted_docs]
    
    # Variantes CER
    variant_labels = ['Base', 'Sans accents', 'Minuscules', 'Chars norm.', 'Sans ponct.', 'Normalisé']
    variant_keys = ['no_accents', 'lowercase', 'normalized_chars', 'no_punctuation', 'normalized']
    cer_variant_values = [micro_cer] + [page_cer_variants.get(k, 0) * 100 for k in variant_keys]
    doc_cer_variant_values = [macro_cer] + [doc_cer_variants.get(k, 0) * 100 for k in variant_keys]
    
    # Tableau des documents
    doc_table_rows = ""
    for doc_name, stats in sorted(document_stats.items(), key=lambda x: x[1]['avg_cer'], reverse=True):
        cer = stats['avg_cer'] * 100
        cer_class = 'text-success' if cer < 5 else ('text-warning' if cer < 15 else 'text-danger')
        doc_table_rows += f"""<tr>
            <td>{doc_name}</td>
            <td class="{cer_class}">{cer:.2f}%</td>
            <td>{stats['avg_iou']*100:.1f}%</td>
            <td>{stats['avg_recall']*100:.1f}%</td>
            <td>{stats['num_pages']}</td>
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
    
    # Données JSON embarquées
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
        'page_cers': page_cers,
        'page_ious': page_ious,
        'page_lines': page_lines,
        'page_names': page_names,
        'doc_names': doc_names,
        'doc_cers': doc_cers,
        'doc_ious': doc_ious,
        'doc_seg_errors': doc_seg_errors,
        'doc_sizes': doc_sizes,
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
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_format=avg_format,
        mean_cer=mean_cer,
        accuracy=accuracy,
        doc_table_rows=doc_table_rows,
        error_stats_html=error_stats_html,
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
