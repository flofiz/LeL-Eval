import sys
import os
import json
from unittest.mock import MagicMock

# Setup path
sys.path.append('/home/florian/Documents/LettresEnLumieres/Scripts/Eval_Scripts')

# Mock editdistance
sys.modules['editdistance'] = MagicMock()
sys.modules['editdistance'].eval.return_value = 0

try:
    from evaluation.html_report import generate_html_report
except ImportError:
    print("Error importing modules")
    sys.exit(1)

# Mock Data
all_metrics = [
    {'name': 'DocA_Page1', 'cer': 0.1, 'iou': 0.9, 'precision': 1.0, 'recall': 1.0, 'format_score': 1.0, 'perplexity': 0.5, 'total_edit_distance': 10, 'total_gt_chars': 100, 'num_ground_truth': 10, 'num_matched': 10, 'num_predicted': 10},
]
document_stats = {
    'DocA': {'avg_cer': 0.15, 'avg_iou': 0.85, 'avg_precision': 0.85, 'avg_recall': 0.85, 'avg_format_score': 1.0, 'num_pages': 1, 'total_ground_truth': 100, 'total_matched': 10, 'total_predicted': 10, 'missing_lines': 0, 'extra_lines': 0, 'segmentation_error_rate': 0.0, 'pages': ['DocA_Page1'], 'avg_cer_base': 0.15, 'total_edit_distance_base': 30, 'total_gt_chars_base': 200}
}
training_pages = {'DocA': 5}

output_path = generate_html_report(all_metrics, document_stats, None, {}, {}, training_pages_per_doc=training_pages, output_dir=".", report_name="debug_refresh_test_2")
print(f"Generated: {output_path}")
