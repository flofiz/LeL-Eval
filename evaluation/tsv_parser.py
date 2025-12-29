"""
Parsing et manipulation des données TSV.

Ce module gère le parsing des fichiers TSV contenant les transcriptions
et les coordonnées des bounding boxes, ainsi que leur conversion.
"""

import re
import numpy as np
from typing import List

from .models import BBox


def parse_tsv_line(line: str) -> BBox:
    """
    Parse une ligne TSV au format: transcription\txmin\tymin\txmax\tymax\tangle
    
    Args:
        line: Ligne TSV à parser
        
    Returns:
        BBox: Objet bounding box avec le texte et les coordonnées
        
    Raises:
        ValueError: Si le format TSV est invalide
    """
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
    
    Args:
        bbox: Bounding box à convertir
        
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
      
    Args:
        response: Réponse du modèle contenant le TSV
        
    Returns:
        Contenu TSV extrait (sans les balises)
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
