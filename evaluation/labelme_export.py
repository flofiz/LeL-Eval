"""
Export au format LabelMe pour visualisation.

Ce module fournit les fonctions pour convertir les résultats
au format LabelMe, permettant la visualisation des bounding boxes.
"""

import base64
import json
from io import BytesIO
from typing import Dict, Tuple

from PIL import Image

from .models import BBox
from .tsv_parser import parse_tsv_line, bbox_to_4points


def get_image_dimensions(base64_image: str) -> Tuple[int, int]:
    """
    Récupère les dimensions d'une image base64.
    
    Args:
        base64_image: Image encodée en base64
        
    Returns:
        Tuple (width, height)
    """
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
