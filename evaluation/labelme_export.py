"""
Export au format LabelMe pour visualisation.

Ce module fournit les fonctions pour convertir les résultats
au format LabelMe, permettant la visualisation des bounding boxes.
"""

import base64
import math
from typing import Dict, Tuple, List, Optional

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


def tsv_to_labelme(tsv_content: str, 
                   image_base64: str, 
                   image_name: str,
                   logprobs: Optional[List[Dict]] = None,
                   full_response_text: Optional[str] = None) -> Dict:
    """
    Convertit un TSV au format LabelMe pour visualisation.
    
    Args:
        tsv_content: Contenu TSV avec les bounding boxes
        image_base64: Image encodée en base64
        image_name: Nom de l'image
        logprobs: Liste des logprobs pour chaque token (optionnel)
        full_response_text: Texte complet de la réponse pour alignement (optionnel)
    
    Returns:
        Dictionnaire au format LabelMe
    """
    # Récupérer les dimensions de l'image
    width, height = get_image_dimensions(image_base64)
    
    # Parser les bounding boxes
    shapes = []
    parsing_errors = []
    has_parsing_error = False
    
    # Préparer le mapping tokens -> char positions si logprobs disponibles
    token_char_map = []
    response_cursor = 0
    
    if logprobs and full_response_text:
        # Reconstruire le texte depuis les tokens pour vérification
        # VLLM retourne parfois des tokens qui ne correspondent pas exactement 1:1 au texte (decoding quirks)
        # Mais on va supposer que l'ordre est bon.
        
        current_pos = 0
        for token_data in logprobs:
            token_text = token_data.get('token', '')
            logprob = token_data.get('logprob')
            perplexity = math.exp(-logprob) if logprob is not None else None
            
            length = len(token_text)
            token_char_map.append({
                'token': token_text,
                'perplexity': perplexity,
                'start': current_pos,
                'end': current_pos + length
            })
            current_pos += length

    try:
        lines = [l for l in tsv_content.strip().split('\n') if l.strip()]
        
        # Cursor pour chercher dans le full_response
        full_text_search_cursor = 0
        
        for i, line in enumerate(lines):
            try:
                bbox = parse_tsv_line(line)
                # Convertir en 4 points
                points = bbox_to_4points(bbox)
                
                shape = {
                    "label": bbox.text,
                    "points": points,
                    "shape_type": "polygon",
                    "flags": {},
                    "group_id": None,
                    "description": "" # Standard labelme field
                }
                
                # Ajouter les infos de perplexité si disponibles
                if logprobs and full_response_text:
                    # Trouver la ligne dans le texte complet
                    # On cherche à partir de la position précédente
                    line_start = full_response_text.find(line, full_text_search_cursor)
                    
                    if line_start != -1:
                        line_end = line_start + len(line)
                        full_text_search_cursor = line_end # update cursor
                        
                        # Trouver les tokens qui intersectent cette plage
                        line_tokens = []
                        for t_map in token_char_map:
                            # Intersection
                            if max(line_start, t_map['start']) < min(line_end, t_map['end']):
                                line_tokens.append({
                                    "token": t_map['token'],
                                    "perplexity": t_map['perplexity']
                                })
                        
                        # Stocker dans un champ custom 'extra_data' (pas standard labelme mais ignoré par UI)
                        # ou dans 'description' au format JSON
                        shape["token_info"] = line_tokens
                        
                        # Calculer stats pour flags ou description
                        if line_tokens:
                            valid_perps = [t['perplexity'] for t in line_tokens if t['perplexity'] is not None]
                            if valid_perps:
                                avg_perp = sum(valid_perps) / len(valid_perps)
                                max_perp = max(valid_perps)
                                shape["description"] += f"Avg Perp: {avg_perp:.2f} | Max Perp: {max_perp:.2f}"
                
                shapes.append(shape)
            except Exception as line_error:
                has_parsing_error = True
                parsing_errors.append({
                    "line_number": i + 1,
                    "line_content": line,
                    "error": str(line_error)
                })
    except Exception as e:
        has_parsing_error = True
        parsing_errors.append({
            "line_number": 0,
            "line_content": "N/A",
            "error": f"Erreur globale: {str(e)}"
        })
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
    
    # Ajouter la sortie brute en cas d'erreur de parsing pour debug
    if has_parsing_error:
        labelme_data["raw_tsv"] = tsv_content
        labelme_data["parsing_errors"] = parsing_errors
        labelme_data["flags"]["parsing_error"] = True
    
    return labelme_data

