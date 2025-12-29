"""
Structures de données pour l'évaluation HTR.

Ce module contient les dataclasses et types utilisés dans le package d'évaluation.
"""

from dataclasses import dataclass


@dataclass
class BBox:
    """
    Représente une boîte englobante orientée.
    
    Attributes:
        xmin: Coordonnée X minimale
        ymin: Coordonnée Y minimale
        xmax: Coordonnée X maximale
        ymax: Coordonnée Y maximale
        angle: Angle de rotation en degrés
        text: Texte transcrit dans la boîte
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    angle: float
    text: str
