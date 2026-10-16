"""Carga de Olivetti Faces."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from src.config import DIR_CACHE, RANDOM_STATE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorOlivetti(CargadorRostros):
    """400 fotografías, 40 identidades, 64×64 píxeles en escala de grises."""

    nombre = "olivetti"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_olivetti_faces(
            shuffle=True,
            random_state=RANDOM_STATE,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        images = np.asarray(rostros.images)
        X = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        alto, ancho = int(images.shape[1]), int(images.shape[2])
        nombres = [f"Identidad {i}" for i in range(int(y.max()) + 1)]
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        return conjunto
