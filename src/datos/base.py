"""Contrato común de los cargadores de rostros."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.modelos import ConjuntoRostros


class CargadorRostros(ABC):
    """Carga un conjunto público y lo deja en el contrato images / X / y."""

    @abstractmethod
    def cargar(self) -> ConjuntoRostros:
        """Devuelve fotografías, píxeles aplanados y la identidad de cada fila."""


def validar_conjunto(conjunto: ConjuntoRostros) -> None:
    """Comprueba formas alineadas y valores finitos antes de separar."""
    if conjunto.images.ndim != 3:
        raise ValueError("images debe tener forma (N, H, W).")
    n, alto, ancho = conjunto.images.shape
    if (alto, ancho) != (conjunto.alto, conjunto.ancho):
        raise ValueError("alto y ancho no coinciden con images.shape.")
    if conjunto.X.shape != (n, alto * ancho):
        raise ValueError(
            f"X debe tener forma {(n, alto * ancho)}; se recibió {conjunto.X.shape}."
        )
    if conjunto.y.shape != (n,):
        raise ValueError(f"y debe tener forma {(n,)}; se recibió {conjunto.y.shape}.")
    if not np.isfinite(conjunto.X).all():
        raise ValueError("X contiene valores no finitos.")
    if len(conjunto.nombres_identidades) < conjunto.n_clases:
        raise ValueError("Faltan nombres para las identidades del conjunto.")
