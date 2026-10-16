"""Contrato de un embedding lineal ajustado solo con TRAIN."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ExtractorEmbedding(ABC):
    """Proyección lineal. ``fit`` ve TRAIN; ``transform`` no vuelve a ajustar."""

    nombre: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Cantidad de coordenadas del embedding."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorEmbedding:
        """Estima la proyección con TRAIN."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta filas ya centradas según lo aprendido en ``fit``."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def comprobar(self, Z: np.ndarray, n_filas: int) -> np.ndarray:
        """Exige filas conservadas, d válido y valores finitos."""
        if Z.ndim != 2 or Z.shape[0] != n_filas:
            raise ValueError(
                f"{self.nombre}: se esperaban {n_filas} filas y se obtuvo {Z.shape}."
            )
        if Z.shape[1] != self.dimension:
            raise ValueError(
                f"{self.nombre}: d={Z.shape[1]} no coincide con {self.dimension}."
            )
        if not np.isfinite(Z).all():
            raise ValueError(f"{self.nombre}: el embedding contiene valores no finitos.")
        return Z
