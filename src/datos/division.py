"""Separación estratificada. El ajuste ocurre después, solo con TRAIN."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.modelos import ConjuntoRostros, Particion


class DivisorEstratificado:
    """Mantiene la proporción de identidades y deja train/test disjuntos."""

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def dividir(self, conjunto: ConjuntoRostros) -> Particion:
        indices = np.arange(conjunto.X.shape[0])
        idx_train, idx_test = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=conjunto.y,
            random_state=self.random_state,
        )
        idx_train = np.sort(idx_train)
        idx_test = np.sort(idx_test)
        self._verificar(idx_train, idx_test, conjunto.y)
        return Particion(
            X_train=conjunto.X[idx_train],
            X_test=conjunto.X[idx_test],
            y_train=conjunto.y[idx_train],
            y_test=conjunto.y[idx_test],
            images_train=conjunto.images[idx_train],
            images_test=conjunto.images[idx_test],
            indices_train=idx_train,
            indices_test=idx_test,
        )

    @staticmethod
    def _verificar(idx_train: np.ndarray, idx_test: np.ndarray, y: np.ndarray) -> None:
        if np.intersect1d(idx_train, idx_test).size:
            raise ValueError("Los índices de train y test no son disjuntos.")
        faltan = set(np.unique(y[idx_test])) - set(np.unique(y[idx_train]))
        if faltan:
            raise ValueError(
                "Hay identidades de test que no aparecen en train: "
                + ", ".join(str(int(c)) for c in sorted(faltan))
            )
