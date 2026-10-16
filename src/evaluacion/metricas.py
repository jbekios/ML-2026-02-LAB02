"""Accuracy, macro-F1, matriz de confusión y tiempo de ajuste."""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.modelos import ResultadoClasificacion


class Evaluador:
    """Ajusta el clasificador en Z_train y mide solo sobre Z_test."""

    def evaluar(
        self,
        modelo: ClassifierMixin,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ResultadoClasificacion:
        etiquetas = np.unique(y_train)
        inicio = time.perf_counter()
        modelo.fit(Z_train, y_train)
        predicciones = np.asarray(modelo.predict(Z_test))
        tiempo_s = time.perf_counter() - inicio
        if predicciones.shape != (y_test.shape[0],):
            raise ValueError(
                f"Se esperaban {y_test.shape[0]} predicciones y llegaron {predicciones.shape}."
            )
        matriz = confusion_matrix(y_test, predicciones, labels=etiquetas)
        return ResultadoClasificacion(
            accuracy=float(accuracy_score(y_test, predicciones)),
            macro_f1=float(f1_score(y_test, predicciones, average="macro", zero_division=0)),
            tiempo_s=float(tiempo_s),
            predicciones=predicciones,
            matriz=matriz,
        )
