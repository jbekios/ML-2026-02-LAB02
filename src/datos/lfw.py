"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

from src.datos.base import CargadorRostros
from src.excepciones import EtapaPendienteAlumno
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    TODO(alumno): implementar ``cargar`` y devolver un ``ConjuntoRostros``.
    Cuando este método deje de lanzar ``EtapaPendienteAlumno``,
    ``python main.py lfw`` reutiliza el mismo bucle de experimentos que Olivetti.

    Pasos esperados:

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5, data_home=...)``.
    2. Conservar como máximo 50 identidades, las de mayor frecuencia.
    3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)`` y guardar los nombres.
    4. Leer alto y ancho desde ``images.shape``.
    5. Llamar a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        raise EtapaPendienteAlumno(
            "datos.lfw.CargadorLFW",
            "Filtre identidades con al menos 20 fotos, conserve hasta 50, "
            "reindexe y y lea alto y ancho desde images.shape. "
            "Después, python main.py lfw evaluará PCA, LDA y Fisherfaces "
            "con el mismo protocolo que Olivetti.",
        )
