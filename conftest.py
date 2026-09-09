"""
Configuracion compartida de pytest.

Agrega mlops_pipeline/src al path de Python para poder hacer
`from ft_engineering import ...` en los tests, sin necesidad de
convertir mlops_pipeline en un paquete (evita modificar la estructura
de carpetas fija del proyecto).
"""
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "mlops_pipeline" / "src"
sys.path.insert(0, str(SRC_PATH))
