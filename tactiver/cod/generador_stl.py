"""Generador de placas táctiles STL de la Fase 1.

El modelo usa milímetros y genera una placa con ejes, marcas y una línea
en relieve. Las etiquetas numéricas se representan en Braille Nemeth.
"""
import math
from math import atan2, degrees, hypot

import cadquery as cq

BASE_THICKNESS = 2.0
ESPACIADO_BRAILLE = 2.4
DIAM_PUNTO_BRAILLE = 1.4
CELDA_PITCH = 6.2
CLEARANCE_BRAILLE = 9.5
RELIEVE_EJE = 1.0
RELIEVE_TICK = 1.0
RELIEVE_LINEA = 1.6
DIAM_LINEA = 2.0
MARGEN_BORDE = 25.0

LETRAS = {
    "a": [1], "b": [1, 2], "c": [1, 4], "d": [1, 4, 5], "e": [1, 5],
    "f": [1, 2, 4], "g": [1, 2, 4, 5], "h": [1, 2, 5], "i": [2, 4],
    "j": [2, 4, 5], "k": [1, 3], "l": [1, 2, 3], "m": [1, 3, 4],
    "n": [1, 3, 4, 5], "o": [1, 3, 5], "p": [1, 2, 3, 4],
    "q": [1, 2, 3, 4, 5], "r": [1, 2, 3, 5], "s": [2, 3, 4],
    "t": [2, 3, 4, 5], "u": [1, 3, 6], "v": [1, 2, 3, 6],
    "w": [2, 4, 5, 6], "x": [1, 3, 4, 6], "y": [1, 3, 4, 5, 6],
    "z": [1, 3, 5, 6],
}
BRAILLE = dict(LETRAS)
BRAILLE["numeral"] = [3, 4, 5, 6]
BRAILLE["menos"] = [3, 6]
_DESPLAZAMIENTO_NEMETH = {1: 2, 2: 3, 4: 5, 5: 6}
for numero, letra in enumerate("abcdefghij", start=1):
    BRAILLE[str(numero % 10)] = sorted(_DESPLAZAMIENTO_NEMETH[p] for p in LETRAS[letra])


def _punto_braille(modelo, cx, cy):
    punto = cq.Workplane("XY").workplane(offset=BASE_THICKNESS).center(cx, cy).sphere(DIAM_PUNTO_BRAILLE / 2)
    return modelo.union(punto)


def _caracter_braille(modelo, caracter, cx, cy):
    posiciones = {1: (-1.2, 2.4), 2: (-1.2, 0), 3: (-1.2, -2.4),
                  4: (1.2, 2.4), 5: (1.2, 0), 6: (1.2, -2.4)}
    for punto in BRAILLE.get(str(caracter).lower(), []):
        dx, dy = posiciones[punto]
        modelo = _punto_braille(modelo, cx + dx, cy + dy)
    return modelo


def _numero_braille(modelo, valor, cx, cy):
    valor = int(valor)
    if valor < 0:
        modelo = _caracter_braille(modelo, "menos", cx, cy)
        cx += CELDA_PITCH
    modelo = _caracter_braille(modelo, "numeral", cx, cy)
    for digito in str(abs(valor)):
        cx += CELDA_PITCH
        modelo = _caracter_braille(modelo, digito, cx, cy)
    return modelo


def _segmento(modelo, inicio, fin, diametro, altura):
    x0, y0 = inicio
    x1, y1 = fin
    largo = hypot(x1 - x0, y1 - y0)
    if largo < 1e-6:
        return modelo
    angulo = degrees(atan2(y1 - y0, x1 - x0))
    cuerpo = (cq.Workplane("XY").workplane(offset=BASE_THICKNESS)
              .center((x0 + x1) / 2, (y0 + y1) / 2)
              .transformed(rotate=(0, 0, angulo))
              .box(largo, diametro, altura, centered=(True, True, False)))
    for x, y in (inicio, fin):
        cuerpo = cuerpo.union(cq.Workplane("XY").workplane(offset=BASE_THICKNESS).center(x, y).circle(diametro / 2).extrude(altura))
    return modelo.union(cuerpo)


def _ticks(minimo, maximo, intervalo):
    inicio = math.ceil(minimo / intervalo) * intervalo
    return list(range(inicio, maximo + 1, intervalo))


def generar_modelo_bana(dim_x=210.0, dim_y=148.0, p1=(0, 0), p2=(150, 90),
                        intervalo_ticks=25, archivo_salida="grafica_bana.stl"):
    """Exporta una gráfica lineal táctil y devuelve el sólido CadQuery."""
    if dim_x <= 2 * MARGEN_BORDE or dim_y <= 2 * MARGEN_BORDE:
        raise ValueError("Las dimensiones de la placa son demasiado pequeñas para los márgenes táctiles.")
    if intervalo_ticks <= 0:
        raise ValueError("El intervalo entre marcas debe ser mayor que cero.")

    p1, p2 = tuple(map(float, p1)), tuple(map(float, p2))
    x_min, x_max = min(0, p1[0], p2[0]), max(0, p1[0], p2[0])
    y_min, y_max = min(0, p1[1], p2[1]), max(0, p1[1], p2[1])
    if x_max - x_min > dim_x - 2 * MARGEN_BORDE or y_max - y_min > dim_y - 2 * MARGEN_BORDE:
        raise ValueError("Los datos no caben en la placa con el margen de 25 mm por lado.")

    origen_x, origen_y = MARGEN_BORDE - x_min, MARGEN_BORDE - y_min
    fisico = lambda p: (p[0] + origen_x, p[1] + origen_y)
    modelo = cq.Workplane("XY").center(dim_x / 2, dim_y / 2).box(dim_x, dim_y, BASE_THICKNESS, centered=(True, True, False))

    x0, x1 = fisico((x_min, 0))[0], fisico((x_max, 0))[0]
    y0, y1 = fisico((0, y_min))[1], fisico((0, y_max))[1]
    modelo = modelo.union(cq.Workplane("XY").center((x0 + x1) / 2, origen_y).box(x1 - x0, 2, BASE_THICKNESS + RELIEVE_EJE, centered=(True, True, False)))
    modelo = modelo.union(cq.Workplane("XY").center(origen_x, (y0 + y1) / 2).box(2, y1 - y0, BASE_THICKNESS + RELIEVE_EJE, centered=(True, True, False)))

    for valor in _ticks(int(x_min), int(x_max), int(intervalo_ticks)):
        x = origen_x + valor
        if valor:
            modelo = modelo.union(cq.Workplane("XY").center(x, origen_y).box(2, 6, BASE_THICKNESS + RELIEVE_TICK, centered=(True, True, False)))
        modelo = _numero_braille(modelo, valor, x, origen_y - CLEARANCE_BRAILLE)
    for valor in _ticks(int(y_min), int(y_max), int(intervalo_ticks)):
        if not valor:
            continue
        y = origen_y + valor
        modelo = modelo.union(cq.Workplane("XY").center(origen_x, y).box(6, 2, BASE_THICKNESS + RELIEVE_TICK, centered=(True, True, False)))
        modelo = _numero_braille(modelo, valor, origen_x - CLEARANCE_BRAILLE - CELDA_PITCH * (len(str(abs(valor))) + 2), y)

    modelo = _segmento(modelo, fisico(p1), fisico(p2), DIAM_LINEA, RELIEVE_LINEA)
    cq.exporters.export(modelo, archivo_salida)
    return modelo


def _ancho_numero(valor):
    return CELDA_PITCH * (len(str(abs(int(round(valor))))) + 1)


def generar_modelo_desde_recta(puntos, dim_x=210.0, dim_y=148.0,
                               archivo_salida="grafica_tactil.stl"):
    """Genera una placa STL de una recta a partir de los puntos del segmentador.

    Cada punto es un dict con ``px``, ``py``, ``valor_x`` y ``valor_y``.
    Solo se usan el primer y último punto: este generador está limitado a
    gráficas lineales y no dibuja curvas ni polilíneas.
    """
    if dim_x < 130 or dim_y < 90:
        raise ValueError("La placa debe medir al menos 130 x 90 mm.")
    if not isinstance(puntos, (list, tuple)) or len(puntos) < 2:
        raise ValueError("Se necesitan al menos dos puntos para generar un STL.")

    calibrados = all(p.get("valor_x") is not None and p.get("valor_y") is not None for p in puntos)
    if calibrados:
        datos = [(float(puntos[0]["valor_x"]), float(puntos[0]["valor_y"])),
                 (float(puntos[-1]["valor_x"]), float(puntos[-1]["valor_y"]))]
    else:
        # En imágenes el eje Y crece hacia abajo; se invierte para mantener
        # la inclinación visual de la recta en la placa táctil.
        datos = [(float(puntos[0]["px"]), -float(puntos[0]["py"])),
                 (float(puntos[-1]["px"]), -float(puntos[-1]["py"]))]
    xs, ys = zip(*datos)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min or y_max == y_min:
        raise ValueError("La curva debe tener variación tanto en X como en Y.")

    # Reserva real para etiquetas Braille a la izquierda y debajo de los ejes.
    izquierda, derecha, abajo, arriba = 55.0, 15.0, 25.0, 15.0
    ancho_plot, alto_plot = dim_x - izquierda - derecha, dim_y - abajo - arriba

    def escalar(x, y):
        return (
            izquierda + (x - x_min) / (x_max - x_min) * ancho_plot,
            abajo + (y - y_min) / (y_max - y_min) * alto_plot,
        )

    modelo = cq.Workplane("XY").center(dim_x / 2, dim_y / 2).box(
        dim_x, dim_y, BASE_THICKNESS, centered=(True, True, False)
    )
    modelo = _segmento(modelo, (izquierda, abajo), (dim_x - derecha, abajo), 2.0, RELIEVE_EJE)
    modelo = _segmento(modelo, (izquierda, abajo), (izquierda, dim_y - arriba), 2.0, RELIEVE_EJE)

    # Cinco marcas por eje dan una referencia táctil clara sin sobrecargar la placa.
    for i in range(5):
        fraccion = i / 4
        x_fis = izquierda + fraccion * ancho_plot
        y_fis = abajo + fraccion * alto_plot
        modelo = _segmento(modelo, (x_fis, abajo - 3), (x_fis, abajo + 3), 1.5, RELIEVE_TICK)
        modelo = _segmento(modelo, (izquierda - 3, y_fis), (izquierda + 3, y_fis), 1.5, RELIEVE_TICK)
        if calibrados:
            x_valor = x_min + fraccion * (x_max - x_min)
            y_valor = y_min + fraccion * (y_max - y_min)
            ancho_x = _ancho_numero(x_valor)
            inicio_x = min(max(3.0, x_fis - ancho_x / 2), dim_x - ancho_x - 3.0)
            modelo = _numero_braille(modelo, round(x_valor), inicio_x, abajo - CLEARANCE_BRAILLE)
            ancho_y = _ancho_numero(y_valor)
            inicio_y = max(3.0, izquierda - CLEARANCE_BRAILLE - ancho_y)
            modelo = _numero_braille(modelo, round(y_valor), inicio_y, y_fis)

    inicio, fin = (escalar(*datos[0]), escalar(*datos[1]))
    modelo = _segmento(modelo, inicio, fin, DIAM_LINEA, RELIEVE_LINEA)

    cq.exporters.export(modelo, archivo_salida)
    return modelo
