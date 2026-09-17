# -*- coding: utf-8 -*-
"""
segmentador.py
==============
Pipeline 100% automático (sin clics manuales, sin SAM2) para:
  1) Detectar los ejes X e Y de una gráfica (OpenCV / Hough).
  2) Detectar la curva de datos por color dominante.
  3) Detectar y clasificar el texto (título, números eje X, números eje Y)
     con Tesseract OCR, según su posición respecto a los ejes.
  4) Calibrar píxel -> valor real usando las etiquetas numéricas leídas.
  5) Exportar un único CSV con todo: ejes, textos y puntos de la curva.

Pensado para ser llamado desde una interfaz web (ver app.py), no requiere
ninguna interacción del usuario más allá de subir la imagen.
"""

import os
import re
import csv
import uuid

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# --- Configuración de Tesseract OCR en Windows ---
# Si el motor de Tesseract no está en el PATH del sistema, hay que indicarle
# a pytesseract dónde encontrar el ejecutable. Ajusta esta ruta si instalaste
# Tesseract en una carpeta distinta a la que aparece por defecto.
_RUTA_TESSERACT_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.isfile(_RUTA_TESSERACT_WINDOWS):
    pytesseract.pytesseract.tesseract_cmd = _RUTA_TESSERACT_WINDOWS


# ============================================================
# 1) DETECCIÓN AUTOMÁTICA DE EJES (Hough)
# ============================================================

def detectar_ejes(gris, largo_minimo=150):
    """
    Devuelve (eje_x, eje_y) como tuplas (x1,y1,x2,y2), o None si no se
    encuentran.

    eje_x = línea horizontal más abajo (se asume que es la base del
    gráfico).

    eje_y = de todas las líneas verticales candidatas, la que está más
    cerca en X del extremo izquierdo de eje_x. No se toma directamente
    "la más a la izquierda de toda la imagen" porque eso puede agarrar
    por error algún trazo vertical de texto (p. ej. el título del eje Y,
    rotado) que quede más a la izquierda que el eje real. El eje Y
    real siempre se junta con el eje X en la esquina inferior izquierda
    del gráfico, así que anclarlo al extremo de eje_x es más confiable.
    Si no se detectó eje_x, se usa el criterio anterior (más a la
    izquierda) como respaldo.
    """
    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLinesP(
        bordes, rho=1, theta=np.pi / 180,
        threshold=60, minLineLength=largo_minimo, maxLineGap=15,
    )

    if lineas is None:
        return None, None

    lineas = lineas.reshape(-1, 4)

    horizontales, verticales = [], []
    for x1, y1, x2, y2 in lineas:
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        largo = np.hypot(dx, dy)
        if largo < largo_minimo:
            continue

        if dy < 15:  # horizontal
            horizontales.append((int(x1), int(y1), int(x2), int(y2)))
        if dx < 15:  # vertical
            verticales.append((int(x1), int(y1), int(x2), int(y2)))

    mejor_h = None
    if horizontales:
        mejor_h = max(horizontales, key=lambda l: (l[1] + l[3]) / 2)

    mejor_v = None
    if verticales:
        if mejor_h is not None:
            x_esquina = min(mejor_h[0], mejor_h[2])  # extremo izquierdo de eje_x
            mejor_v = min(verticales, key=lambda l: abs(((l[0] + l[2]) / 2) - x_esquina))
        else:
            mejor_v = min(verticales, key=lambda l: (l[0] + l[2]) / 2)

    return mejor_h, mejor_v


# ============================================================
# 2) DETECCIÓN AUTOMÁTICA DE LA CURVA (color dominante)
# ============================================================

def detectar_curva(imagen_bgr, sat_minima=60, tolerancia_hue=12, area_minima=40):
    """
    Encuentra la curva de datos asumiendo que es la línea de color más
    "saturado" (distinto de fondo blanco/gris y de texto/ejes negros).
    Devuelve una máscara booleana del mismo tamaño que la imagen, o None
    si no se detecta ningún color dominante.
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_color = s > sat_minima
    if not mask_color.any():
        return None

    hues = h[mask_color]
    hist = np.bincount(hues, minlength=180)
    hue_dom = int(np.argmax(hist))

    lower = max(hue_dom - tolerancia_hue, 0)
    upper = min(hue_dom + tolerancia_hue, 179)

    mask_hue = ((h >= lower) & (h <= upper) & mask_color).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hue, connectivity=8)
    if n_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_mayor = 1 + int(np.argmax(areas))
    if areas[idx_mayor - 1] < area_minima:
        return None

    return labels == idx_mayor


# ============================================================
# 3) DETECCIÓN Y CLASIFICACIÓN AUTOMÁTICA DE TEXTO
# ============================================================

def _mapa_distancia_curva(mascara_curva, forma):
    """
    Devuelve un mapa (mismo tamaño que la imagen) donde cada píxel
    contiene la distancia en píxeles al punto más cercano de la curva
    de datos. Se usa para saber si un número leído por OCR está "pegado"
    a la curva (etiqueta de dato) o lejos de ella (probable etiqueta de
    eje). None si no hay curva detectada.
    """
    if mascara_curva is None:
        return None
    no_curva = (~mascara_curva).astype(np.uint8) * 255
    return cv2.distanceTransform(no_curva, cv2.DIST_L2, 5)


def detectar_textos(gris, eje_x_fila, eje_y_col, mascara_curva=None, margen=8, radio_dato=18, escala_ocr=2.0):
    """
    Corre OCR sobre TODA la imagen una sola vez y clasifica cada token
    según su posición respecto a los ejes detectados:
      - número MUY CERCA de la curva de datos (a menos de radio_dato
        píxeles) -> "etiqueta de dato": es un valor escrito junto a un
        punto de la curva (común en gráficos con las cifras encima de
        cada punto), NO una etiqueta de eje. Se guarda aparte porque
        además sirve como fuente alterna de calibración cuando no se
        pueden leer los números del eje (ver procesar_imagen).
      - número cuyo centro está por debajo/cerca de la fila del eje X
        -> etiqueta numérica del eje X
      - número cuyo centro está a la izquierda/cerca de la columna del
        eje Y -> etiqueta numérica del eje Y
      - texto no numérico por ENCIMA del eje X y a la derecha del eje Y
        -> candidato a título principal de la gráfica (se concatenan
        todas las palabras encontradas ahí).
      - texto no numérico por DEBAJO de la fila de etiquetas numéricas
        del eje X -> candidato a título del eje X (p. ej. "Número de
        salidas"), también horizontal, así que se lee sin rotar.
        El texto girado 90° a la izquierda del eje Y (título del eje Y)
        NO se procesa aquí porque el OCR horizontal no lo lee bien; lo
        maneja aparte detectar_texto_vertical().

    escala_ocr: factor de ampliación aplicado a la imagen SOLO para el
    OCR (no afecta al resto del pipeline). Tesseract suele fallar en
    números pequeños o en negrita de gráficos chicos; agrandar la
    imagen antes de leerla mejora bastante la tasa de acierto. Las
    coordenadas leídas se reescalan de vuelta al tamaño original antes
    de devolverlas.
    """
    mapa_distancia_curva = _mapa_distancia_curva(mascara_curva, gris.shape)

    if escala_ocr and escala_ocr != 1:
        gris_ocr = cv2.resize(gris, None, fx=escala_ocr, fy=escala_ocr, interpolation=cv2.INTER_CUBIC)
    else:
        gris_ocr = gris
    datos = pytesseract.image_to_data(gris_ocr, output_type=Output.DICT, config="--psm 11")

    etiquetas_x, etiquetas_y, etiquetas_dato = [], [], []
    partes_titulo = []     # (cy, texto, x, y, w, h) del título principal (arriba)
    partes_titulo_x = []   # (cy, texto, x, y, w, h) del título del eje X (abajo)

    n = len(datos["text"])
    for i in range(n):
        texto = datos["text"][i].strip()
        if not texto:
            continue

        try:
            conf = float(datos["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf != -1 and conf < 30:
            continue

        x = int(round(datos["left"][i] / escala_ocr))
        y = int(round(datos["top"][i] / escala_ocr))
        w = int(round(datos["width"][i] / escala_ocr))
        h = int(round(datos["height"][i] / escala_ocr))
        cx, cy = x + w / 2, y + h / 2

        es_numero = bool(re.fullmatch(r"-?\d+([.,]\d+)?", texto))

        # Si el número está pegado a la curva, es una etiqueta de dato
        # (el valor de ese punto escrito junto a él), no una etiqueta de
        # eje -- se guarda aparte para no arruinar la calibración de los
        # ejes con un valor que no corresponde a una marca de escala.
        cerca_de_la_curva = False
        if es_numero and mapa_distancia_curva is not None:
            fila_px = min(max(int(round(cy)), 0), mapa_distancia_curva.shape[0] - 1)
            col_px = min(max(int(round(cx)), 0), mapa_distancia_curva.shape[1] - 1)
            cerca_de_la_curva = mapa_distancia_curva[fila_px, col_px] < radio_dato

        if es_numero and cerca_de_la_curva:
            etiquetas_dato.append({
                "valor": float(texto.replace(",", ".")),
                "centro_x": cx, "centro_y": cy,
                "px": x, "py": y, "pw": w, "ph": h,
            })
        elif es_numero and eje_x_fila is not None and cy > eje_x_fila - margen:
            etiquetas_x.append({
                "valor": float(texto.replace(",", ".")),
                "centro_x": cx, "centro_y": cy,
                "px": x, "py": y, "pw": w, "ph": h,
            })
        elif es_numero and eje_y_col is not None and cx < eje_y_col + margen:
            etiquetas_y.append({
                "valor": float(texto.replace(",", ".")),
                "centro_x": cx, "centro_y": cy,
                "px": x, "py": y, "pw": w, "ph": h,
            })
        elif not es_numero:
            en_zona_titulo = (eje_x_fila is None) or (cy < eje_x_fila)
            a_la_derecha_del_eje_y = (eje_y_col is None) or (cx > eje_y_col)
            if en_zona_titulo and a_la_derecha_del_eje_y:
                partes_titulo.append((cy, texto, x, y, w, h))
            elif eje_x_fila is not None and cy > eje_x_fila:
                partes_titulo_x.append((cy, texto, x, y, w, h))

    partes_titulo.sort(key=lambda t: t[0])
    titulo = " ".join(t[1] for t in partes_titulo).strip()

    # Bounding box del título completo = unión de las cajas de cada palabra
    titulo_bbox = None
    if partes_titulo:
        xs1 = [t[2] for t in partes_titulo]
        ys1 = [t[3] for t in partes_titulo]
        xs2 = [t[2] + t[4] for t in partes_titulo]
        ys2 = [t[3] + t[5] for t in partes_titulo]
        tx, ty = min(xs1), min(ys1)
        tw, th = max(xs2) - tx, max(ys2) - ty
        titulo_bbox = {"px": tx, "py": ty, "pw": tw, "ph": th}

    partes_titulo_x.sort(key=lambda t: t[0])
    titulo_x = " ".join(t[1] for t in partes_titulo_x).strip()

    titulo_x_bbox = None
    if partes_titulo_x:
        xs1 = [t[2] for t in partes_titulo_x]
        ys1 = [t[3] for t in partes_titulo_x]
        xs2 = [t[2] + t[4] for t in partes_titulo_x]
        ys2 = [t[3] + t[5] for t in partes_titulo_x]
        tx, ty = min(xs1), min(ys1)
        tw, th = max(xs2) - tx, max(ys2) - ty
        titulo_x_bbox = {"px": tx, "py": ty, "pw": tw, "ph": th}

    return etiquetas_x, etiquetas_y, etiquetas_dato, titulo, titulo_bbox, titulo_x, titulo_x_bbox


# ============================================================
# 3b) TEXTO VERTICAL (título del eje Y, rotado 90°)
# ============================================================

def detectar_texto_vertical(gris, x_limite, margen=6, ancho_minimo=15):
    """
    Busca texto girado 90° (típicamente el título del eje Y, que se lee
    de abajo hacia arriba o de arriba hacia abajo) en la franja de la
    imagen a la izquierda de x_limite (borde izquierdo de las etiquetas
    numéricas del eje Y, o del propio eje Y si no hay etiquetas).

    Prueba las dos rotaciones posibles (90° horario y 90° antihorario)
    porque no sabemos de antemano en qué sentido está escrito el texto,
    y se queda con la que produce texto con mayor confianza promedio.

    Devuelve (texto, bbox) donde bbox está en coordenadas de la imagen
    ORIGINAL (sin rotar): {"px","py","pw","ph"}. Si no encuentra nada
    confiable, devuelve ("", None).
    """
    x_limite = int(x_limite) - margen
    if x_limite < ancho_minimo:
        return "", None

    franja = gris[:, :x_limite]
    alto_franja, ancho_franja = franja.shape[:2]
    if alto_franja == 0 or ancho_franja == 0:
        return "", None

    mejor_texto, mejor_bbox, mejor_conf = "", None, -1.0

    for rotacion in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotada = cv2.rotate(franja, rotacion)
        datos = pytesseract.image_to_data(rotada, output_type=Output.DICT, config="--psm 6")

        palabras, confs, cajas = [], [], []

        for i in range(len(datos["text"])):
            texto = datos["text"][i].strip()
            if not texto:
                continue
            try:
                conf = float(datos["conf"][i])
            except (ValueError, TypeError):
                conf = -1
            if conf < 30:
                continue

            x, y = datos["left"][i], datos["top"][i]
            w, h = datos["width"][i], datos["height"][i]

            # Recuadro de la imagen rotada -> recuadro en la franja original.
            if rotacion == cv2.ROTATE_90_CLOCKWISE:
                px, py = y, alto_franja - (x + w)
            else:  # ROTATE_90_COUNTERCLOCKWISE
                px, py = ancho_franja - (y + h), x
            pw, ph = h, w

            palabras.append(texto)
            confs.append(conf)
            cajas.append((px, py, pw, ph))

        if not palabras:
            continue

        conf_prom = sum(confs) / len(confs)
        if conf_prom > mejor_conf:
            xs1 = [c[0] for c in cajas]
            ys1 = [c[1] for c in cajas]
            xs2 = [c[0] + c[2] for c in cajas]
            ys2 = [c[1] + c[3] for c in cajas]
            mejor_bbox = {
                "px": min(xs1), "py": min(ys1),
                "pw": max(xs2) - min(xs1), "ph": max(ys2) - min(ys1),
            }
            mejor_texto, mejor_conf = " ".join(palabras), conf_prom

    return mejor_texto, mejor_bbox


# ============================================================
# 4) CALIBRACIÓN LINEAL PÍXEL -> VALOR REAL
# ============================================================

def ajustar_lineal(pixeles, valores):
    if len(pixeles) < 2:
        return None, None
    m, b = np.polyfit(pixeles, valores, 1)
    return m, b


# ============================================================
# 5) PIPELINE COMPLETO
# ============================================================

def procesar_imagen(ruta_imagen, dir_resultados):
    """
    Ejecuta todo el pipeline sobre una imagen y devuelve un dict con:
      - ruta_zip (contiene 4 CSVs separados: resumen, ejes, textos, curva)
      - ruta_overlay
      - resumen (dict con lo detectado, para mostrar en la interfaz)
    No requiere ninguna interacción del usuario.
    """
    os.makedirs(dir_resultados, exist_ok=True)
    uid = uuid.uuid4().hex[:8]

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta_imagen}")

    alto_imagen, ancho_imagen = imagen.shape[:2]

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # --- Ejes (guardamos también los 4 extremos completos, no solo la posición) ---
    eje_x, eje_y = detectar_ejes(gris)
    eje_x_fila = ((eje_x[1] + eje_x[3]) / 2) if eje_x else None
    eje_y_col = ((eje_y[0] + eje_y[2]) / 2) if eje_y else None

    # --- Curva ---
    mascara_curva = detectar_curva(imagen)

    # --- Texto (título, números eje X, números eje Y, etiquetas de dato) ---
    etiquetas_x, etiquetas_y, etiquetas_dato, titulo, titulo_bbox, titulo_x, titulo_x_bbox = \
        detectar_textos(gris, eje_x_fila, eje_y_col, mascara_curva=mascara_curva)

    # --- Título del eje Y (texto vertical, rotado 90°) ---
    # Se busca a la izquierda del borde izquierdo de las etiquetas numéricas
    # del eje Y (o del propio eje Y, si no hubo etiquetas numéricas legibles).
    if etiquetas_y:
        x_limite_vertical = min(e["px"] for e in etiquetas_y)
    else:
        x_limite_vertical = eje_y_col

    titulo_eje_y, titulo_eje_y_bbox = ("", None)
    if x_limite_vertical is not None:
        titulo_eje_y, titulo_eje_y_bbox = detectar_texto_vertical(gris, x_limite_vertical)

    # --- Calibración ---
    m_x, b_x = ajustar_lineal([e["centro_x"] for e in etiquetas_x], [e["valor"] for e in etiquetas_x])
    m_y, b_y = ajustar_lineal([e["centro_y"] for e in etiquetas_y], [e["valor"] for e in etiquetas_y])

    # --- Fallback: si no se pudo calibrar el eje Y con etiquetas de eje
    #     (porque no hay o no se leyeron bien), pero la gráfica tiene
    #     números escritos junto a los puntos de la curva (etiquetas de
    #     dato), se usan esos como calibración: cada etiqueta se empareja
    #     con la fila (posición vertical) del punto de la curva más
    #     cercano en esa misma columna, y se regresa píxel -> valor con
    #     esos pares. Es lo que salva casos como gráficos de líneas con
    #     las cifras impresas sobre cada punto pero sin ejes numéricos
    #     legibles.
    calibrado_y_por_datos = False
    if m_y is None and etiquetas_dato and mascara_curva is not None:
        filas_c, cols_c = np.where(mascara_curva)
        if cols_c.size > 0:
            cols_unicas = np.unique(cols_c)
            pares_fila, pares_valor = [], []
            for e in etiquetas_dato:
                idx = np.searchsorted(cols_unicas, e["centro_x"])
                idx = min(max(idx, 0), len(cols_unicas) - 1)
                col_cercana = cols_unicas[idx]
                fila_prom = filas_c[cols_c == col_cercana].mean()
                pares_fila.append(fila_prom)
                pares_valor.append(e["valor"])
            m_y, b_y = ajustar_lineal(pares_fila, pares_valor)
            calibrado_y_por_datos = m_y is not None

    def pixel_a_x(x_px):
        return None if m_x is None else m_x * x_px + b_x

    def pixel_a_y(y_px):
        return None if m_y is None else m_y * y_px + b_y

    # --- Puntos de la curva calibrados ---
    puntos_curva = []
    if mascara_curva is not None:
        filas_c, cols_c = np.where(mascara_curva)
        if cols_c.size > 0:
            for col in np.unique(cols_c):
                fila_prom = filas_c[cols_c == col].mean()
                puntos_curva.append((
                    int(col), float(fila_prom),
                    pixel_a_x(col), pixel_a_y(fila_prom),
                ))

    # --- Overlay de verificación ---
    overlay = imagen.copy()
    if eje_x:
        cv2.line(overlay, eje_x[:2], eje_x[2:], (0, 0, 255), 2)   # rojo
    if eje_y:
        cv2.line(overlay, eje_y[:2], eje_y[2:], (255, 0, 0), 2)   # azul
    if mascara_curva is not None:
        overlay[mascara_curva] = (0, 255, 0)                       # verde
    for e in etiquetas_x + etiquetas_y:
        cv2.rectangle(
            overlay, (e["px"], e["py"]),
            (e["px"] + e["pw"], e["py"] + e["ph"]), (0, 255, 255), 1,
        )
    if titulo_eje_y_bbox:
        b = titulo_eje_y_bbox
        cv2.rectangle(
            overlay, (b["px"], b["py"]),
            (b["px"] + b["pw"], b["py"] + b["ph"]), (255, 0, 255), 1,  # magenta
        )
    if titulo_x_bbox:
        b = titulo_x_bbox
        cv2.rectangle(
            overlay, (b["px"], b["py"]),
            (b["px"] + b["pw"], b["py"] + b["ph"]), (0, 165, 255), 1,  # naranja
        )
    for e in etiquetas_dato:
        cv2.rectangle(
            overlay, (e["px"], e["py"]),
            (e["px"] + e["pw"], e["py"] + e["ph"]), (128, 0, 128), 1,  # púrpura
        )

    nombre_overlay = f"overlay_{uid}.png"
    ruta_overlay = os.path.join(dir_resultados, nombre_overlay)
    cv2.imwrite(ruta_overlay, overlay)

    # --- Un solo CSV con todo: resumen, dimensiones de imagen, ejes completos,
    #     textos con su recuadro, y puntos de la curva ---
    nombre_csv = f"descripcion_{uid}.csv"
    ruta_csv = os.path.join(dir_resultados, nombre_csv)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tipo", "elemento", "x1", "y1", "x2", "y2",
            "ancho_px", "alto_px", "valor_x", "valor_y", "texto", "descripcion",
        ])

        # --- Resumen general (incluye dimensiones de la imagen) ---
        partes_resumen = []
        partes_resumen.append(f"título '{titulo}'" if titulo else "sin título detectado")
        partes_resumen.append("eje X detectado" if eje_x else "eje X no detectado")
        partes_resumen.append("eje Y detectado" if eje_y else "eje Y no detectado")
        partes_resumen.append(
            f"curva detectada ({len(puntos_curva)} puntos)" if mascara_curva is not None
            else "curva no detectada"
        )
        descripcion_general = (
            f"Gráfica de {ancho_imagen}x{alto_imagen} px con " + ", ".join(partes_resumen) + "."
        )
        w.writerow([
            "resumen", "grafica", "", "", "", "",
            ancho_imagen, alto_imagen, "", "", titulo, descripcion_general,
        ])

        # --- Ejes: se guardan los 4 extremos completos de cada línea (x1,y1,x2,y2) ---
        if eje_x:
            desc = f"Eje X (horizontal), de ({eje_x[0]},{eje_x[1]}) a ({eje_x[2]},{eje_x[3]})."
            w.writerow(["eje", "eje_x", eje_x[0], eje_x[1], eje_x[2], eje_x[3], "", "", "", "", "", desc])
        if eje_y:
            desc = f"Eje Y (vertical), de ({eje_y[0]},{eje_y[1]}) a ({eje_y[2]},{eje_y[3]})."
            w.writerow(["eje", "eje_y", eje_y[0], eje_y[1], eje_y[2], eje_y[3], "", "", "", "", "", desc])

        # --- Textos: título (con su recuadro completo) y etiquetas numéricas ---
        if titulo_bbox:
            desc = f"Título leído por OCR: '{titulo}', recuadro {titulo_bbox['pw']}x{titulo_bbox['ph']} px."
            w.writerow([
                "texto", "titulo", titulo_bbox["px"], titulo_bbox["py"], "", "",
                titulo_bbox["pw"], titulo_bbox["ph"], "", "", titulo, desc,
            ])
        elif titulo:
            w.writerow(["texto", "titulo", "", "", "", "", "", "", "", "", titulo, f"Título leído por OCR: '{titulo}'."])
        else:
            w.writerow(["texto", "titulo", "", "", "", "", "", "", "", "", "", "No se detectó título."])

        if titulo_eje_y_bbox:
            desc = (
                f"Título del eje Y leído por OCR (texto vertical): '{titulo_eje_y}', "
                f"recuadro {titulo_eje_y_bbox['pw']}x{titulo_eje_y_bbox['ph']} px."
            )
            w.writerow([
                "texto", "titulo_eje_y", titulo_eje_y_bbox["px"], titulo_eje_y_bbox["py"], "", "",
                titulo_eje_y_bbox["pw"], titulo_eje_y_bbox["ph"], "", "", titulo_eje_y, desc,
            ])
        elif titulo_eje_y:
            w.writerow([
                "texto", "titulo_eje_y", "", "", "", "", "", "", "", "", titulo_eje_y,
                f"Título del eje Y leído por OCR (texto vertical): '{titulo_eje_y}'.",
            ])
        else:
            w.writerow(["texto", "titulo_eje_y", "", "", "", "", "", "", "", "", "", "No se detectó título vertical del eje Y."])

        if titulo_x_bbox:
            desc = (
                f"Título del eje X leído por OCR: '{titulo_x}', "
                f"recuadro {titulo_x_bbox['pw']}x{titulo_x_bbox['ph']} px."
            )
            w.writerow([
                "texto", "titulo_eje_x", titulo_x_bbox["px"], titulo_x_bbox["py"], "", "",
                titulo_x_bbox["pw"], titulo_x_bbox["ph"], "", "", titulo_x, desc,
            ])
        elif titulo_x:
            w.writerow([
                "texto", "titulo_eje_x", "", "", "", "", "", "", "", "", titulo_x,
                f"Título del eje X leído por OCR: '{titulo_x}'.",
            ])
        else:
            w.writerow(["texto", "titulo_eje_x", "", "", "", "", "", "", "", "", "", "No se detectó título del eje X."])

        for e in etiquetas_dato:
            desc = (
                f"Etiqueta de dato (número pegado a la curva) con valor {e['valor']}, "
                f"recuadro en ({e['px']},{e['py']}) de {e['pw']}x{e['ph']} px."
                + (" Usada como calibración del eje Y por falta de etiquetas de eje." if calibrado_y_por_datos else "")
            )
            w.writerow([
                "texto", "etiqueta_dato", e["px"], e["py"], "", "",
                e["pw"], e["ph"], "", e["valor"], "", desc,
            ])

        for e in etiquetas_y:
            desc = (
                f"Etiqueta numérica del eje Y con valor {e['valor']}, "
                f"recuadro en ({e['px']},{e['py']}) de {e['pw']}x{e['ph']} px."
            )
            w.writerow([
                "texto", "etiqueta_eje_y", e["px"], e["py"], "", "",
                e["pw"], e["ph"], "", e["valor"], "", desc,
            ])

        for e in etiquetas_x:
            desc = (
                f"Etiqueta numérica del eje X con valor {e['valor']}, "
                f"recuadro en ({e['px']},{e['py']}) de {e['pw']}x{e['ph']} px."
            )
            w.writerow([
                "texto", "etiqueta_eje_x", e["px"], e["py"], "", "",
                e["pw"], e["ph"], e["valor"], "", "", desc,
            ])

        # --- Curva: puntos píxel + valor calibrado ---
        for col, fila, x_dato, y_dato in puntos_curva:
            if x_dato is not None and y_dato is not None:
                desc = f"Punto de la curva: valor real (x={x_dato:.3f}, y={y_dato:.3f}), píxel ({col}, {fila:.1f})."
            else:
                desc = f"Punto de la curva en píxel ({col}, {fila:.1f}); sin calibración suficiente para valor real."
            w.writerow([
                "curva", "punto", col, f"{fila:.1f}", "", "",
                "", "",
                "" if x_dato is None else f"{x_dato:.3f}",
                "" if y_dato is None else f"{y_dato:.3f}",
                "",
                desc,
            ])

    resumen = {
        "eje_x_detectado": eje_x is not None,
        "eje_y_detectado": eje_y is not None,
        "curva_detectada": mascara_curva is not None,
        "n_etiquetas_x": len(etiquetas_x),
        "n_etiquetas_y": len(etiquetas_y),
        "n_etiquetas_dato": len(etiquetas_dato),
        "titulo": titulo,
        "titulo_eje_y": titulo_eje_y,
        "titulo_eje_x": titulo_x,
        "n_puntos_curva": len(puntos_curva),
        "calibrado_x": m_x is not None,
        "calibrado_y": m_y is not None,
        "calibrado_y_por_datos": calibrado_y_por_datos,
        "ancho_imagen": ancho_imagen,
        "alto_imagen": alto_imagen,
    }

    return {
        "ruta_csv": ruta_csv, "nombre_csv": nombre_csv,
        "ruta_overlay": ruta_overlay, "nombre_overlay": nombre_overlay,
        "resumen": resumen,
    }