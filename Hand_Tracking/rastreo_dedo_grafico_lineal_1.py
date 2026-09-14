import cv2
import mediapipe as mp
import math
###import winsound

import pyttsx3
def hablar(texto):
    engine = pyttsx3.init()
    engine.setProperty("rate", 200)
    engine.setProperty("volume", 1.0)

    engine.say(texto)
    engine.runAndWait()
    engine.stop()

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ============================================================
# PUNTOS DE LA GRÁFICA
# ============================================================

puntos = {
    "Enero": {
        "x": 80,
        "y": 350,
        "valor": 11},
    "Febrero": {
        "x": 120,
        "y": 330,
        "valor": 13},
    "Marzo": {
        "x": 160,
        "y": 330,
        "valor": 13},
    "Abril": {
        "x": 200,
        "y": 280,
        "valor": 17},
    "Mayo": {
        "x": 240,
        "y": 250,
        "valor": 20},
    "Junio": {
        "x": 280,
        "y": 220,
        "valor": 23},
    "Julio": {
        "x": 320,
        "y": 180,
        "valor": 26},
    "Agosto": {
        "x": 360,
        "y": 170,
        "valor": 27},
    "Septiembre": {
        "x": 400,
        "y": 200,
        "valor": 25},
    "Octubre": {
        "x": 440,
        "y": 240,
        "valor": 21},
    "Noviembre": {
        "x": 480,
        "y": 290,
        "valor": 17},
    "Diciembre": {
        "x": 520,
        "y": 330,
        "valor": 14}
}


# ============================================================
# SEGMENTOS DE LA LÍNEA
# ============================================================

segmentos = []
nombres = list(puntos.keys())

for i in range(len(nombres) - 1):
    nombre1 = nombres[i]
    nombre2 = nombres[i + 1]

    p1 = puntos[nombre1]
    p2 = puntos[nombre2]

    valor1 = p1["valor"]
    valor2 = p2["valor"]

    # Determinar tendencia
    if valor2 > valor1:
        tendencia = "aumento"

    elif valor2 < valor1:
        tendencia = "disminución"

    else:
        tendencia = "estable"


    segmentos.append({
        "inicio": nombre1,
        "fin": nombre2,

        "x1": p1["x"],
        "y1": p1["y"],

        "x2": p2["x"],
        "y2": p2["y"],

        "tendencia": tendencia,

        "valor_inicial": valor1,
        "valor_final": valor2
    })


# ============================================================
# SONIDOS
# ============================================================

sonidos_puntos = {
    "Enero": 500,
    "Febrero": 550,
    "Marzo": 600,
    "Abril": 650,
    "Mayo": 700,
    "Junio": 750,
    "Julio": 800,
    "Agosto": 850,
    "Septiembre": 900,
    "Octubre": 950,
    "Noviembre": 1000,
    "Diciembre": 1050
}

sonidos_tendencia = {
    "aumento": 1200,
    "disminución": 400,
    "estable": 800}


# ============================================================
# DISTANCIA ENTRE DOS PUNTOS
# ============================================================
def distancia(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1)**2 +
        (y2 - y1)**2)

# ============================================================
# DISTANCIA DE UN PUNTO A UN SEGMENTO
# ============================================================
def distancia_segmento(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # Si el segmento tiene longitud 0
    if dx == 0 and dy == 0:
        return distancia(px, py, x1, y1)

    # Proyección del punto sobre el segmento
    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / (dx * dx + dy * dy)

    # Limitar al segmento
    t = max(0, min(1, t))

    # Punto más cercano del segmento
    cercano_x = x1 + t * dx
    cercano_y = y1 + t * dy

    return distancia(
        px,
        py,
        cercano_x,
        cercano_y)

elemento_anterior = None     # ELEMENTO ANTERIOR

# ============================================================
# MEDIAPIPE
# ============================================================
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        frame = cv2.flip(frame, 1)     # Espejo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # BGR → RGB
        results = hands.process(frame_rgb)     # Procesar mano
        elemento_actual = None

        # ====================================================
        # DETECTAR MANO
        # ====================================================
        if results.multi_hand_landmarks is not None:

            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark 8 = punta del índice
                dedo = hand_landmarks.landmark[8]

                x = int(dedo.x * width)
                y = int(dedo.y * height)

                # Dibujar dedo
                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (255, 0, 0),
                    -1
                )

                # ====================================================
                # 1. BUSCAR SI ESTÁ SOBRE UN PUNTO
                # ====================================================

                for nombre, punto in puntos.items():
                    d = distancia(
                        x,
                        y,
                        punto["x"],
                        punto["y"])

                    if d <= 20:
                        elemento_actual = (
                            "punto",
                            nombre)
                        break

                # ====================================================
                # 2. BUSCAR SI ESTÁ SOBRE UNA LÍNEA
                # ====================================================

                if elemento_actual is None:
                    for segmento in segmentos:
                        d = distancia_segmento(
                            x,
                            y,
                            segmento["x1"],
                            segmento["y1"],

                            segmento["x2"],
                            segmento["y2"])

                        if d <= 15:
                            elemento_actual = (
                                "tendencia",
                                segmento)
                            break

        # ====================================================
        # CAMBIO DE ELEMENTO
        # ====================================================
        # Convertimos el segmento en un identificador sencillo
        # para poder compararlo con el elemento anterior.

        if elemento_actual is not None:
            tipo = elemento_actual[0]
            if tipo == "punto":
                identificador = (
                    "punto",
                    elemento_actual[1])
            else:
                segmento = elemento_actual[1]
                identificador = (
                    "tendencia",
                    segmento["inicio"],
                    segmento["fin"])
        else:
            identificador = None

        # ====================================================
        # REPRODUCIR SONIDO
        # ====================================================
        if identificador != elemento_anterior:
            if elemento_actual is not None:
                tipo = elemento_actual[0]

                # --------------------------------------------
                # PUNTO
                # --------------------------------------------
                if tipo == "punto":
                    nombre = elemento_actual[1]
                    valor = puntos[nombre]["valor"]
                    texto = f"{nombre}. Temperatura de {valor} grados Celsius."

                    # frecuencia = sonidos_puntos[nombre]
                    # winsound.Beep(
                    #     frecuencia,
                    #     1000)  ##milisegundos
                    print(
                        nombre,
                        "=",
                        valor,
                        "°C")
                    hablar(texto)

                # --------------------------------------------
                # TENDENCIA
                # --------------------------------------------
                elif tipo == "tendencia":
                    segmento = elemento_actual[1]

                    inicio = segmento["inicio"]
                    fin = segmento["fin"]
                    valor_inicio = segmento["valor_inicial"]
                    valor_fin = segmento["valor_final"]

                    tendencia = segmento["tendencia"]
                    texto = (f"De {inicio} a {fin}, "
                             f"la temperatura presenta una tendencia de {tendencia}. "
                             f"Pasa de {valor_inicio} a {valor_fin} grados Celsius.")
                    
                    # frecuencia = sonidos_tendencia[tendencia]
                    # winsound.Beep(
                    #     frecuencia,
                    #     1500)  ##milisegundos
                    
                    print(
                        segmento["inicio"],
                        "→",
                        segmento["fin"],
                        ":",
                        tendencia)

                    hablar(texto)
                
        elemento_anterior = identificador

        # ====================================================
        # DIBUJAR PUNTOS
        # ====================================================
        for nombre, punto in puntos.items():
            cv2.circle(
                frame,
                (punto["x"], punto["y"]),
                6,
                (0, 0, 255),
                -1)

        # ====================================================
        # DIBUJAR LÍNEAS
        # ====================================================
        for segmento in segmentos:
            cv2.line(
                frame,
                (segmento["x1"],
                 segmento["y1"]),

                (segmento["x2"],
                 segmento["y2"]),

                (0, 255, 0),
                2
            )

        # ====================================================
        # MOSTRAR INFORMACIÓN
        # ====================================================
        if elemento_actual is not None:
            tipo = elemento_actual[0]

            if tipo == "punto":
                nombre = elemento_actual[1]
                texto = (
                    nombre +
                    ": " +
                    str(puntos[nombre]["valor"]) +
                    " °C")
            else:
                segmento = elemento_actual[1]
                texto = (
                    segmento["inicio"] +
                    " → " +
                    segmento["fin"] +
                    ": " +
                    segmento["tendencia"])

            cv2.putText(
                frame,
                texto,
                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,

                (0, 0, 255),
                2
            )

        # ====================================================
        # MOSTRAR
        # ====================================================
        cv2.imshow("Grafica interactiva", frame)

        # ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()