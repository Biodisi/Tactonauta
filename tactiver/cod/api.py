# -*- coding: utf-8 -*-
"""
api.py
======
API HTTP mínima para conectar el pipeline automático de segmentación
de gráficas con una interfaz web YA EXISTENTE (cualquier stack: React,
Vue, PHP, HTML plano, etc.). No trae interfaz propia: solo expone
endpoints que tu frontend puede llamar directamente.

Ejecutar:
    pip install -r requirements.txt
    python api.py
    (queda escuchando en http://localhost:5000)

------------------------------------------------------------------
ENDPOINTS
------------------------------------------------------------------

POST /api/procesar
    Recibe una imagen (multipart/form-data, campo "imagen") y devuelve
    un JSON con el resumen de lo detectado + las URLs para descargar
    el CSV y el overlay de verificación.

    curl -F "imagen=@grafica.png" http://localhost:5000/api/procesar

    Respuesta:
    {
      "ok": true,
      "resumen": { ... },
      "csv_url": "/api/resultados/descripcion_xxxx.csv",
      "csv_download_url": "/api/resultados/descripcion_xxxx.csv/descargar",
      "overlay_url": "/api/resultados/overlay_xxxx.png"
    }

POST /api/procesar?formato=csv
    Igual que arriba, pero devuelve DIRECTAMENTE el archivo CSV como
    descarga (útil si tu interfaz solo necesita el archivo, sin JSON
    intermedio).

    curl -F "imagen=@grafica.png" "http://localhost:5000/api/procesar?formato=csv" -o resultado.csv

GET /api/resultados/<nombre_archivo>
    Sirve un archivo ya generado (CSV u overlay PNG) para visualizarlo
    inline (por ejemplo, para mostrar el overlay en un <img> de tu web).

GET /api/resultados/<nombre_archivo>/descargar
    Igual, pero forzando la descarga (Content-Disposition: attachment).

GET /api/salud
    Chequeo simple de que la API está viva: {"ok": true}
------------------------------------------------------------------
"""

import os
import uuid

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from segmentador import procesar_imagen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
EXTENSIONES_VALIDAS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB máx por imagen


@app.after_request
def habilitar_cors(response):
    # CORS abierto para que tu interfaz (en otro dominio/puerto) pueda
    # llamar a esta API sin problemas. Restringe el origen si lo necesitas,
    # p. ej. response.headers["Access-Control-Allow-Origin"] = "https://tu-web.com"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def extension_valida(nombre_archivo):
    return "." in nombre_archivo and \
        nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_VALIDAS


@app.route("/api/salud", methods=["GET"])
def salud():
    return jsonify({"ok": True})


@app.route("/api/procesar", methods=["POST"])
def procesar():
    archivo = request.files.get("imagen")

    if archivo is None or archivo.filename == "":
        return jsonify({"ok": False, "error": "No se envió ninguna imagen (campo 'imagen')."}), 400

    if not extension_valida(archivo.filename):
        return jsonify({"ok": False, "error": "Formato no soportado. Usa PNG, JPG, JPEG, BMP o TIFF."}), 400

    nombre_seguro = secure_filename(archivo.filename)
    nombre_unico = f"{uuid.uuid4().hex[:8]}_{nombre_seguro}"
    ruta_subida = os.path.join(UPLOAD_DIR, nombre_unico)
    archivo.save(ruta_subida)

    try:
        resultado = procesar_imagen(ruta_subida, RESULTADOS_DIR)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al procesar la imagen: {e}"}), 500

    # Si el frontend solo quiere el archivo CSV directo (sin JSON):
    if request.args.get("formato") == "csv":
        return send_from_directory(RESULTADOS_DIR, resultado["nombre_csv"], as_attachment=True)

    return jsonify({
        "ok": True,
        "resumen": resultado["resumen"],
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
    })


@app.route("/api/resultados/<path:nombre_archivo>", methods=["GET"])
def servir_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=False)


@app.route("/api/resultados/<path:nombre_archivo>/descargar", methods=["GET"])
def descargar_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)