import hashlib
import json
import os
import shutil
import sys

import fitz  # PyMuPDF
from PIL import Image

from graficos_tactiles.tactiverso.backend.chart_filter_pipeline.classify_charts import es_grafico_lineal

AREA_MINIMA = 15000
CARPETA_RESULTADOS = "resultados_rapido"
BLACKLIST_PATH = os.path.join(CARPETA_RESULTADOS, "hashes_no_son_graficos.json")

_deplot_model = None
_deplot_processor = None


def cargar_deplot():
    global _deplot_model, _deplot_processor
    if _deplot_model is None:
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        print("Cargando modelo DePlot (solo una vez)...")
        torch.set_default_device("cpu")
        _deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
        _deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    return _deplot_model, _deplot_processor


def extraer_datos_del_grafico(image_path):
    model, processor = cargar_deplot()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image,
        text="Generate underlying data table of the figure below:",
        return_tensors="pt",
    )
    predictions = model.generate(**inputs, max_new_tokens=512)
    resultado = processor.decode(predictions[0], skip_special_tokens=True)
    return resultado.replace("<0x0A>", "\n")


def hash_archivo(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def es_proporcion_de_logo(w, h, tolerancia=0.25, area_maxima_logo=60000):
    area = w * h
    if area > area_maxima_logo:
        return False  # muy grande para ser un ícono/logo, sea cuadrada o no
    ratio = w / h if h else 0
    return (1 - tolerancia) <= ratio <= (1 + tolerancia)


def cargar_blacklist():
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH) as f:
            return set(json.load(f))
    return set()


def guardar_blacklist(hashes):
    with open(BLACKLIST_PATH, "w") as f:
        json.dump(sorted(hashes), f, indent=2)


def _agrupar_rects(rects, umbral=15):
    """Agrupa rectángulos de trazos vectoriales cercanos/superpuestos en un solo bloque."""
    grupos = [fitz.Rect(r) for r in rects]
    cambiado = True
    while cambiado:
        cambiado = False
        nuevos = []
        usados = [False] * len(grupos)
        for i in range(len(grupos)):
            if usados[i]:
                continue
            actual = fitz.Rect(grupos[i])
            for j in range(i + 1, len(grupos)):
                if usados[j]:
                    continue
                otro = grupos[j]
                cercano = (
                    actual.x0 - umbral <= otro.x1 and otro.x0 - umbral <= actual.x1 and
                    actual.y0 - umbral <= otro.y1 and otro.y0 - umbral <= actual.y1
                )
                if cercano:
                    actual |= otro
                    usados[j] = True
                    cambiado = True
            nuevos.append(actual)
            usados[i] = True
        grupos = nuevos
    return grupos


def extraer_dibujos_vectoriales(page, page_num, output_dir, contador_inicial):
    """Detecta gráficos dibujados directamente con vectores (matplotlib/R/LaTeX),
    agrupando los trazos sueltos (líneas, ejes, leyenda) en bloques y renderizándolos
    como imagen."""
    extraidas = []
    drawings = page.get_drawings()
    if not drawings:
        return extraidas, contador_inicial

    rects = [d["rect"] for d in drawings if d.get("rect") and d["rect"].width > 2 and d["rect"].height > 2]
    if not rects:
        return extraidas, contador_inicial

    grupos = _agrupar_rects(rects, umbral=15)

    contador = contador_inicial
    for g in grupos:
        if g.width < 80 or g.height < 60:  # descarta subrayados/viñetas sueltas
            continue
        contador += 1
        pad = 3
        clip = fitz.Rect(g.x0 - pad, g.y0 - pad, g.x1 + pad, g.y1 + pad)
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2))
        nombre = f"page{page_num+1}_drawing{contador}.png"
        ruta = os.path.join(output_dir, nombre)
        pix.save(ruta)
        extraidas.append({"file_path": ruta, "page": page_num + 1, "width": clip.width, "height": clip.height})

    return extraidas, contador


def extraer_imagenes_crudas(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extraidas = []

    for page_num, page in enumerate(doc):
        # 1. Imágenes raster embebidas (fotos/capturas pegadas, ej. exportadas de Excel)
        for i, im in enumerate(page.get_image_info(xrefs=True)):
            xref = im["xref"]
            w = im["bbox"][2] - im["bbox"][0]
            h = im["bbox"][3] - im["bbox"][1]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            nombre = f"page{page_num+1}_img{i+1}.png"
            ruta = os.path.join(output_dir, nombre)
            pix.save(ruta)
            extraidas.append({"file_path": ruta, "page": page_num + 1, "width": w, "height": h})

        # 2. Dibujos vectoriales (gráficos hechos con matplotlib/R/LaTeX directo en el PDF)
        vectoriales, _ = extraer_dibujos_vectoriales(page, page_num, output_dir, 0)
        extraidas.extend(vectoriales)

    return extraidas


def procesar_pdf(pdf_path):
    nombre_base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(CARPETA_RESULTADOS, f"extracted_{nombre_base}")
    lineales_dir = os.path.join(CARPETA_RESULTADOS, f"lineales_{nombre_base}")

    print(f"\n{'='*50}\nProcesando: {pdf_path}\n{'='*50}")

    if not os.path.exists(pdf_path):
        print("  ⚠️  No se encontró el archivo.")
        return

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    blacklist = cargar_blacklist()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. Extracción barata (sin IA): imágenes raster + dibujos vectoriales
    extraidas = extraer_imagenes_crudas(pdf_path, output_dir)
    print(f"Extraídas en bruto: {len(extraidas)}")

    # 2. Filtros baratos: tamaño, duplicados, blacklist, proporción (CON DIAGNÓSTICO)
    vistos = set()
    candidatos = []
    nuevos_logos = set()

    for r in extraidas:
        area = r["width"] * r["height"]
        nombre_img = os.path.basename(r["file_path"])

        if area < AREA_MINIMA:
            print(f"  ⏭️  {nombre_img}: muy chica ({r['width']:.0f}x{r['height']:.0f}={area:.0f}px²), descartada")
            continue

        h = hash_archivo(r["file_path"])
        if h in vistos:
            print(f"  ⏭️  {nombre_img}: duplicada dentro del mismo PDF, descartada")
            continue
        vistos.add(h)

        if h in blacklist:
            print(f"  ⏭️  {nombre_img}: en blacklist de logos (de un paper anterior), descartada")
            continue

        if es_proporcion_de_logo(r["width"], r["height"]):
            print(f"  🔲 {nombre_img}: proporción cuadrada ({r['width']:.0f}x{r['height']:.0f}) -> logo, descartado")
            nuevos_logos.add(h)
            continue

        print(f"  ✅ {nombre_img}: pasa filtros baratos ({r['width']:.0f}x{r['height']:.0f})")
        candidatos.append(r)

    print(f"Después de filtros baratos: {len(candidatos)}")

    # 3. Clasificación geométrica (barata, sin IA)
    lineales = []
    for r in candidatos:
        es_lineal, confianza, etiqueta = es_grafico_lineal(r["file_path"])
        estado = "LINEAL" if es_lineal else f"descartado ({etiqueta})"
        print(f"  {os.path.basename(r['file_path'])}: {estado} (conf {confianza:.2f})")
        if es_lineal:
            r["clasificacion_confianza"] = confianza
            lineales.append(r)

    print(f"Gráficos lineales: {len(lineales)} de {len(extraidas)} extraídos")

    # 4. SOLO ahora, sobre lo que sobrevivió, corremos el modelo pesado
    for r in lineales:
        print(f"  Extrayendo datos/ejes de {os.path.basename(r['file_path'])} con DePlot...")
        r["data_table"] = extraer_datos_del_grafico(r["file_path"])

    if os.path.exists(lineales_dir):
        shutil.rmtree(lineales_dir)
    os.makedirs(lineales_dir)
    for r in lineales:
        shutil.copy(r["file_path"], os.path.join(lineales_dir, os.path.basename(r["file_path"])))

    blacklist.update(nuevos_logos)
    guardar_blacklist(blacklist)

    with open(os.path.join(CARPETA_RESULTADOS, f"resultado_{nombre_base}.json"), "w") as f:
        json.dump(lineales, f, indent=2, ensure_ascii=False)

    print(f"\nGuardados en: {lineales_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pipeline_rapido.py /ruta/al/paper.pdf")
        sys.exit(1)
    procesar_pdf(sys.argv[1])