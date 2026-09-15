# Chart Filter Pipeline

Extrae figuras de un PDF académico, descarta lo que no sirve (logos, tablas,
figuras compuestas de varios paneles), clasifica geométricamente si es un
gráfico de línea, y para los que sí lo son, extrae su tabla de datos
aproximada con el modelo DePlot (Google).

## Uso

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pipeline_rapido.py /ruta/al/paper.pdf
```

Los resultados quedan en `resultados_rapido/lineales_<nombre_pdf>/` (imágenes)
y `resultados_rapido/resultado_<nombre_pdf>.json` (datos extraídos).

## Qué hace, paso a paso

1. **Extracción barata (sin IA):** con PyMuPDF, saca imágenes raster
   embebidas y también dibujos vectoriales (gráficos hechos directo en el
   PDF con matplotlib/R/LaTeX, no como imagen pegada).
2. **Filtros baratos:** descarta por tamaño mínimo, duplicados, y una
   blacklist persistente de logos ya identificados en papers anteriores
   (se guarda en `resultados_rapido/hashes_no_son_graficos.json`).
3. **Detección de figuras compuestas:** cuenta cuántos "sistemas de ejes"
   distintos hay en la imagen (buscando líneas rectas largas). Si encuentra
   más de uno, la descarta — no separamos paneles individuales todavía.
4. **Clasificación geométrica línea vs. barra/torta:** mide qué tan "lleno"
   está el rectángulo que envuelve la forma principal de color. Una línea
   ocupa poco de su propio rectángulo (trazo fino); una barra o porción de
   torta lo llena casi entero.
5. **Extracción de datos (DePlot):** SOLO sobre lo que sobrevivió los pasos
   anteriores (no sobre todo lo extraído), se corre el modelo DePlot una
   única vez por ejecución para sacar la tabla de datos aproximada de cada
   gráfico de línea.

## Limitaciones conocidas (para quien retome esto)

- **Figuras compuestas de varios paneles** (ej. Figura con sub-paneles a,b,c,d)
  se descartan enteras por ahora — no hay lógica para separar cada panel
  individual todavía. Es la mejora más importante pendiente.
- El detector de "compuesta" es un heurístico de geometría (cuenta líneas
  rectas largas), no siempre generaliza bien entre distintos estilos de
  gráfico (Excel vs matplotlib vs Origin/SigmaPlot pueden dibujar los bordes
  con distinto grosor/color). Umbral actual calibrado sobre ~10 papers de
  prueba, puede necesitar ajuste con más datos.
- Los datos que devuelve DePlot son una **reconstrucción aproximada por IA**,
  no una lectura exacta de píxeles — no reemplaza una digitalización real
  del gráfico (trazar la curva, ubicar la leyenda con coordenadas exactas,
  detectar cruces de líneas). Si el proyecto necesita eso a futuro
  (relevante para accesibilidad táctil), es una pieza aparte por construir,
  ningún repo probado (ChartSenseAI, ChartReader, pdffigures2) lo resuelve
  hoy.
- No requiere Nyckel, comtypes, docx2pdf, tensorflow, langchain ni ninguna
  otra dependencia del ChartSenseAI original — se reescribió la extracción
  de datos directo con `transformers`/`torch`, sin esas dependencias.

## Por qué esta versión y no otras que se probaron

- **pdffigures2** (extractor especializado en papers académicos, Java/Scala):
  mucho más rápido, pero falla en detectar figuras cuando hay varias
  apiladas muy cerca en una misma columna (limitación conocida de la
  herramienta). En las pruebas, en varios papers solo detectaba 1 de 3
  gráficos de línea reales.
- **ChartSenseAI tal cual (sin modificar):** clasificaba bien, pero cargaba
  el modelo DePlot completo (1.1 GB) por cada imagen extraída, incluyendo
  logos y basura — larguísimo. Esta versión (`pipeline_rapido.py`) reescribe
  la extracción para cargar el modelo una sola vez y correrlo solo sobre lo
  que ya se filtró como gráfico de línea válido.
