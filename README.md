# Asistente de Regresión Lineal Simple

Aplicación interactiva (Streamlit) para **explicar relaciones entre dos variables y estimar valores cuantitativos** con regresión lineal simple,
adaptada al contexto de cualquier negocio. Combina tres enfoques: *estadístico* (construcción y validación), *empresarial* (el contexto orienta qué
variables considerar) y *educativo* (cada resultado viene explicado).

> Probado con Python 3.12 y 3.14 y las versiones exactas de `requirements.txt` (Streamlit 1.67, pandas 3.0, numpy 2.4, scipy 1.17, statsmodels 0.14, plotly 6.7).
> **Desplegar en internet:** vea [`DESPLIEGUE.md`](DESPLIEGUE.md). Vercel **no** puede ejecutar Streamlit; el manual explica por qué y usa Streamlit Community Cloud.
> **¿Quiere una dirección en Vercel?** [`MANUAL_VERCEL.html`](MANUAL_VERCEL.html) (ábralo con doble clic) explica cómo publicar en Vercel una página que *muestra* la app alojada en Streamlit; la página ya está lista en [`vercel/`](vercel/).

## Instalación y uso

```bash
cd app_regresion
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Se abre en `http://localhost:8501`. Para empezar sin datos propios, en el **Módulo 1** elija un caso de ejemplo (9 sectores, datos *sintéticos*)
y pulse «Cargar ejemplo». Para usar sus datos: Módulo 2 → cargue un **CSV, TXT, XLS o XLSX**.
Dos bases de **1.000 registros** (salud y banca) y su guía de prueba están en [`sample_data/LEEME_BASES_DE_PRUEBA.md`](sample_data/LEEME_BASES_DE_PRUEBA.md).

```bash
python sample_data/generate.py                        # regenera los archivos de ejemplo
python -m pip install -r requirements-dev.txt         # una sola vez
python -m pytest tests -q                             # pruebas automáticas
```

## Los 13 módulos

| # | Módulo | Qué hace |
|---|---|---|
| 1 | Problema de negocio | Texto libre (sector, core, problema, resultado, factores, restricciones, pregunta). Resume lo que «entendió» y orienta Y y X. |
| 2 | Carga y exploración | TXT/CSV/XLS/XLSX; detecta separador, codificación, coma decimal, fechas, encabezado desplazado. Primeras 5 filas, tipos, duplicados, atípicos, inconsistencias, descriptivas. |
| 3 | Preparación | Faltantes (mapa, % y patrón), imputación **recomendada y evaluada** (media/mediana/moda/ffill/interpolación), atípicos, transformaciones, codificación, bitácora y antes/después. |
| 4 | Análisis exploratorio | Distribuciones, frecuencias, dispersión con LOWESS, Pearson y Spearman, linealidad vs. curvatura, tendencias y estacionalidad, lectura de negocio. |
| 5 | Variable Y | Ranking explicable de candidatas; usted confirma. Revisión de Y y transformación opcional (ln, √). |
| 6 | Variables X | Partición entrenamiento/prueba; preselección **solo con entrenamiento**; signo esperado y «se conoce antes»; alerta de fuga de información. |
| 7 | Modelos | Un modelo simple por candidata (y por variantes ln/√): β0, β1, r, R², R² ajustado, RMSE, MAE, EE, t, p, F, IC; recta con IC y IP; residuos. |
| 8 | Evaluación | Comparación con datos de prueba y validación cruzada (LOOCV), supuestos con semáforo, puntaje multicriterio, recomendación y alternativas. |
| 9 | Interpretación | Intercepto y pendiente: técnica y en lenguaje sencillo, viabilidad, ejemplos con datos reales. |
| 10 | Simulación | Escenarios pesimista/base/optimista, X interactivo, comparación base vs. simulado, IC e IP, alertas de extrapolación, «¿qué X necesito para esta meta?». |
| 11 | Sensibilidad | ±5 %, ±10 % y valores propios; elasticidad; interpretación para el negocio. |
| 12 | Informe ejecutivo | Hallazgos, calidad, estimaciones, riesgos y recomendaciones accionables. Descarga en HTML, Markdown y Excel. |
| 13 | Centro de aprendizaje | 19 conceptos (definición sencilla, técnica, ejemplo, interpretación, utilidad) y un laboratorio interactivo. |

## Arquitectura

```
app.py                  navegación por módulos, tema, configuración global
core/                   MOTOR (Python puro, sin Streamlit, probado de forma independiente)
  loader.py             lectura de archivos y detección de formatos
  profiling.py          tipos, duplicados, atípicos, inconsistencias, descriptivas
  cleaning.py           plan de limpieza reproducible (CleaningPlan) y aplicación anti-fuga
  eda.py                correlaciones, linealidad, patrones temporales
  context.py            contexto de negocio → sugerencia de Y/X y riesgo de fuga
  regression.py         MCO simple con fórmulas explícitas, IC/IP, Cook, PRESS, HC3
  diagnostics.py        linealidad, independencia, homocedasticidad, normalidad, influencia
  comparison.py         partición, un modelo por candidata, puntaje y recomendación
  simulation.py         escenarios, sensibilidad, extrapolación, regresión inversa
  interpretation.py     lectura de β0 y β1 (lin-lin, log-lin, lin-log, log-log)
  report.py             informe ejecutivo (Markdown/HTML)
  education.py          contenido del centro de aprendizaje
ui/                     una página por módulo + gráficos Plotly + componentes + estado
sample_data/            generador de datasets sintéticos y sus contextos de negocio
vercel/                 página estática (index.html + vercel.json) que muestra la app dentro de Vercel
MANUAL_VERCEL.html      manual paso a paso para publicar esa página (se abre con doble clic)
tests/                  motor, validación contra statsmodels, 7 sectores, humo de interfaz
```

## Decisiones metodológicas (por qué se comporta así)

- **Sin fuga de información.** La partición entrenamiento/prueba se hace *antes* de calcular medias, medianas, límites de atípicos o categorías: esas estadísticas
  se calculan solo con entrenamiento. La selección preliminar de X también usa solo entrenamiento. Con menos de 25 observaciones no se reserva prueba y se
  usa validación cruzada «dejando uno fuera» (PRESS, cerrada en regresión simple).
- **Y nunca se imputa.** Las filas sin Y se descartan; Y también está protegida de recortes.
- **No se elige por R².** El puntaje (0–100) combina capacidad predictiva en datos nuevos (40), supuestos (20), coherencia con el negocio (15), significancia (15)
  y calidad de datos (10), con penalizaciones por posible fuga, sobreajuste e influencia. Si hay un R² mayor que se descartó, la app explica por qué.
- **Fuga de información.** Una X con |r| ≥ 0,98 con Y, o que comparte concepto con Y y |r| ≥ 0,9, se marca y no se recomienda salvo que usted confirme que se conoce *antes*.
- **Honestidad.** Los estados son «recomendado», «con reservas» (p. ej. predice bien pero la inferencia es optimista por autocorrelación) y «no confiable»
  (R² < 0,10, casi sin mejora frente al promedio, curvatura, supuestos rotos). En ese caso se sugieren alternativas en vez de forzar un modelo.
- **IC vs. IP.** Se distinguen siempre: intervalo de confianza del promedio e intervalo de predicción de una observación individual.
- **Transformaciones.** ln(Y) o √Y se modelan en su escala y los resultados vuelven a la escala original (factor de Duan para ln; intervalos por transformación monótona).
- **El contexto de negocio NO usa un LLM.** Cruza el texto con los nombres de columna (raíces, sinónimos multisectoriales y coincidencia literal). Es una heurística
  explicable, no una comprensión semántica profunda: la decisión final es siempre suya.

## Pruebas

| Archivo | Qué verifica |
|---|---|
| `test_data_engine.py` | Carga (CSV/TXT con coma decimal/Excel con encabezado desplazado/.xls), perfilado, anti-fuga en limpieza, AED. |
| `test_regression.py` | Coeficientes, EE, t, p, F, R², IC, IP, Cook, PRESS y HC3 **idénticos a statsmodels**; Breusch-Pagan, White, Durbin-Watson. |
| `test_pipeline_datasets.py` | 9 sectores de extremo a extremo: fuga (retail), heterocedasticidad (logística), autocorrelación (energía), log-log (inmobiliario), costo fijo (manufactura), curvatura (agro), ausencia de relación (satisfacción), y las bases de 1.000 registros de **salud** y **banca** (fugas, cuota posterior, valores imposibles, y mejora real al limpiar). |
| `test_ui_smoke.py` | Recorre los 13 módulos con cada dataset, partición temporal y 99 % de confianza sin excepciones. |

## Cómo extenderla

- **Nuevo método estadístico:** agregue una función en `core/` que devuelva un objeto con la misma interfaz que `SLRResult` (`predict`, `coef_table`, `equation`)
  y regístrelo en `comparison.build_candidate`. Los módulos de interfaz solo consumen esa interfaz.
- **Regresión múltiple:** el motor está pensado para una X por modelo; para varias X, extienda `regression.fit_slr` a MCO matricial y `diagnostics` (la mayoría de
  las pruebas ya son válidas con más columnas).
- **Nuevo sector:** no requiere código; basta con describir el negocio en el Módulo 1 y cargar los datos.
- **Sinónimos de negocio:** `core/context.py → SYNONYM_GROUPS`.

## Limitaciones conocidas

- Los datos de ejemplo son simulados; no representan negocios reales.
- La detección de contexto es léxica: si los nombres de columna no se parecen a la descripción, el ranking depende más de la calidad de los datos.
- Un solo factor explicativo por modelo (por diseño). La app avisa cuándo conviene pasar a regresión múltiple.
- No se probó con pandas 2.x ni con Python distinto de 3.14.
- Los archivos `.xls` requieren `xlrd`; archivos con celdas combinadas o varias tablas por hoja pueden necesitar limpieza previa.
