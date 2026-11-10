# Asistente de Regresión Lineal Múltiple

Aplicación interactiva (Streamlit) para **estimar un resultado numérico con dos o más variables a la vez** mediante regresión lineal múltiple.
Usted carga un archivo y la aplicación hace el resto: explica de qué tratan los datos, propone la variable a estimar (Y), prueba las combinaciones
de variables explicativas (X), elige el mejor modelo y lo interpreta en lenguaje sencillo.

> **Proyecto académico** de Ivan Alonso Velandia Duarte ([VelandiaIvan](https://github.com/VelandiaIvan)), estudiante de la Maestría en Analítica de Datos.
> No es un producto comercial: los resultados tienen fines educativos y no constituyen asesoría profesional. La app no guarda los archivos cargados;
> aun así, use datos de ejemplo o anonimizados. Parte del código se generó y revisó con herramientas de inteligencia artificial.

> Probado con Python 3.12 y las versiones exactas de `requirements.txt` (Streamlit 1.67, pandas 3.0, numpy 2.4, scipy 1.17, statsmodels 0.14, plotly 6.7).
> **Desplegar o actualizar en internet:** vea [`DESPLIEGUE.md`](DESPLIEGUE.md) (Streamlit Community Cloud).

## En qué se diferencia de la app de regresión simple

| | Regresión simple | **Esta app (múltiple)** |
|---|---|---|
| Modelo | Un modelo por cada X: `Ŷ = β0 + β1·X` | Combinaciones de **2 o más X**: `Ŷ = β0 + β1·X1 + β2·X2 + …` |
| Mejor modelo | La mejor variable individual | La mejor **combinación**, con todas sus variables aportando |
| Inicio | Formulario de negocio de 8 campos | **Se cargan los datos primero**; la app los explica y sugiere Y. El contexto escrito es opcional (2 preguntas) |
| Diagnósticos | 4 supuestos + influencia | Los mismos **+ multicolinealidad (VIF)** y R² ajustado |
| Simulación | Se mueve una X | Se mueve **cada X por separado**; descomposición del efecto y «tornado» de sensibilidad |
| Colores | Azul | Verde azulado / ámbar |

## Instalación y uso

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Se abre en `http://localhost:8501`. En el **módulo 1** cargue un **CSV, TXT, XLS o XLSX** (una fila por caso, una columna por variable; se necesita
una columna numérica a estimar y al menos dos más que la expliquen) o pruebe uno de los 9 casos de ejemplo sintéticos.

```bash
python -m pip install -r requirements-dev.txt         # una sola vez
python -m pytest tests -q                             # 79 pruebas automáticas
```

## Los 13 módulos

| # | Módulo | Qué hace |
|---|---|---|
| 1 | Carga de datos | TXT/CSV/XLS/XLSX; detecta separador, codificación, coma decimal y fechas. Tipos, duplicados, atípicos, descriptivas. Incluye los ejemplos. |
| 2 | Qué dicen sus datos | **Lectura automática**: tema de la tabla, qué parece medir cada columna, calidad, y **variable Y sugerida con sus razones**. Contexto opcional. |
| 3 | Preparación | Faltantes, imputación recomendada y evaluada, atípicos, transformaciones, codificación de categóricas (dummies), bitácora. |
| 4 | Análisis exploratorio | Distribuciones, dispersión, correlaciones de Pearson y Spearman, linealidad, patrones en el tiempo. |
| 5 | Variable Y | Confirmar la variable a estimar y revisar su distribución (se modela en su propia escala). |
| 6 | Variables X | Partición entrenamiento/prueba; candidatas (mínimo 2); signo esperado, «se conoce antes» y alerta de fuga de información. |
| 7 | Modelos | Combinaciones de 2+ variables: coeficientes, EE, t, p, IC, β estandarizado, VIF, R² ajustado, F; comparación contra cada X por sí sola. |
| 8 | Evaluación | Datos de prueba y validación cruzada (LOOCV), supuestos con semáforo (incluye VIF), puntaje multicriterio, recomendación y alternativas. |
| 9 | Interpretación | Cada coeficiente «manteniendo constantes las demás», peso relativo de las variables, ejemplos con datos reales. |
| 10 | Simulación | Escenarios pesimista/base/optimista, un control por variable, aporte de cada cambio, IC e IP, alertas de extrapolación, meta inversa. |
| 11 | Sensibilidad | Qué variable mueve más a Y (tornado ±10 %) y detalle ±5 %, ±10 % o valores propios, una variable a la vez. |
| 12 | Informe ejecutivo | Hallazgos, calidad, estimaciones, riesgos y recomendaciones. Descarga en HTML, Markdown y Excel. |
| 13 | Centro de aprendizaje | 25 conceptos (de la regresión simple a la múltiple: efecto parcial, R² ajustado, VIF, selección de variables, dummies) y un laboratorio. |

## Arquitectura

```
app.py                     navegación por módulos, tema, configuración global
core/                      MOTOR (Python puro, sin Streamlit, probado de forma independiente)
  loader.py                lectura de archivos y detección de formatos
  profiling.py             tipos, duplicados, atípicos, inconsistencias, descriptivas
  cleaning.py              plan de limpieza reproducible y aplicación anti-fuga
  eda.py                   correlaciones, linealidad, patrones temporales
  understanding.py         «¿de qué tratan los datos?»: tema, papel de cada columna, explicación de la Y sugerida
  context.py               ranking de candidatas a Y (incluye «qué columna se explica mejor con las demás») y riesgo de fuga
  multiple.py              MCO matricial: β=(XᵀX)⁻¹Xᵀy, EE, t, p, IC, F, R² ajustado, VIF, Cook, PRESS, HC3, IC/IP de predicción
  selection.py             búsqueda de combinaciones de 2+ variables, puntaje multicriterio, parsimonia y recomendación
  diagnostics.py           RESET, Breusch-Pagan/White, Shapiro, Durbin-Watson/Breusch-Godfrey, VIF, influencia
  interpretation_multi.py  efectos parciales, importancia relativa, lenguaje sencillo
  simulation_multi.py      escenarios, simulación multivariable, sensibilidad, tornado, meta inversa
  report_multi.py          informe ejecutivo (Markdown/HTML)
  regression.py, comparison.py, simulation.py, interpretation.py, report.py
                           regresión simple: se conserva como base (partición, métricas) y referencia
  education.py             contenido del centro de aprendizaje
ui/                        una página por módulo + gráficos Plotly + componentes + estado
sample_data/               datasets sintéticos de ejemplo y un archivo real de práctica (ventas_restaurante.csv)
tests/                     motor (validado contra statsmodels), selección, 9 sectores y humo de interfaz
```

## Cómo se elige el mejor modelo

1. **Se apartan** las variables que no estarán disponibles al decidir y las que tienen riesgo alto de fuga de información (|r| ≥ 0,98 con Y).
2. Con hasta 9 candidatas se evalúan **todas las combinaciones de 2 o más**; con más, selección paso a paso (adelante y atrás).
3. Cada combinación se mide con validación cruzada «dejando uno fuera», usando solo datos de entrenamiento.
4. Las finalistas se ajustan completas y reciben un **puntaje (0–100)**: capacidad predictiva con datos nuevos (40), supuestos incluida la
   multicolinealidad (20), significancia global y de cada coeficiente (15), coherencia con el negocio (15) y calidad de datos (10); se restan puntos por
   cada variable sin efecto significativo, por sobreajuste, VIF > 10 y observaciones muy influyentes.
5. **No se elige por R²** (siempre ganaría el modelo con más variables). Si dos modelos quedan a menos de 1 punto, gana el de menos variables.
6. El estado final es «recomendado», «con reservas» o «no confiable», con sus razones y alternativas.

## Otras decisiones metodológicas

- **Sin fuga de información:** la partición entrenamiento/prueba se hace antes de calcular estadísticas de limpieza. **Y nunca se imputa.**
- **Sugerencia de Y sin texto:** combina qué tan bien se explica cada columna con las demás (45), si su nombre es un resultado típico (20), calidad
  de datos (20) y conexión con otras variables (15). Un precio que se mueve en sentido contrario al resto se trata como factor, no como resultado.
- **Sin modelo de lenguaje:** el «entendimiento» de los datos es léxico y estadístico (diccionario de conceptos + tipos + correlaciones). Es una guía explicable.
- **IC vs. IP:** siempre se distingue el intervalo de confianza del promedio y el de predicción de un caso individual.

## Limitaciones conocidas

- Los datos de ejemplo son simulados. `ventas_restaurante.csv` es un archivo de práctica de clase.
- Solo relaciones lineales entre las X y Y; las curvaturas se detectan y se sugiere transformar en el módulo de preparación. No incluye interacciones automáticas.
- Las variables categóricas deben codificarse (módulo 3) para entrar al modelo.
- Con muchas variables y pocos datos los coeficientes son inestables: regla práctica, 10–15 observaciones por variable.
- Los coeficientes describen asociaciones, no causalidad.
