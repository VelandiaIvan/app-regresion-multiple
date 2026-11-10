# Bases de prueba de 1.000 registros

Dos bases **sintéticas** (generadas con semilla fija: se regeneran idénticas con `python sample_data/generate.py`). No representan
hospitales ni bancos reales. Cada una trae *suciedad* y *trampas* a propósito para practicar cada módulo de la app.

| | `atencion_hospitalaria.csv` | `cartera_credito.csv` |
|---|---|---|
| **Sector** | Salud (clínica con hospitalización) | Banca (crédito de libre inversión) |
| **Registros** | 1.000 (990 únicos + 10 duplicados exactos) | 1.000 (990 únicos + 10 duplicados exactos) |
| **Columnas** | 12 | 14 |
| **Y a estimar** | `costo_total_millones` | `monto_aprobado_millones` |
| **Fechas** | `fecha_ingreso` (2024-2025, ordenada) | `fecha_desembolso` (2023-2025, ordenada) |
| **Codificación** | UTF-8 con BOM (Excel muestra bien las tildes) | UTF-8 con BOM |

Se abren con doble clic en Excel, y también están en el selector de ejemplos del módulo 1 de la app (*Carga de datos*).

## Columnas y trampas

### Salud · `atencion_hospitalaria.csv`

| Columna | Qué es | Trampa / uso |
|---|---|---|
| `id_paciente` | Identificador | Debe excluirse de los modelos. |
| `fecha_ingreso` | Fecha | Orden temporal para la partición temporal. |
| `servicio`, `tipo_afiliacion` | Categóricas | Para codificar (módulo 3) y ver diferencias por grupo (módulo 4). |
| `edad` | Años | 3 % faltante. |
| `comorbilidades`, `nivel_complejidad` | Conteo / ordinal 1–4 | Poca cantidad de valores distintos. |
| `indice_masa_corporal` | IMC | 8 % faltante y **sin relación real** con el costo (distractor). |
| `dias_estancia` | Días | **Mejor predictor legítimo.** 2 valores **negativos** (error de captura). |
| `num_procedimientos` | Conteo | 2 % faltante; segundo mejor predictor. |
| `costo_total_millones` | **Y** | Costo multiplicativo → varianza crece con el nivel; 6 casos extremos. |
| `costo_facturado_eps_millones` | Facturado a la EPS | ⚠ **Fuga**: ≈ 0,94 × costo total. R² ≈ 1,00. |

### Banca · `cartera_credito.csv`

| Columna | Qué es | Trampa / uso |
|---|---|---|
| `id_credito` | Identificador | Debe excluirse. |
| `fecha_desembolso` | Fecha | Orden temporal. |
| `segmento`, `ciudad` | Categóricas | Para codificar / segmentar. |
| `edad`, `plazo_meses`, `tasa_ea_pct` | Perfil / condiciones | Prácticamente sin relación con el monto. |
| `ingreso_mensual_millones` | Ingreso del cliente | **Mejor predictor legítimo.** 4 % faltante, 2 valores **negativos**, 3 con **error de digitación** (×6). |
| `antiguedad_laboral_anios`, `score_crediticio`, `deuda_vigente_millones` | Perfil | Faltantes 3 %, 6 %, 2 %; efecto menor. |
| `monto_aprobado_millones` | **Y** | Tope de política de 120 millones; dispersión mayor a mayores ingresos. |
| `monto_desembolsado_millones` | Monto desembolsado | ⚠ **Fuga**: ≈ el monto aprobado. R² ≈ 1,00. |
| `cuota_mensual_millones` | Cuota | ⚠ Se calcula **después** del monto: no está disponible para decidirlo. |

## Receta de prueba, módulo por módulo

1. **Módulo 1** → «¿No tiene datos a la mano?» → elija el ejemplo → *Cargar ejemplo*. En el **módulo 2** la app explica los datos y sugiere la variable Y.
2. **Módulo 2** → confirme 1.000 filas, 10 duplicados y las inconsistencias (valores negativos).
3. **Módulo 5** → la app debe recomendar **la Y de la tabla** (no la variable con fuga).
4. **Módulo 6** → `costo_facturado_eps_millones` / `monto_desembolsado_millones` aparecen con **riesgo de fuga alto** y sin marcar; `cuota_mensual_millones` con riesgo medio.
5. **Módulos 7 y 8** *sin limpiar nada*: el estado será **«no confiable»**. Ese es el punto de partida.
6. **Vuelva al módulo 3** y aplique: eliminar duplicados, convertir negativos a faltante, imputar faltantes con la recomendación de la app.
   - *Banca:* además, **eliminar filas atípicas del ingreso** (solo entrenamiento, k = 3).
   - *Salud:* en el módulo 5 pruebe **ln(Y)**.
7. **Módulo 8** de nuevo: el estado mejora a **«con reservas»**.

### Resultados de referencia (partición aleatoria 20 %, semilla 42, confianza 95 %)

Las cifras exactas cambian con la semilla y la partición; sirven para saber si la app se comporta como se espera.

| Escenario | Mejor modelo | R² (entren.) | R² (prueba) | Estado |
|---|---|---|---|---|
| Salud · sin limpiar | `dias_estancia` | ≈ 0,55 | ≈ 0,49 | no confiable (varianza no constante y residuos no normales) |
| Salud · limpia + ln(Y) | `dias_estancia` / raíz | ≈ 0,50 | ≈ 0,57 | con reservas |
| Banca · sin limpiar | `ingreso_mensual_millones` | ≈ 0,66 | ≈ 0,80 | no confiable (atípicos, negativos y heterocedasticidad) |
| Banca · limpia (atípicos eliminados en entrenamiento) | `ingreso_mensual_millones` | ≈ 0,60 | ≈ 0,82 | con reservas |

### Una lección incluida a propósito (banca)

Si en el módulo 3 convierte los ingresos altos en *faltantes* y los rellena con la mediana, el R² del ingreso **baja de ≈ 0,65 a ≈ 0,36**:
esos ingresos altos eran **reales** (segmento Empresario) y eran justamente los que sostenían la relación. Un atípico no es siempre un error.

## Regenerar o cambiar las bases

```powershell
python sample_data/generate.py
```

Las funciones `salud_hospitalario()` y `banca_credito()` están en `sample_data/generate.py`; cambie la semilla (`seed`) para obtener otra versión.


## `ventas_restaurante.csv` (archivo real de práctica, 186 registros)

Cuatro columnas separadas por `;` y con coma decimal: `Bowl Price` (precio del plato), `Bowls` (platos vendidos), `Soda` y `Beer` (bebidas vendidas).
No tiene contexto escrito: sirve para comprobar que la app, por sí sola, sugiere **`Bowls`** como variable a estimar y arma un modelo múltiple
(`Soda + Bowl Price`, R² ajustado ≈ 0,76; el precio tiene coeficiente negativo). Cárguelo desde el módulo 1 con «Upload».
