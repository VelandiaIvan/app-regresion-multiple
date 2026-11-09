"""Genera conjuntos de datos SINTÉTICOS de varios sectores para probar la aplicación.

Todos los datos son simulados (semilla fija): sirven para demostrar el método, no representan
negocios reales. Cada conjunto incluye «suciedad» intencional (faltantes, duplicados, atípicos,
formatos regionales) y, en algunos casos, trampas metodológicas (fuga de información, relación
no lineal, heterocedasticidad, autocorrelación, ausencia de relación).

Uso:  python sample_data/generate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _holes(rng, s: pd.Series, frac: float) -> pd.Series:
    s = s.astype(float).copy()
    s.iloc[rng.choice(len(s), int(len(s) * frac), replace=False)] = np.nan
    return s


def retail_ventas(seed: int = 1) -> pd.DataFrame:
    """Retail: ventas semanales por tienda. Trampa: `ingresos_netos_millones` ≈ función de Y (fuga)."""
    rng = np.random.default_rng(seed)
    n = 240
    fecha = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    ciudad = rng.choice(["Bogotá", "Medellín", "Cali", "Manizales"], n, p=[.35, .3, .2, .15])
    publicidad = rng.gamma(4, 6, n).round(1)                         # millones COP
    trafico = (900 + 11 * publicidad + rng.normal(0, 120, n)).round(0)
    descuento = rng.choice([0, 5, 10, 15, 20], n, p=[.35, .25, .2, .1, .1]).astype(float)
    temp = (18 + 4 * np.sin(np.arange(n) / 8) + rng.normal(0, 1.5, n)).round(1)
    ventas = 140 + 3.4 * publicidad + 0.045 * trafico + 1.1 * descuento + rng.normal(0, 28, n)
    ventas[rng.choice(n, 3, replace=False)] *= 1.9                    # picos atípicos (promociones)
    df = pd.DataFrame({
        "id_registro": np.arange(1, n + 1), "semana": fecha, "ciudad": ciudad,
        "inversion_publicidad_millones": publicidad, "trafico_clientes": trafico,
        "descuento_pct": descuento, "temperatura_c": temp,
        "ventas_millones": ventas.round(1),
    })
    df["ingresos_netos_millones"] = (df["ventas_millones"] * 0.82 + rng.normal(0, 1.2, n)).round(1)   # FUGA
    df["trafico_clientes"] = _holes(rng, df["trafico_clientes"], 0.06)
    df["descuento_pct"] = _holes(rng, df["descuento_pct"], 0.04)
    df.loc[rng.choice(n, 6, replace=False), "ciudad"] = rng.choice(["medellin ", "BOGOTA", "cali"], 6)  # etiquetas sucias
    df = pd.concat([df, df.sample(3, random_state=seed)], ignore_index=True)                            # duplicados
    return df


def logistica_entregas(seed: int = 2) -> pd.DataFrame:
    """Logística: tiempo de entrega vs distancia. Trampa: heterocedasticidad (más dispersión a mayor distancia)."""
    rng = np.random.default_rng(seed)
    n = 300
    dist = rng.gamma(3, 9, n).round(1) + 2
    peso = rng.gamma(2.5, 40, n).round(0)
    veh = rng.choice(["Moto", "Camioneta", "Camión"], n, p=[.45, .35, .2])
    exp = rng.integers(0, 15, n)
    lluvia = rng.choice([0, 1], n, p=[.75, .25])
    sigma = 0.15 + 0.035 * dist
    tiempo = 0.6 + 0.052 * dist + 0.0018 * peso - 0.03 * exp + 1.1 * lluvia + rng.normal(0, 1, n) * sigma
    return pd.DataFrame({
        "distancia_km": dist, "peso_kg": peso, "tipo_vehiculo": veh,
        "experiencia_conductor_anios": exp, "lluvia": lluvia,
        "tiempo_entrega_horas": np.clip(tiempo, 0.2, None).round(2),
    })


def energia_consumo(seed: int = 3) -> pd.DataFrame:
    """Servicios: consumo mensual de energía. Trampas: efecto U de la temperatura, tendencia y autocorrelación."""
    rng = np.random.default_rng(seed)
    n = 96
    fecha = pd.date_range("2018-01-01", periods=n, freq="MS")
    t = np.arange(n)
    temp = (20 + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)).round(1)
    ocupacion = np.clip(60 + 0.25 * t + rng.normal(0, 6, n), 30, 100).round(1)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + rng.normal(0, 600)                     # errores autocorrelacionados
    consumo = 52000 + 0.35 * (temp - 20) ** 2 * 1400 / 6 + 520 * ocupacion + e
    return pd.DataFrame({
        "fecha": fecha, "consumo_kwh": consumo.round(0), "temperatura_media_c": temp,
        "ocupacion_pct": ocupacion, "tarifa_cop_kwh": (650 + 4 * t + rng.normal(0, 15, n)).round(1),
    })


def inmobiliario_precios(seed: int = 4) -> pd.DataFrame:
    """Inmobiliario: precio de vivienda. Relación log-log con el área (sesgo a la derecha)."""
    rng = np.random.default_rng(seed)
    n = 400
    area = np.exp(rng.normal(4.3, 0.45, n)).round(1)
    estrato = rng.integers(1, 7, n)
    antig = rng.integers(0, 45, n)
    dist = rng.gamma(2, 3, n).round(1)
    hab = np.clip((area / 28 + rng.normal(0, .6, n)).round(), 1, 6)
    ln_p = 2.1 + 0.95 * np.log(area) + 0.10 * estrato - 0.008 * antig - 0.015 * dist + rng.normal(0, .16, n)
    df = pd.DataFrame({"area_m2": area, "habitaciones": hab, "antiguedad_anios": antig,
                       "distancia_centro_km": dist, "estrato": estrato, "precio_millones": np.exp(ln_p).round(1)})
    df["antiguedad_anios"] = _holes(rng, df["antiguedad_anios"], 0.05)
    return df


def manufactura_costos(seed: int = 5) -> pd.DataFrame:
    """Manufactura: costo total vs unidades. El intercepto representa el costo fijo (interpretable)."""
    rng = np.random.default_rng(seed)
    n = 150
    horas = rng.uniform(80, 400, n).round(1)
    unidades = (horas * rng.normal(3.1, 0.25, n)).round(0)
    oper = rng.integers(8, 40, n)
    turno = rng.choice(["Mañana", "Tarde", "Noche"], n)
    costo = 180 + 0.62 * unidades + 0.8 * oper + rng.normal(0, 22, n)
    df = pd.DataFrame({
        "horas_maquina": horas, "operarios": oper, "turno": turno, "unidades_producidas": unidades,
        "temperatura_planta_c": rng.normal(26, 2, n).round(1),
        "defectos_pct": np.clip(rng.normal(3, 1, n), 0, None).round(2),
        "costo_total_millones": costo.round(1),
    })
    df["precio_unitario_cop"] = 1450.0
    df["ingreso_total_millones"] = (df["unidades_producidas"] * df["precio_unitario_cop"] / 1e3).round(1)
    return df


def agro_rendimiento(seed: int = 6) -> pd.DataFrame:
    """Agro: rendimiento vs fertilizante (rendimientos decrecientes → curvatura)."""
    rng = np.random.default_rng(seed)
    n = 120
    fert = rng.uniform(0, 300, n).round(0)
    lluvia = rng.normal(1500, 280, n).round(0)
    alt = rng.integers(1200, 2000, n)
    rend = 2.0 + 5.2 * (1 - np.exp(-fert / 90)) + 0.0009 * (lluvia - 1500) - 0.0006 * (alt - 1600) + rng.normal(0, .45, n)
    return pd.DataFrame({"finca": [f"F{i:03d}" for i in range(1, n + 1)], "fertilizante_kg_ha": fert,
                         "lluvia_mm": lluvia, "altitud_msnm": alt, "rendimiento_ton_ha": rend.round(2)})


def servicio_satisfaccion(seed: int = 7) -> pd.DataFrame:
    """Servicio al cliente: casi sin relación. La app debe advertir que no hay modelo lineal confiable."""
    rng = np.random.default_rng(seed)
    n = 70
    return pd.DataFrame({
        "tiempo_espera_min": rng.gamma(3, 4, n).round(1),
        "antiguedad_cliente_meses": rng.integers(1, 120, n),
        "edad": rng.integers(18, 75, n),
        "canal": rng.choice(["Web", "Telefono", "Sucursal"], n),
        "satisfaccion_0_10": np.clip(rng.normal(7.2, 1.6, n) - 0.01 * rng.gamma(3, 4, n), 0, 10).round(1),
    })


DATASETS = {
    "Retail · ventas semanales (con fuga de información)": ("ventas_retail.csv", retail_ventas,
        "Ventas, publicidad, tráfico y descuentos. Incluye faltantes, duplicados, etiquetas sucias y una variable que filtra Y."),
    "Logística · tiempo de entrega (heterocedástico)": ("entregas_logistica.xlsx", logistica_entregas,
        "Tiempo de entrega vs distancia/peso. La dispersión crece con la distancia."),
    "Servicios · consumo de energía (serie con autocorrelación)": ("consumo_energia.txt", energia_consumo,
        "Serie mensual con tendencia; la temperatura actúa en forma de U (no lineal)."),
    "Inmobiliario · precio de vivienda (log-log)": ("precios_vivienda.csv", inmobiliario_precios,
        "Precio vs área con sesgo a la derecha; mejora con logaritmos."),
    "Manufactura · costo total (costo fijo + variable)": ("produccion_manufactura.csv", manufactura_costos,
        "Costo vs unidades producidas; el intercepto es el costo fijo. Incluye ingreso_total derivado de las unidades."),
    "Agro · rendimiento por fertilizante (curvo)": ("rendimiento_agricola.csv", agro_rendimiento,
        "Rendimientos decrecientes con el fertilizante (relación cóncava)."),
    "Servicio al cliente · satisfacción (sin relación)": ("satisfaccion_servicio.csv", servicio_satisfaccion,
        "Muestra pequeña y casi sin relación: la app debe advertir que no hay modelo confiable."),
}


CONTEXTS = {
    "Retail · ventas semanales (con fuga de información)": dict(
        sector="Comercio minorista", actividad="Cadena de tiendas de ropa en 4 ciudades",
        core="Vender más por tienda y semana optimizando el presupuesto de mercadeo",
        problema="No sabemos cuánto retorno de ventas genera cada peso invertido en publicidad.",
        objetivo_y="Ventas semanales por tienda (millones de pesos)",
        factores="Inversión en publicidad, tráfico de clientes, descuentos, temperatura",
        restricciones="El presupuesto de publicidad semanal no supera 60 millones. Los descuentos máximos son del 20%.",
        pregunta="¿Cuánto aumentan las ventas por cada millón adicional invertido en publicidad?",
        y="ventas_millones", date="semana"),
    "Logística · tiempo de entrega (heterocedástico)": dict(
        sector="Logística y transporte", actividad="Distribución de última milla en una ciudad",
        core="Cumplir promesas de entrega en tiempo y a bajo costo",
        problema="Las entregas largas generan reclamos; necesitamos prometer tiempos realistas al cliente.",
        objetivo_y="Tiempo de entrega en horas",
        factores="Distancia en km, peso de la carga, tipo de vehículo, lluvia, experiencia del conductor",
        restricciones="La distancia máxima de servicio es 120 km.",
        pregunta="¿Qué tiempo de entrega debemos prometer según la distancia?",
        y="tiempo_entrega_horas", date=None),
    "Servicios · consumo de energía (serie con autocorrelación)": dict(
        sector="Servicios públicos / energía", actividad="Administración de un edificio corporativo",
        core="Controlar el costo energético y planificar contratos de suministro",
        problema="Queremos proyectar el consumo mensual para negociar el contrato de energía.",
        objetivo_y="Consumo mensual de energía (kWh)",
        factores="Temperatura media, ocupación del edificio, tarifa",
        restricciones="La ocupación no puede superar el 100%.",
        pregunta="¿Cuánto consumo adicional implica un punto más de ocupación?",
        y="consumo_kwh", date="fecha"),
    "Inmobiliario · precio de vivienda (log-log)": dict(
        sector="Inmobiliario", actividad="Avalúos y venta de vivienda urbana",
        core="Fijar precios de venta competitivos y justificables",
        problema="Los avalúos son lentos; queremos una estimación rápida del precio de un inmueble.",
        objetivo_y="Precio de la vivienda (millones de pesos)",
        factores="Área en m2, estrato, antigüedad, distancia al centro, habitaciones",
        restricciones="Solo viviendas entre 20 y 250 m2.",
        pregunta="¿Cuánto cambia el precio por cada m2 adicional?",
        y="precio_millones", date=None),
    "Manufactura · costo total (costo fijo + variable)": dict(
        sector="Manufactura", actividad="Planta de producción por lotes",
        core="Controlar el costo unitario y fijar precios con margen",
        problema="Necesitamos separar el costo fijo del variable para presupuestar y cotizar.",
        objetivo_y="Costo total de producción (millones)",
        factores="Unidades producidas, horas de máquina, número de operarios, turno",
        restricciones="La capacidad máxima es 1.400 unidades por lote.",
        pregunta="¿Cuál es el costo fijo y cuánto cuesta cada unidad adicional?",
        y="costo_total_millones", date=None),
    "Agro · rendimiento por fertilizante (curvo)": dict(
        sector="Agroindustria", actividad="Cultivos de café en zona andina",
        core="Maximizar la producción por hectárea con uso eficiente de insumos",
        problema="El fertilizante es costoso y no sabemos a partir de qué dosis deja de rendir.",
        objetivo_y="Rendimiento (toneladas por hectárea)",
        factores="Fertilizante kg/ha, lluvia, altitud",
        restricciones="La dosis máxima autorizada es 300 kg/ha.",
        pregunta="¿Cuánto aumenta el rendimiento con más fertilizante?",
        y="rendimiento_ton_ha", date=None),
    "Servicio al cliente · satisfacción (sin relación)": dict(
        sector="Servicios al cliente", actividad="Centro de atención de una aseguradora",
        core="Mejorar la satisfacción para retener clientes",
        problema="Se cree que la espera explica la satisfacción, pero no hay evidencia.",
        objetivo_y="Satisfacción del cliente (0 a 10)",
        factores="Tiempo de espera, antigüedad del cliente, edad, canal",
        restricciones="",
        pregunta="¿La espera explica la satisfacción?",
        y="satisfaccion_0_10", date=None),
}


def write_all(folder: Path = HERE) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for _, (fname, fn, _) in DATASETS.items():
        df = fn()
        path = folder / fname
        if fname.endswith(".xlsx"):
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                df.to_excel(xw, sheet_name="Entregas", startrow=3, index=False)
                xw.sheets["Entregas"]["A1"] = "Reporte operativo de entregas (datos sintéticos)"
                pd.DataFrame({"nota": ["Datos simulados con fines demostrativos."]}).to_excel(xw, sheet_name="Notas", index=False)
        elif fname.endswith(".txt"):
            txt = df.copy()
            txt["fecha"] = txt["fecha"].dt.strftime("%d/%m/%Y")
            txt.to_csv(path, sep=";", decimal=",", index=False, encoding="cp1252")
        else:
            df.to_csv(path, index=False, encoding="utf-8")
        out.append(path)
    return out


if __name__ == "__main__":
    for p in write_all():
        print("generado:", p.name)
