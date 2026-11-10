# Manual de despliegue paso a paso

> **Resumen en 30 segundos.** Esta app está hecha con **Streamlit**, y **Vercel no puede ejecutarla** (ver el apartado 1).
> La opción gratuita y más sencilla es **Streamlit Community Cloud**, que publica directamente desde su repositorio de GitHub.
> Son 4 grandes pasos: (1) probar en su computador, (2) subir el código a GitHub, (3) crear la app en `share.streamlit.io`, (4) verificarla.

---

## 1. ¿Por qué no se puede en Vercel?

Vercel ejecuta el código como **funciones sin servidor** (se encienden por unos segundos, responden y se apagan). Streamlit necesita
lo contrario: **un servidor encendido todo el tiempo** que mantenga una conexión **WebSocket** abierta con el navegador. Vercel no
admite WebSockets en sus funciones, y el personal del foro de Streamlit indica que no parece posible desplegar Streamlit allí.

**Qué sí se puede hacer si quiere usar Vercel de todas formas:** una arquitectura dividida: la interfaz web en **Vercel** (React/Next.js) y el
cálculo estadístico en una **API de Python** alojada en otro servicio (Render, Railway…). Es el patrón de su otro proyecto (el zip con
`frontend/` y `backend/`). Implicaría **reescribir la interfaz de los 13 módulos en React**; el motor de `core/` sí se reutiliza detrás de la API.
Es un proyecto aparte, no un ajuste. Si lo quiere, pídalo como tarea nueva.

**Una opción mucho más simple:** publicar en Vercel una página que *muestre* la app alojada en Streamlit (con su dirección `.vercel.app` o su dominio).
La página ya está lista en la carpeta `vercel/` y se explica paso a paso en **`MANUAL_VERCEL.html`** (ábralo con doble clic). El cálculo sigue corriendo en Streamlit.
Condiciones: la app de Streamlit debe ser pública y el plan gratuito de Vercel es solo para uso personal no comercial.

---

## 2. Antes de empezar: qué necesita

| Requisito | Detalle |
|---|---|
| Cuenta de GitHub | La que ya usa: `velandiaivan640-droid`. Debe ser **administrador** del repositorio (lo es: es el dueño). |
| Repositorio | `app-regresion-multiple`. Por lo que pude comprobar, hoy es **privado** (GitHub responde "no encontrado" a una consulta sin sesión). Sirve privado o público. |
| Cuenta de Streamlit | Se crea sola al entrar a `share.streamlit.io` con GitHub. Gratis. |
| Python 3.12 | Ya lo tiene instalado en su computador. Es la versión que se usará en la nube. |
| Git | Ya lo usa (GitHub Desktop o la línea de comandos). |

> **Privacidad.** Streamlit Community Cloud aloja las apps en Estados Unidos. Todo archivo que alguien cargue en la app pública
> pasa por ese servidor. **No suba datos confidenciales** (por ejemplo, los del estudio de Manizales ni cartera real de un banco) a una app
> pública. Las bases del repositorio son sintéticas.

---

## 3. Paso 1 · Probar en su computador (5 minutos)

Abra **PowerShell** y ejecute, línea por línea:

```powershell
cd C:\Users\DELL\Desktop\app_regresion
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Se abrirá `http://localhost:8501`. Compruebe: (a) en el módulo 1 (*Carga de datos*) abra «¿No tiene datos a la mano?» y cargue **«Banca · monto de crédito aprobado (1000 registros)»**,
(b) recorra los módulos hasta el 8. Para cerrar, `Ctrl + C` en la terminal.

*(Opcional)* correr las pruebas automáticas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

> Si Windows dice que "la ejecución de scripts está deshabilitada", no hay problema: estos comandos llaman directamente a
> `.venv\Scripts\python.exe` y **no** usan `Activate.ps1`.

---

## 4. Paso 2 · Subir el código a GitHub

La carpeta `C:\Users\DELL\Desktop\app_regresion` ya está conectada a su repositorio. En PowerShell:

```powershell
cd C:\Users\DELL\Desktop\app_regresion
git status
```

Verá archivos nuevos y modificados (la interfaz, las bases de prueba, `requirements.txt`, etc.). **Aviso de saltos de línea:** mensajes del tipo
*"LF will be replaced by CRLF"* son normales en Windows; ignórelos.

**4.1. Quite del seguimiento los archivos compilados (`.pyc`).** El primer commit incluyó por error `core/__pycache__`. El nuevo `.gitignore`
evita que vuelvan a subir, pero los ya subidos hay que retirarlos del repositorio (no se borran de su disco):

```powershell
git rm -r --cached core/__pycache__
```

*(Si responde `pathspec ... did not match any files`, ya no estaban; continúe.)*

**4.2. Revise qué va a subir.** No debe aparecer ningún archivo del estudio de Manizales, claves, ni `.env`:

```powershell
git add .
git status
```

**4.3. Confirme y suba:**

```powershell
git commit -m "App de regresion lineal multiple: interfaz Streamlit, bases de prueba y despliegue"
git push origin main
```

La primera vez, Git puede abrir una ventana para iniciar sesión en GitHub. Al terminar, entre a
`https://github.com/velandiaivan640-droid/app-regresion-multiple` y confirme que ve `app.py`, `core/`, `ui/`, `requirements.txt`.

> **Para actualizar la app ya publicada** (`regresion-multiple.streamlit.app`) no hay que crear nada nuevo: basta con subir los cambios a la rama `main` de GitHub (sección 7) y Streamlit Community Cloud la reconstruye sola en 1–3 minutos.

---

## 5. Paso 3 · Crear la app en Streamlit Community Cloud

1. Entre a **https://share.streamlit.io** y pulse **Continue with GitHub**. Autorice el acceso.
2. **Repositorio privado:** si su repositorio es privado, Streamlit le pedirá un permiso adicional de GitHub (alcance `repo`). Acéptelo.
   Es necesario para poder leer el código.
3. Arriba a la derecha pulse **Create app**.
4. Cuando pregunte si ya tiene una app, elija **Yup, I have an app**.
5. Complete el formulario:

   | Campo | Qué escribir |
   |---|---|
   | **Repository** | `velandiaivan640-droid/app-regresion-multiple` (o el nuevo nombre si lo renombró) |
   | **Branch** | `main` |
   | **Main file path** | `app.py` |
   | **App URL** | un nombre libre, p. ej. `regresion-multiple` → quedará `https://regresion-multiple.streamlit.app` |

6. Pulse **Advanced settings** y en **Python version** elija **3.12**. Deje *Secrets* vacío (la app no usa claves). Pulse **Save**.

   > ⚠ **La versión de Python no se puede cambiar después.** Para cambiarla hay que borrar la app y volver a desplegarla. La documentación
   > de Streamlit indica 3.12 como valor por defecto, pero algunos usuarios reportan que reciben 3.14, para la que muchos paquetes aún no
   > están listos. **Elíjala siempre de forma explícita.**

7. Pulse **Deploy**. Verá una pantalla de arranque; instalar las dependencias tarda **unos minutos**. A la derecha hay un panel de
   **logs** (solo lo ven quienes tienen permiso de escritura en el repositorio).
8. Cuando cargue la pantalla de inicio de la app (con el menú de 13 módulos), el despliegue terminó. Copie la URL.

---

## 6. Paso 4 · Verificar que todo funciona

Haga esta lista en la URL pública:

- [ ] Módulo 1 → «¿No tiene datos a la mano?» → **«Salud · costo de hospitalización (1000 registros)»** → *Cargar ejemplo*: aparecen **1.000 registros**, 10 duplicados y las columnas con su tipo.
- [ ] Módulo 2 (*Qué dicen sus datos*): explica de qué trata la tabla y sugiere `costo_total_millones` como variable a estimar.
- [ ] Módulos 3 a 8 abren sin error. En el 7 todos los modelos tienen **2 o más variables**; en el 8 debe ver el veredicto, el VIF y los gráficos de diagnóstico.
- [ ] Módulo 10: mueva los deslizadores (uno por variable); al salir del rango observado aparece la alerta de **extrapolación**.
- [ ] Módulo 12: descargue el informe HTML.
- [ ] Módulo 1 → cargue un CSV propio (por ejemplo `sample_data/ventas_restaurante.csv`): sin escribir nada, la app debe sugerir `Bowls` como Y.

---

## 7. Actualizar la app después

Cada vez que haga cambios:

```powershell
cd C:\Users\DELL\Desktop\app_regresion
git add .
git commit -m "Describa el cambio"
git push origin main
```

Streamlit detecta el cambio y actualiza la app. Si cambió `requirements.txt`, la reinstalación puede tardar unos minutos.
Streamlit limita las actualizaciones desde GitHub a **5 por minuto**. En el panel de la app (**Manage app**) puede ver los logs y reiniciarla (**Reboot**).

---

## 8. Quién puede ver la app

**No pude verificar** si una app nueva queda pública o privada por defecto, ni las opciones de visibilidad, la memoria disponible o el tiempo
de inactividad antes de que la app "se duerma" en el plan gratuito. **Antes de compartir el enlace, abra la URL en una ventana de incógnito**
para comprobar quién puede verla, y revise las opciones en el menú **Share** del panel de la app. Una app dormida suele reactivarse al abrir el
enlace, con una pequeña espera.

---

## 9. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Error installing requirements` / falla al instalar | Se eligió Python 3.14 (o muy antiguo) | Borrar la app y **desplegar de nuevo eligiendo 3.12** (paso 5.6). |
| `ModuleNotFoundError: No module named 'xxx'` | Falta el paquete en `requirements.txt` | Agréguelo, `git push`, y espere el reinstalado. |
| La app se queda mucho tiempo en *"in the oven"* | Primera instalación o servidor ocupado | Espere unos minutos; si persiste, **Manage app → Reboot**. |
| `This repository does not exist` / no aparece mi repo | Streamlit no tiene permiso sobre el repo privado | En GitHub → *Settings → Applications* otorgue el acceso `repo` a Streamlit y reintente. |
| Cambié `requirements.txt` y no se nota | Tarda en reinstalar | Espere; mire los logs; si no, **Reboot**. |
| La app funciona local pero no en la nube | Versión de Python o de paquetes distinta | Use `py -3.12` y las versiones exactas de `requirements.txt` (paso 3). |
| Al cargar un `.xls` antiguo falla | Falta `xlrd` | Ya está en `requirements.txt`; verifique que se desplegó esa versión. |
| Un archivo muy grande se cae | Memoria limitada del plan gratuito | Use una muestra del archivo o aloje la app en un servicio con más memoria. |

---

## 10. Alternativas si Streamlit Community Cloud no le sirve

| Opción | Cuándo conviene |
|---|---|
| **Render** (servicio web con Docker) o **Railway** | Más memoria/control, o necesita un dominio propio. De pago o con plan gratuito limitado. |
| **Hugging Face Spaces** (SDK Streamlit) | Alternativa gratuita similar; se sube el mismo código. |
| **Máquina propia o servidor de la empresa** | Datos confidenciales: la app corre en su red y nada sale. `streamlit run app.py --server.port 8501` |
| **Vercel + API aparte** | Solo si quiere una interfaz web hecha a la medida (proyecto nuevo; ver apartado 1). |

---

### Fuentes consultadas

- Streamlit Docs: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [Deploy!](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [App dependencies](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies), [Upgrade Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python) y [Status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
- Vercel y WebSockets (foros y artículos de terceros; no pude leer la documentación oficial de Vercel directamente): [Vercel Community: WebSockets con Fluid Compute](https://community.vercel.com/t/does-vercel-support-websockets-now-that-we-have-fluid-compute/27205), [Foro de Streamlit: Streamlit en Vercel](https://discuss.streamlit.io/t/deploy-streamlit-in-vercel/38070).
- Versión de Python por defecto reportada por usuarios: [Foro de Streamlit](https://discuss.streamlit.io/t/all-my-streamlit-community-cloud-apps-fail-to-deploy-after-rebuild-python-3-14-requirements-installation-error/122130).
