# Inspección de grietas

Aplicación Streamlit para visualizar bordes, estimar la inclinación de segmentos y clasificar imágenes de concreto con una CNN.

## Archivos requeridos

- `app.py`: aplicación.
- `modelo_base_grietas.keras`: modelo entrenado.
- `requirements.txt`: dependencias instaladas por Streamlit Community Cloud.

No subas la carpeta `.venv`; las dependencias se instalan desde `requirements.txt`.

## Ejecutar localmente

Usa Python 3.11 y ejecuta desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Sube `app.py`, `modelo_base_grietas.keras`, `requirements.txt` y `README.md` a un repositorio de GitHub.
2. En Streamlit Community Cloud, elige **Create app** y conecta tu cuenta de GitHub.
3. Selecciona el repositorio y la rama, y establece `app.py` como archivo principal.
4. En las opciones avanzadas, selecciona Python 3.11 si está disponible y despliega la app.

La primera instalación puede tardar porque TensorFlow es una dependencia grande.

## Alcance

El entrenamiento del notebook usa una sola época. La predicción y el nivel de riesgo son resultados preliminares y no sustituyen una inspección estructural profesional.