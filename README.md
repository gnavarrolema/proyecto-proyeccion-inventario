# Proyección de Inventario

Este proyecto es una aplicación web diseñada para realizar pronósticos de demanda de inventario utilizando una variedad de modelos de machine learning. La aplicación permite a los usuarios subir sus datos, seleccionar un modelo de pronóstico, y obtener proyecciones junto con cálculos de stock de seguridad para optimizar los niveles de inventario.

## Características

- **API de Pronóstico:** Expone endpoints para realizar predicciones de demanda.
- **Múltiples Modelos:** Soporta varios modelos de pronóstico:
  - SARIMA
  - LSTM
  - GRU
  - XGBoost
- **Optimización de Hiperparámetros:** Incluye la capacidad de ajustar los hiperparámetros de los modelos para mejorar la precisión.
- **Cálculo de Stock de Seguridad:** Ayuda a determinar los niveles óptimos de inventario para evitar roturas de stock.
- **Interfaz Web:** Una interfaz de usuario sencilla para interactuar con la API y visualizar los resultados.

## Tecnologías Utilizadas

### Backend
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- XGBoost
- TensorFlow/Keras

### Frontend
- JavaScript
- Tailwind CSS
- PostCSS

## Prerrequisitos

Asegúrate de tener instalados los siguientes programas:
- Python 3.8+
- Node.js y npm

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd proyecto_proyeccion_inventario
    ```

2.  **Configura el entorno de Python:**
    - Crea y activa un entorno virtual:
      ```bash
      python -m venv env
      source env/bin/activate  # En Windows usa `env\Scripts\activate`
      ```
    - Instala las dependencias de Python:
      ```bash
      pip install -r requirements.txt
      ```

3.  **Configura el entorno de Node.js:**
    - Instala las dependencias de Node.js:
      ```bash
      npm install
      ```
    - Compila los assets del frontend:
      ```bash
      npm run build
      ```

## Cómo Ejecutar la Aplicación

Una vez completada la instalación, puedes iniciar la aplicación Flask:

```bash
python run.py
```

Una vez que la aplicación se esté ejecutando, abre tu navegador web y ve a `http://127.0.0.1:5000` para acceder a la interfaz de usuario.

## Carga de Datos (CSV o Excel)

La aplicación soporta archivos de datos en formato `.csv`, `.xlsx` y `.xls`:
- Para CSV: use separador `;`, decimal `,` y miles `.` si su archivo sigue formato regional ES/LA.
- Para Excel: basta con subir el archivo; se detecta y carga automáticamente.

Las columnas esperadas incluyen al menos: `Mes&Año`, `ARTÍCULO`, `CANTIDADES`, `PRODUCCIÓN`.

## Estructura del Proyecto

```
/
├── app/                    # Directorio principal de la aplicación Flask
│   ├── api/                # Módulo de la API (rutas y lógica)
│   ├── models/             # Modelos de Machine Learning y optimizadores
│   ├── static/             # Archivos estáticos (CSS, JS, imágenes)
│   ├── templates/          # Plantillas HTML
│   └── utils/              # Utilidades (carga de datos, métricas, etc.)
├── data/                   # Directorio para los datos de entrada
├── logs/                   # Directorio para archivos de log
├── results/                # Directorio para guardar resultados
├── tests/                  # Pruebas unitarias
├── run.py                  # Punto de entrada para ejecutar la aplicación
├── requirements.txt        # Dependencias de Python
└── package.json            # Dependencias de Node.js
```