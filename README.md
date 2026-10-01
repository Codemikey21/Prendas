# 👕 Prendas — Clasificación de imágenes con ciencia de datos

Proyecto académico de **Ciencia de Datos** que utiliza una red neuronal para clasificar imágenes de prendas de vestir, calzado y accesorios. Incluye un cuaderno de entrenamiento con **Fashion-MNIST**, un modelo guardado en formato Keras y una aplicación interactiva desarrollada con **Streamlit**.

## Información académica

| Campo | Información |
| --- | --- |
| Estudiante | Miguel Angel Solano Diaz |
| Universidad | UNIVERSIDAD AUTONOMA DE BUCARAMANGA (UNAB) |
| Materia | Ciencia de Datos |
| Proyecto | Prendas |

## Objetivo

Aplicar técnicas de ciencia de datos y aprendizaje profundo para reconocer categorías de prendas a partir de imágenes, recorriendo las etapas de exploración, preparación de datos, entrenamiento, evaluación e inferencia.

## Funcionalidades

- Cargar imágenes en formato PNG, JPG, JPEG o WEBP.
- Clasificar imágenes en diez categorías.
- Mostrar la clase predicha y las probabilidades estimadas, ordenadas de mayor a menor.
- Dibujar una prenda sobre un lienzo cuando la dependencia opcional del canvas esté disponible y sea compatible.
- Consultar el proceso de entrenamiento y las visualizaciones en un cuaderno Jupyter.

## Datos y categorías

El cuaderno carga **Fashion-MNIST** mediante `tf.keras.datasets.fashion_mnist.load_data()`. Las imágenes se normalizan dividiendo los valores de sus píxeles entre 255.

El modelo trabaja con imágenes de **28 × 28 píxeles en escala de grises** y distingue estas categorías:

| Etiqueta | Categoría |
| --- | --- |
| 0 | Camiseta |
| 1 | Pantalón |
| 2 | Jersey |
| 3 | Vestido |
| 4 | Abrigo |
| 5 | Sandalia |
| 6 | Camisa |
| 7 | Zapatilla |
| 8 | Bolso |
| 9 | Botín |

## Modelo y resultados

El cuaderno define una red neuronal densa con la siguiente arquitectura:

```text
Entrada de 28 × 28 píxeles
        ↓
Flatten
        ↓
Dense(64, ReLU)
        ↓
Dense(32, ReLU)
        ↓
Dense(16, ReLU)
        ↓
Dense(10, Softmax)
```

El entrenamiento utiliza el optimizador **Adam**, la función de pérdida `sparse_categorical_crossentropy` y **30 épocas**.

La ejecución guardada en el cuaderno registra los siguientes resultados sobre el conjunto de prueba:

| Métrica | Valor |
| --- | --- |
| Exactitud (`accuracy`) | 88,01 % |
| Pérdida (`loss`) | 0,3801 |

Estos valores corresponden a la ejecución registrada en el repositorio; pueden variar al volver a entrenar. No representan la exactitud sobre fotografías externas o dibujos del usuario.

## Tecnologías

| Herramienta | Uso |
| --- | --- |
| Python | Lenguaje de programación |
| TensorFlow y Keras | Entrenamiento y carga del modelo |
| NumPy | Manipulación de arreglos y probabilidades |
| Pillow | Procesamiento de imágenes |
| Streamlit | Interfaz web interactiva |
| Matplotlib | Visualizaciones del cuaderno |
| Jupyter / Google Colab | Ejecución del cuaderno |

## Estructura del repositorio

```text
Prendas/
├── .devcontainer/
│   └── devcontainer.json
└── Documents/
    └── prendas/
        ├── app.py
        ├── introduccion keras.ipynb
        ├── prendas.keras
        └── requirements.txt
```

- **`app.py`**: interfaz, preprocesamiento de imágenes y predicciones.
- **`introduccion keras.ipynb`**: exploración, entrenamiento, evaluación y visualización de resultados.
- **`prendas.keras`**: modelo guardado que utiliza la aplicación.
- **`requirements.txt`**: dependencias principales de la aplicación.
- **`.devcontainer/devcontainer.json`**: configuración del entorno de desarrollo en contenedor.

## Instalación y ejecución

Necesitas Git y una instalación de Python compatible con las dependencias del proyecto. El contenedor del repositorio utiliza Python 3.11.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Codemikey21/Prendas.git
cd Prendas
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activarlo en **Linux o macOS**:

```bash
source .venv/bin/activate
```

### 3. Instalar las dependencias

Desde la raíz del repositorio:

```bash
python -m pip install --upgrade pip
python -m pip install -r Documents/prendas/requirements.txt
```

Para habilitar la opción de dibujo, instalar también la dependencia opcional:

```bash
python -m pip install streamlit-drawable-canvas
```

Esta dependencia no está incluida en `requirements.txt`. Si el canvas no está disponible o presenta incompatibilidades con Streamlit, se puede utilizar la carga de imágenes.

### 4. Iniciar la aplicación

```bash
python -m streamlit run Documents/prendas/app.py
```

Abrir la dirección que indique la terminal, normalmente [http://localhost:8501](http://localhost:8501).

El archivo `prendas.keras` debe permanecer en la misma carpeta que `app.py`. No es necesario volver a entrenar para utilizar el modelo incluido.

## Cómo usar la aplicación

1. Subir una imagen o dibujar una prenda en el lienzo disponible.
2. Pulsar **Predecir con la imagen subida** o **Predecir con el dibujo**, según corresponda.
3. Consultar la categoría predicha y las probabilidades de las diez clases.

La aplicación convierte la imagen a escala de grises, ajusta el contraste, invierte sus tonos si detecta una imagen mayormente clara, la redimensiona a 28 × 28 píxeles y normaliza sus valores.

Para obtener entradas similares a las del entrenamiento, usar una sola prenda centrada, con fondo oscuro y tonos claros. Las fotografías con fondos complejos y los dibujos muy distintos a Fashion-MNIST pueden producir predicciones incorrectas. El clasificador siempre elige entre las diez categorías disponibles.

## Explorar o volver a entrenar el modelo

Abrir `Documents/prendas/introduccion keras.ipynb` en Google Colab o en un entorno Jupyter.

Para utilizar Jupyter localmente, instalar sus herramientas y Matplotlib además de las dependencias principales:

```bash
python -m pip install jupyter matplotlib
python -m jupyter notebook "Documents/prendas/introduccion keras.ipynb"
```

El cuaderno contiene celdas específicas de Google Colab para montar Google Drive y guardar el modelo. Al ejecutarlo localmente, omitir las celdas de `google.colab` y adaptar la ruta de guardado. Por ejemplo, si el directorio de trabajo es `Documents/prendas`:

```python
model.save("prendas.keras")
```

La celda `!nvidia-smi` sirve para consultar una GPU NVIDIA en el entorno original y puede omitirse si no está disponible. La primera carga de Fashion-MNIST requiere conexión a Internet para descargar los datos.

## Autor

**Miguel Angel Solano Diaz**  
Estudiante — Ciencia de Datos  
**UNIVERSIDAD AUTONOMA DE BUCARAMANGA (UNAB)**

[Repositorio del proyecto](https://github.com/Codemikey21/Prendas)
