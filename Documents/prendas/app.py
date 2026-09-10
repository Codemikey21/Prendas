from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None


st.set_page_config(
    page_title="Predictor de prendas con Tensorflow",
    page_icon="👕",
    layout="centered",
)

st.title("Predictor de prendas con Tensorflow")

MODEL_PATH = Path(__file__).resolve().parent / "prendas.keras"
CLASS_NAMES = [
    "Camiseta",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en {MODEL_PATH}. "
            "Asegúrate de que prendas.keras esté en la misma carpeta que app.py."
        )
    return keras.models.load_model(MODEL_PATH)


model = load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convierte una imagen a 28x28, escala de grises y normalizada como en el entrenamiento."""
    grayscale_image = ImageOps.grayscale(image)
    grayscale_image = ImageOps.autocontrast(grayscale_image)

    # Si la imagen es mayormente clara, se asume fondo blanco y se invierte para dejar
    # fondo negro y objeto con tonos más claros, como en el conjunto de entrenamiento.
    array = np.array(grayscale_image, dtype=np.float32)
    if array.mean() > 127:
        array = 255.0 - array

    resized_image = Image.fromarray(array.astype(np.uint8)).resize(
        (28, 28), Image.Resampling.LANCZOS
    )
    normalized = np.array(resized_image, dtype=np.float32) / 255.0
    normalized = normalized.reshape(1, 28, 28, 1)
    return normalized


def predict_from_image(image: Image.Image) -> tuple[str, np.ndarray]:
    prepared = preprocess_image(image)
    probabilities = model.predict(prepared, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    return predicted_label, probabilities


st.caption(
    "Usa el canvas para dibujar una prenda sobre fondo negro o sube una imagen similar a las que se usaron en entrenamiento."
)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("1) Dibuja una prenda")
    canvas_result = None

    if st_canvas is not None:
        try:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=12,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=280,
                height=280,
                drawing_mode="freedraw",
                display_toolbar=True,
                key="draw_canvas",
            )
        except Exception:
            st.warning(
                "El canvas no está disponible en esta versión de Streamlit. Puedes usar la opción de subir una imagen."
            )
    else:
        st.info(
            "El canvas no está disponible en esta distribución. Sube una imagen para hacer la predicción."
        )

    draw_button = st.button("Predecir con el dibujo", key="predict_draw") if st_canvas is not None else False

with right_col:
    st.subheader("2) Sube una imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    upload_button = st.button("Predecir con la imagen subida", key="predict_upload")


prediction_placeholder = st.empty()

if draw_button:
    if canvas_result is None or canvas_result.image_data is None:
        prediction_placeholder.warning("Primero dibuja algo en el canvas.")
    else:
        image_array = np.array(canvas_result.image_data)
        if image_array.size == 0:
            prediction_placeholder.warning("No se detectó contenido en el canvas.")
        else:
            if image_array.dtype.kind == "f":
                image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
            rgb_image = image_array[:, :, :3]
            drawing_image = Image.fromarray(rgb_image, mode="RGB")
            label, probabilities = predict_from_image(drawing_image)
            prediction_placeholder.success(f"Predicción del dibujo: {label}")
            st.write("Probabilidades estimadas:")
            for idx in np.argsort(probabilities)[::-1]:
                st.write(f"- {CLASS_NAMES[idx]}: {probabilities[idx] * 100:.1f}%")

if upload_button:
    if uploaded_file is None:
        prediction_placeholder.warning("Primero sube una imagen.")
    else:
        uploaded_image = Image.open(uploaded_file)
        label, probabilities = predict_from_image(uploaded_image)
        prediction_placeholder.success(f"Predicción de la imagen subida: {label}")
        st.write("Probabilidades estimadas:")
        for idx in np.argsort(probabilities)[::-1]:
            st.write(f"- {CLASS_NAMES[idx]}: {probabilities[idx] * 100:.1f}%")


st.markdown("---")

st.subheader("Instrucciones")
st.markdown(
    """
    - Dibuja sobre fondo negro con trazos blancos o grises claros para que se parezca a las imágenes usadas en entrenamiento.
    - Si subes una imagen, procura que sea similar o parecida a las páginas/clothes con las que se entrenó el modelo.
    - El modelo recibe imágenes en escala de grises, normalizadas dividiendo entre 255 y reducidas a 28x28 píxeles.
    - Las clases previstas son: Camiseta, Pantalón, Jersey, Vestido, Abrigo, Sandalia, Camisa, Zapatilla, Bolso y Botín.
    - El modelo utiliza una capa de salida Softmax para devolver la probabilidad de cada clase.
    """
)

st.caption(
    "Recomendación: usa imágenes con fondo oscuro, prendas centradas y tonos claros para lograr mejores resultados."
)
