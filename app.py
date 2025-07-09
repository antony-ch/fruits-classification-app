import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import io

# --- Konfigurasi Halaman & Gaya CSS ---
st.set_page_config(page_title="Fruits Classification App")

def load_css():
    """Memuat CSS kustom untuk tampilan yang modern dan elegan."""
    st.markdown("""
    <style>
    /* Mengimpor font Poppins dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Variabel Warna untuk kemudahan modifikasi */
    :root {
        --primary-bg-color: #f0f2f6; /* Latar belakang abu-abu lembut */
        --secondary-bg-color: #ffffff; /* Latar belakang kartu/konten */
        --primary-text-color: #262730; /* Teks utama (gelap) */
        --secondary-text-color: #5a5a5a; /* Teks sekunder (abu-abu) */
        --accent-color: #4A90E2; /* Warna aksen biru */
        --accent-gradient: linear-gradient(to right, #4A90E2, #82c6ff);
        --border-color: #dfe6e9;
    }

    /* Terapkan font dan warna latar belakang utama ke seluruh aplikasi */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: var(--primary-bg-color);
    }

    /* Kontainer utama untuk memusatkan konten */
    .main .block-container {
        max-width: 900px; /* Sedikit diperlebar untuk layout 2 kolom */
        padding: 2rem 1.5rem;
        margin: auto;
    }

    /* Gaya untuk judul utama */
    h1 {
        font-weight: 700;
        color: var(--primary-text-color);
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Gaya untuk sub-judul/deskripsi */
    .app-description {
        text-align: center;
        color: var(--secondary-text-color);
        margin-bottom: 2.5rem;
        font-size: 1.1rem;
    }

    /* Gaya untuk area upload file */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        background-color: var(--secondary-bg-color);
        border-radius: 15px;
        padding: 25px;
        transition: all 0.3s ease-in-out;
    }

    .stFileUploader:hover {
        border-color: var(--accent-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .stFileUploader label {
        font-weight: 600;
        color: var(--primary-text-color);
    }

    /* Membuat gambar yang ditampilkan memiliki sudut membulat dan bayangan */
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease-in-out;
    }
    .stImage img:hover {
        transform: scale(1.03);
    }

    /* Kontainer untuk hasil prediksi (dibuat seperti kartu) */
    .result-container {
        background-color: var(--secondary-bg-color);
        border-radius: 15px;
        padding: 25px 30px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        height: 100%; /* Membuat tinggi container hasil sama dengan gambar */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-container h3 {
        font-weight: 600;
        color: var(--primary-text-color);
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color);
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.1rem;
        color: var(--secondary-text-color);
        margin-bottom: 0.5rem;
    }
    .result-text strong {
        font-weight: 600;
        color: var(--primary-text-color);
        font-size: 1.2rem;
    }

    /* Progress Bar Kustom */
    .progress-bar-container {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .progress-bar-fill {
        height: 100%;
        background-image: var(--accent-gradient);
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease-in-out;
    }
    .confidence-text {
        text-align: right;
        font-weight: 600;
        color: var(--accent-color);
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        color: var(--secondary-text-color);
        font-size: 0.9em;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Logika Aplikasi ---

# Daftar nama kelas
CLASS_NAMES = [
    'Apple', 'Avocado', 'Banana', 'Carrot', 'Dragonfruit', 'Durian', 'Grapes', 'Guanabana',
    'Guava', 'Lychee', 'Mango', 'Melon', 'Orange', 'Papaya', 'Pineapple',
    'Rambutan', 'Salak', 'Strawberry', 'Tomato', 'Watermelon'
]

@st.cache_resource
def load_fruit_model() -> tf.keras.Model:
    """Memuat model klasifikasi buah dari file .h5."""
    try:
        model = tf.keras.models.load_model("fruits_classification_streamlit.h5", compile=False)
        return model
    except FileNotFoundError:
        st.error("File model 'fruits_classification_streamlit.h5' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip ini.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Melakukan prapemrosesan gambar agar sesuai dengan input model."""
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(model: tf.keras.Model, image: Image.Image) -> tuple[str, float]:
    """Melakukan prediksi pada gambar yang diberikan."""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# --- Antarmuka Pengguna (UI) ---

load_css()
model = load_fruit_model()

st.title("Fruits Classification App")
st.markdown("<p class='app-description'>Unggah gambar buah dan model akan memprediksi jenis buah tersebut.</p>", unsafe_allow_html=True)

if model is not None:
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Membuat dua kolom dengan perbandingan lebar 1:1
        col1, col2 = st.columns(2, gap="medium")

        # Kolom 1: Untuk menampilkan gambar yang diunggah
        with col1:
            st.image(image, caption='Gambar yang diupload', use_container_width=True)

        # Kolom 2: Untuk menampilkan hasil prediksi
        with col2:
            # Lakukan prediksi
            with st.spinner('Menganalisis gambar...'):
                predicted_class, confidence = predict(model, image)

            # Menampilkan hasil dalam satu blok HTML untuk styling yang konsisten
            result_html = f"""
            <div class="result-container">
                <h3>Hasil Prediksi</h3>
                <p class="result-text">Prediksi Buah: <strong>{predicted_class}</strong></p>
                <p class="result-text">Tingkat Kepercayaan:</p>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {confidence:.2f}%;"></div>
                </div>
                <p class="confidence-text">{confidence:.2f}%</p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

    else:
        # Tampilan saat belum ada file yang diunggah
        st.info("Silakan unggah gambar untuk memulai proses klasifikasi.")

# Footer
st.markdown(
    "<div class='footer'>Copyright © 2025. Fruits Classification App</div>",
    unsafe_allow_html=True
)