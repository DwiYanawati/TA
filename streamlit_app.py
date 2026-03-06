import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Daun Kedelai",
    page_icon="🌿",
    layout="wide"
)

# Title
st.title("🌿 Deteksi Penyakit Daun Kedelai")
st.markdown("Aplikasi deteksi penyakit pada daun kedelai menggunakan **YOLOv9**")

# Sidebar
with st.sidebar:
    st.header("Menu")
    menu = st.radio("Pilih Mode:", ["📤 Upload Gambar", "📷 Kamera Real-time", "ℹ️ Informasi"])
    
    st.markdown("---")
    st.subheader("Cara Penggunaan")
    st.info(
        """
        **Upload Gambar:**
        1. Pilih mode **Upload Gambar**
        2. Upload foto daun kedelai
        3. Klik tombol deteksi
        
        **Kamera Real-time:**
        1. Pilih mode **Kamera Real-time**
        2. Izinkan akses kamera
        3. Deteksi otomatis berjalan
        """
    )

# Load model dengan caching
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is not None:
    st.sidebar.success("✅ Model siap digunakan!")
    
    # ============= MODE UPLOAD GAMBAR =============
    if menu == "📤 Upload Gambar":
        st.header("📤 Upload Gambar untuk Deteksi")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar daun kedelai...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Asli")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Deteksi Penyakit", type="primary"):
                with st.spinner("Menganalisis gambar..."):
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    results = model(img_bgr)
                    result_img = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("✅ Hasil Deteksi")
                        st.image(result_img_rgb, use_container_width=True)
                    
                    st.subheader("📊 Detail Deteksi")
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            confidence = float(box.conf[0])
                            st.success(f"{i+1}. {class_name} ({(confidence*100):.1f}%)")
                    else:
                        st.warning("Tidak ada penyakit yang terdeteksi")
    
    # ============= MODE KAMERA REAL-TIME =============
    elif menu == "📷 Kamera Real-time":
        st.header("📷 Deteksi Real-time dengan Kamera")
        st.markdown("Arahkan kamera ke daun kedelai - **deteksi otomatis berjalan**")
        
        # Info bar
        col1, col2 = st.columns(2)
        with col1:
            st.info("🎥 Kamera Aktif")
        with col2:
            st.info("⚡ Real-time Detection")
        
        st.markdown("---")
        
        # Tempat video
        video_placeholder = st.empty()
        
        # Tombol stop
        col1, col2, col3 = st.columns(3)
        with col2:
            stop_button = st.button("⏹️ STOP KAMERA", type="primary")
        
        # Buka kamera
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Tidak dapat mengakses kamera")
            else:
                # Loop deteksi
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Gagal membaca frame")
                        break
                    
                    # Rotasi kamera (sesuaikan)
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
                    # Deteksi YOLO
                    results = model(frame)
                    frame_detected = results[0].plot()
                    
                    # Konversi BGR ke RGB
                    frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)
                    
                    # Tampilkan
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Lepaskan kamera (TANPA destroyAllWindows)
                cap.release()
                st.success("✅ Kamera dimatikan")
                
        except Exception as e:
            st.error(f"Error kamera: {e}")
    
    # ============= MODE INFORMASI =============
    else:
        st.header("ℹ️ Informasi Sistem")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "YOLOv9", "Object Detection")
        with col2:
            st.metric("Framework", "Streamlit", "Web App")
        with col3:
            st.metric("Fitur", "Upload & Kamera", "Real-time")
        
        st.markdown("---")
        st.subheader("Tentang Aplikasi")
        st.write("""
        Aplikasi ini menggunakan **YOLOv9** untuk mendeteksi penyakit pada daun kedelai.
        
        **Fitur:**
        - 📤 Upload gambar untuk deteksi
        - 📷 Kamera real-time langsung
        - ℹ️ Informasi sistem
        
        **Dikembangkan untuk:** Skripsi/Tugas Akhir
        """)

else:
    st.error("❌ Gagal memuat model. Pastikan file best.pt ada di folder yang benar.")

# Footer
st.markdown("---")
st.markdown(
    "<center>© 2026 | Deteksi Penyakit Daun Kedelai | Universitas Islam Indonesia</center>",
    unsafe_allow_html=True
)
