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
        3. Tunggu hasil deteksi
        
        **Kamera Real-time:**
        1. Pilih mode **Kamera Real-time**
        2. Izinkan akses kamera
        3. **Langsung deteksi otomatis!**
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
    
    # Mode Upload Gambar
    if menu == "📤 Upload Gambar":
        st.header("Upload Gambar untuk Deteksi")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Pilih gambar daun kedelai...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Baca gambar
            image = Image.open(uploaded_file)
            
            # Tampilkan dalam 2 kolom
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Asli")
                st.image(image, use_container_width=True)
            
            # Tombol deteksi
            if st.button("🔍 Deteksi Penyakit", type="primary"):
                with st.spinner("Menganalisis gambar..."):
                    # Konversi ke format OpenCV
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Deteksi
                    results = model(img_bgr)
                    result_img = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("✅ Hasil Deteksi")
                        st.image(result_img_rgb, use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    st.subheader("📊 Detail Deteksi")
                    
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Buat card untuk setiap deteksi
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"**Deteksi #{i+1}**")
                            with col_b:
                                st.success(f"{class_name} ({(confidence*100):.1f}%)")
                    else:
                        st.warning("Tidak ada penyakit yang terdeteksi")
    
    # Mode Kamera Real-time LANGSUNG (seperti Flask)
    elif menu == "📷 Kamera Real-time":
        st.header("📷 Deteksi Real-time Langsung")
        st.markdown("Arahkan kamera ke daun kedelai - **deteksi otomatis berjalan**")
        
        # Informasi
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎥 Kamera Aktif")
        with col2:
            st.info("⚡ Real-time Detection")
        with col3:
            st.info("🔄 Auto update")
        
        st.markdown("---")
        
        # Tempat untuk video
        video_placeholder = st.empty()
        
        # Tombol stop
        stop_button = st.button("⏹️ Stop Kamera")
        
        # Buka kamera
        cap = cv2.VideoCapture(0)
        
        # Rotate kamera (seperti di Flask)
        # Sesuaikan dengan kebutuhan: 0, 90, 180, 270
        rotate_angle = 180  # Ganti sesuai kebutuhan
        
        # Variabel untuk FPS
        fps_display = st.empty()
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        # Loop real-time detection
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera")
                break
            
            # Rotasi frame (seperti di Flask)
            if rotate_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Deteksi YOLO langsung
            results = model(frame)
            frame_detected = results[0].plot()
            
            # Hitung FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                fps_display.metric("FPS", f"{fps:.1f}")
                frame_count = 0
                start_time = time.time()
            
            # Konversi BGR ke RGB untuk Streamlit
            frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)
            
            # Tampilkan di placeholder
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Kontrol loop
            if stop_button:
                break
        
        # Lepaskan kamera
        cap.release()
        cv2.destroyAllWindows()
        st.success("Kamera dimatikan")
    
    else:  # Mode Informasi
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
        1. **Upload Gambar** - Deteksi dari file gambar
        2. **Kamera Real-time** - **Deteksi langsung saat kamera diarahkan**
        
        **Keunggulan Real-time:**
        - Deteksi otomatis tanpa tombol
        - Streaming langsung seperti CCTV
        - Bounding box langsung muncul
        """)

else:
    st.error("❌ Gagal memuat model. Pastikan file best.pt ada di folder yang benar.")

# Footer
st.markdown("---")
st.markdown(
    "<center>© 2026 | Deteksi Penyakit Daun Kedelai | Universitas Islam Indonesia</center>",
    unsafe_allow_html=True
)
