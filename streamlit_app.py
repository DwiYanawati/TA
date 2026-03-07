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
    
        # ============= MODE KAMERA REAL-TIME (WEBCAM USER) =============
    elif menu == "📷 Kamera Real-time":
        st.header("📷 Deteksi Real-time dengan Kamera")
        st.markdown("Arahkan kamera ke daun kedelai - **deteksi otomatis berjalan**")
        
        # Import library webrtc
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        import av
        
        # Info bar
        col1, col2 = st.columns(2)
        with col1:
            st.info("🎥 Kamera User")
        with col2:
            st.info("⚡ Real-time Detection")
        
        st.markdown("---")
        
        # Class untuk memproses video
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.threshold = 0.5
            
            def recv(self, frame):
                # Konversi frame ke format yang bisa diproses
                img = frame.to_ndarray(format="bgr24")
                
                # Deteksi dengan YOLO
                if self.model is not None:
                    results = self.model(img)
                    img = results[0].plot()
                
                # Kembalikan frame yang sudah diproses
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Konfigurasi WebRTC
        rtc_config = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
        
        # Tampilkan streamer
        ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.model = model
        
        st.markdown("---")
        st.caption("💡 Klik 'START' untuk mengaktifkan kamera. Izinkan akses kamera di browser Anda.")
        
    # ============= MODE INFORMASI =============
    else:  # menu == "ℹ️ Informasi"
        st.header("ℹ️ Informasi Sistem")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "YOLOv9", "Object Detection")
        with col2:
            st.metric("Framework", "Streamlit", "Web App")
        with col3:
            st.metric("Fitur", "Upload & Kamera", "Real-time")
        
        st.markdown("---")
        
        # TABS untuk Informasi
        tab1, tab2, tab3 = st.tabs(["📖 Tentang", "🎯 Cara Kerja", "📊 Training"])
        
        with tab1:
            st.subheader("📖 Tentang Aplikasi")
            st.write("""
            Aplikasi ini menggunakan **YOLOv9** untuk mendeteksi penyakit pada daun kedelai.
            
            **Model dilatih dengan:**
            - Dataset penyakit daun kedelai
            - Arsitektur YOLOv9
            - Framework Ultralytics
            """)
        
        with tab2:
            st.subheader("🎯 Cara Kerja")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📤 Upload Gambar**")
                st.write("""
                1. Upload gambar daun kedelai
                2. Sistem memproses gambar 
                3. Hasil deteksi ditampilkan dengan bounding box
                4. Detail deteksi (nama penyakit, confidence) ditampilkan
                """)
            
            with col2:
                st.markdown("**📷 Kamera Real-time**")
                st.write("""
                1. Akses kamera diizinkan
                2. Frame diproses langsung (real-time)
                3. Bounding box muncul otomatis di video
                """)
        
        with tab3:
            st.subheader("📊 Model Training")
            st.write("""
            **Dataset:** Penyakit Daun Kedelai (5 kelas)
            **Arsitektur:** YOLOv9
            **Framework:** Ultralytics
            """)
        
        st.markdown("---") 

# Footer
st.markdown("---")
st.markdown(
    "<center>© 2026 | Deteksi Penyakit Daun Kedelai | Universitas Islam Indonesia</center>",
    unsafe_allow_html=True
)



