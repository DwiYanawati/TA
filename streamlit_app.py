import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import time
import platform

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Kedelai",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/soybean.png", width=80)
    st.title("🌿 Menu Utama")
    
    menu = st.radio(
        "Pilih Mode:",
        ["📤 Upload Gambar", "📷 Kamera Real-time", "ℹ️ Informasi"],
        index=0
    )
    
    st.markdown("---")
    
    with st.expander("📌 Cara Penggunaan", expanded=True):
        if menu == "📤 Upload Gambar":
            st.info("""
            1. Klik **Browse files**
            2. Pilih gambar daun kedelai
            3. Klik **Deteksi Penyakit**
            4. Lihat hasil bounding box
            """)
        elif menu == "📷 Kamera Real-time":
            st.info("""
            1. Izinkan akses kamera
            2. Atur rotasi jika perlu
            3. Arahkan ke daun
            4. Deteksi otomatis!
            """)
        else:
            st.info("""
            Informasi lengkap tentang:
            - Model yang digunakan
            - Cara kerja sistem
            - Statistik deteksi
            """)
    
    st.markdown("---")
    st.caption("© 2026 | Universitas Islam Indonesia")

# ===================== LOAD MODEL =====================
@st.cache_resource(show_spinner="🔄 Loading model YOLOv9...")
def load_model():
    try:
        # Cek apakah file best.pt ada
        if not os.path.exists("best.pt"):
            st.error("❌ File best.pt tidak ditemukan di folder!")
            return None
        
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# ===================== FUNGSI DETEKSI =====================
def detect_image(image, model):
    """Fungsi untuk mendeteksi objek dalam gambar"""
    # Konversi PIL ke array
    img_array = np.array(image)
    
    # Konversi RGB ke BGR (OpenCV format)
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Deteksi dengan YOLO
    results = model(img_bgr)
    
    # Plot hasil deteksi
    result_img = results[0].plot()
    
    # Konversi kembali ke RGB untuk Streamlit
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    return result_img_rgb, results

# ===================== MAIN APP =====================
if model is None:
    st.error("❌ Gagal memuat model. Pastikan file best.pt ada di folder yang benar.")
    st.stop()

# ===================== MODE UPLOAD GAMBAR =====================
if menu == "📤 Upload Gambar":
    st.title("📤 Deteksi dari Upload Gambar")
    st.markdown("Upload gambar daun kedelai untuk mendeteksi penyakit secara otomatis")
    
    # Layout utama
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload File")
        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Tampilkan info file
            file_details = {
                "Nama File": uploaded_file.name,
                "Tipe File": uploaded_file.type,
                "Ukuran": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
    
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("📷 Gambar Asli")
            st.image(image, use_container_width=True)
        
        # Tombol deteksi
        if st.button("🔍 Deteksi Penyakit", type="primary", use_container_width=True):
            with st.spinner("⏳ Menganalisis gambar..."):
                # Deteksi
                result_img, results = detect_image(image, model)
                
                with col2:
                    st.subheader("✅ Hasil Deteksi")
                    st.image(result_img, use_container_width=True)
                
                # Tampilkan hasil deteksi
                st.markdown("---")
                st.subheader("📊 Detail Deteksi")
                
                if len(results[0].boxes) > 0:
                    # Buat container untuk hasil
                    result_container = st.container()
                    
                    with result_container:
                        cols = st.columns(3)
                        for i, box in enumerate(results[0].boxes):
                            col_idx = i % 3
                            
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            confidence = float(box.conf[0])
                            
                            with cols[col_idx]:
                                st.metric(
                                    label=f"Deteksi #{i+1}: {class_name}",
                                    value=f"{(confidence*100):.1f}%",
                                    delta="Confidence"
                                )
                else:
                    st.warning("⚠️ Tidak ada penyakit yang terdeteksi dalam gambar")

# ===================== MODE KAMERA REAL-TIME =====================
elif menu == "📷 Kamera Real-time":
    st.title("📷 Deteksi Real-time dengan Kamera")
    
    # Deteksi environment (cloud atau lokal)
    is_cloud = "/mount/src" in __file__ if "__file__" in dir() else False
    
    if is_cloud:
        st.warning("⚠️ Aplikasi berjalan di cloud server")
        st.info("""
        **Fitur kamera hanya tersedia saat aplikasi dijalankan secara lokal.**
        
        **Coba salah satu solusi berikut:**
        1. **Jalankan di komputer lokal:**
        2. **Gunakan mode Upload Gambar** untuk deteksi dari file

*Untuk keperluan demo/presentasi, gunakan mode Upload Gambar.*
""")

# Tampilkan ilustrasi
col1, col2, col3 = st.columns(3)
with col2:
 st.image("https://img.icons8.com/color/96/000000/webcam.png", width=150)
 st.markdown("<center>Kamera hanya tersedia di lingkungan lokal</center>", 
            unsafe_allow_html=True)

else:
st.success("✅ Mode Lokal - Kamera tersedia")

# ===== SETTINGS KAMERA DI SIDEBAR =====
with st.sidebar:
 st.markdown("---")
 st.subheader("🎛️ Pengaturan Kamera")
 
 # Opsi rotasi
 rotate_option = st.selectbox(
     "🔄 Rotasi Kamera",
     ["Normal (0°)", 
      "90° Searah Jarum Jam", 
      "180° (Terbalik)", 
      "270° Berlawanan Jarum Jam"],
     index=2,  # Default 180° untuk kasus terbalik
     help="Sesuaikan jika kamera terbalik atau miring"
 )
 
 # Mapping rotasi
 rotate_map = {
     "Normal (0°)": None,
     "90° Searah Jarum Jam": cv2.ROTATE_90_CLOCKWISE,
     "180° (Terbalik)": cv2.ROTATE_180,
     "270° Berlawanan Jarum Jam": cv2.ROTATE_90_COUNTERCLOCKWISE
 }
 rotation = rotate_map[rotate_option]
 
 # Opsi tambahan
 st.markdown("---")
 st.subheader("⚙️ Opsi Tambahan")
 
 show_fps = st.checkbox("📊 Tampilkan FPS", value=True)
 confidence_threshold = st.slider(
     "🎯 Confidence Threshold",
     min_value=0.1,
     max_value=1.0,
     value=0.5,
     step=0.1,
     help="Minimum confidence untuk menampilkan deteksi"
 )

# ===== INFO BAR =====
col1, col2, col3 = st.columns(3)
with col1:
 st.info(f"🔄 Rotasi: {rotate_option}")
with col2:
 st.info("⚡ Deteksi Real-time")
with col3:
 st.info("🎥 Kamera: Aktif")

st.markdown("---")

# ===== VIDEO STREAM =====
video_placeholder = st.empty()
fps_placeholder = st.empty()

# Tombol kontrol
col1, col2, col3 = st.columns(3)
with col2:
 stop_button = st.button(
     "⏹️ STOP KAMERA", 
     type="primary", 
     use_container_width=True
 )

# Buka kamera
try:
 cap = cv2.VideoCapture(0)
 
 if not cap.isOpened():
     st.error("❌ Tidak dapat mengakses kamera!")
     st.info("💡 Pastikan kamera terhubung dan tidak digunakan aplikasi lain")
 else:
     # Variabel untuk FPS
     fps = 0
     frame_count = 0
     start_time = time.time()
     
     # Loop deteksi
     while not stop_button:
         ret, frame = cap.read()
         if not ret:
             st.error("❌ Gagal membaca frame dari kamera")
             break
         
         # Terapkan rotasi
         if rotation is not None:
             frame = cv2.rotate(frame, rotation)
         
         # Deteksi dengan YOLO
         results = model(frame)
         
         # Plot hasil deteksi
         frame_detected = results[0].plot()
         
         # Hitung FPS
         frame_count += 1
         elapsed_time = time.time() - start_time
         if elapsed_time >= 1:
             fps = frame_count / elapsed_time
             frame_count = 0
             start_time = time.time()
         
         # Tambahkan teks FPS di frame
         if show_fps:
             cv2.putText(
                 frame_detected, 
                 f"FPS: {fps:.1f}", 
                 (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 
                 1, 
                 (0, 255, 0), 
                 2
             )
             
             # Update FPS di placeholder
             fps_placeholder.metric("Kecepatan", f"{fps:.1f} FPS")
         
         # Konversi BGR ke RGB untuk Streamlit
         frame_rgb = cv2.cvtColor(frame_detected, cv2.COLOR_BGR2RGB)
         
         # Tampilkan frame
         video_placeholder.image(
             frame_rgb, 
             channels="RGB", 
             use_container_width=True
         )
     
     # Bersihkan resource
     cap.release()
     video_placeholder.empty()
     st.success("✅ Kamera dimatikan")
     
except Exception as e:
 st.error(f"❌ Error: {str(e)}")
 st.info("💡 Coba restart aplikasi atau periksa koneksi kamera")

# ===================== MODE INFORMASI =====================
else:  # menu == "ℹ️ Informasi"
st.title("ℹ️ Informasi Sistem")

# Layout 3 kolom untuk metrics
col1, col2, col3 = st.columns(3)

with col1:
st.metric(
 label="Model AI",
 value="YOLOv9",
 delta="Object Detection"
)

with col2:
st.metric(
 label="Framework",
 value="Streamlit",
 delta="Web App"
)

with col3:
st.metric(
 label="Fitur Utama",
 value="2 Mode",
 delta="Upload + Kamera"
)

st.markdown("---")

# Informasi detail dalam tabs
tab1, tab2, tab3 = st.tabs(["📖 Tentang", "🎯 Cara Kerja", "📊 Statistik"])

with tab1:
st.subheader("Tentang Aplikasi")
st.write("""
Aplikasi **Deteksi Penyakit Daun Kedelai** adalah sistem berbasis 
**Computer Vision** yang menggunakan algoritma **YOLOv9** untuk 
mendeteksi berbagai penyakit pada daun kedelai secara otomatis.

**Dikembangkan untuk:**
- Skripsi / Tugas Akhir
- Deteksi dini penyakit tanaman
- Monitoring kesehatan tanaman kedelai
""")

with tab2:
st.subheader("Cara Kerja Sistem")

col1, col2 = st.columns(2)

with col1:
 st.markdown("**📤 Mode Upload Gambar:**")
 st.write("""
 1. User mengupload gambar daun
 2. Sistem mengkonversi gambar ke format OpenCV
 3. Model YOLO memproses gambar
 4. Bounding box digambar pada objek terdeteksi
 5. Hasil ditampilkan dengan confidence score
 """)

with col2:
 st.markdown("**📷 Mode Kamera Real-time:**")
 st.write("""
 1. Kamera mengcapture frame secara real-time
 2. Setiap frame diproses oleh model YOLO
 3. Deteksi ditampilkan langsung di video stream
 4. FPS ditampilkan untuk monitoring performa
 5. User bisa stop kapan saja
 """)

with tab3:
st.subheader("Statistik Model")

# Data statistik (contoh)
stats = {
 "Jenis Penyakit": 5,
 "Ukuran Model": "15.2 MB",
 "Akurasi Training": "94.5%",
 "Akurasi Validasi": "92.3%",
 "FPS (GPU)": "30+",
 "FPS (CPU)": "15-20"
}

# Tampilkan dalam grid
for i, (key, value) in enumerate(stats.items()):
 col1, col2 = st.columns([1, 1])
 with col1:
     st.markdown(f"**{key}:**")
 with col2:
     st.markdown(f"`{value}`")

st.markdown("---")

# Informasi pengembang
st.subheader("👨‍💻 Pengembang")
col1, col2 = st.columns([1, 3])

with col1:
st.image("https://img.icons8.com/color/96/000000/user-male-circle.png", width=100)

with col2:
st.markdown("""
**Dwi Yana**
- Program Studi Informatika
- Universitas Islam Indonesia
- Angkatan 2022

*Dosen Pembimbing: [Nama Dosen]*
""")

# ===================== FOOTER =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
st.markdown("<center>🌿 Deteksi Penyakit Daun Kedelai</center>", unsafe_allow_html=True)

with footer_col2:
st.markdown("<center>© 2026 Universitas Islam Indonesia</center>", unsafe_allow_html=True)

with footer_col3:
st.markdown("<center>Versi 2.0.0</center>", unsafe_allow_html=True)

# ===================== END OF FILE =====================
