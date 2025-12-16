import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    """
    Menampilkan sidebar informasi Proyek.
    Versi ini TIDAK MEMILIKI SLIDER THRESHOLD (Fixed Value).
    """
    st.sidebar.header("🛡️ Argus / Drowsiness System")
    
    st.sidebar.info(
        """
        **Status Sistem:** Aktif
        **Mode:** Real-time Monitoring
        """
    )
    
    st.sidebar.markdown("---")
    
    # --- ABOUT SECTION ---
    st.sidebar.subheader("ℹ️ Tentang Model")
    st.sidebar.markdown(
        """
        Sistem ini menggunakan **Convolutional Neural Network (CNN)** untuk mendeteksi kelelahan berdasarkan kondisi mata.
        
        **Indikator:**
        - 🟢 **Aman:** Mata Terbuka
        - 🔴 **Bahaya:** Mata Tertutup / Mengantuk
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Project Tim Drowsiness Detection © 2025")

    # --- PENTING ---
    # Meskipun tidak ada slider, kita harus me-return nilai default
    # agar app.py tidak error.
    # Ubah angka 0.50 ini jika ingin sistem lebih sensitif atau tidak.
    FIXED_THRESHOLD = 0.50 
    
    return FIXED_THRESHOLD

def show_stats(drowsiness_score, history_list):
    """
    Menampilkan statistik dan grafik live.
    
    Args:
        drowsiness_score (float): Nilai kantuk saat ini (0.0 - 1.0).
        history_list (list): List berisi data history skor kantuk (untuk grafik).
    """
    
    # 1. Tampilkan Metric Utama (Angka Besar)
    # Kita buat 3 kolom kecil untuk statistik
    m1, m2 = st.columns(2)
    
    with m1:
        st.metric(
            label="Skor Kantuk (Live)", 
            value=f"{drowsiness_score:.2f}",
            delta="Bahaya" if drowsiness_score > 0.5 else "Aman",
            delta_color="inverse" # Merah jika naik (bahaya), Hijau jika turun
        )
        
    with m2:
        # Menghitung rata-rata dari history terakhir
        if len(history_list) > 0:
            avg_score = np.mean(history_list)
        else:
            avg_score = 0.0
        st.metric(label="Rata-rata (30 frame)", value=f"{avg_score:.2f}")

    # 2. Tampilkan Grafik Garis (Line Chart)
    st.markdown("### 📈 Grafik Aktivitas Mata")
    
    if len(history_list) > 0:
        # Konversi list ke DataFrame agar Streamlit bisa membacanya sebagai grafik
        chart_data = pd.DataFrame(history_list, columns=["Tingkat Kantuk"])
        
        # Tampilkan Line Chart
        st.line_chart(chart_data, height=150)
    else:
        st.info("Menunggu data stream...")