import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import math
from skimage.metrics import structural_similarity as ssim

# Konfigurasi halaman web Streamlit
st.set_page_config(page_title="PCA Image Compression", layout="wide")

st.title("🖼️ Aplikasi Kompresi Citra RGB dengan PCA & EDA")
st.write("Unggah gambar berwarna Anda, sesuaikan jumlah komponen utama (k), dan lihat analisis statistiknya secara real-time.")

# Fungsi Penunjang Perhitungan
def calculate_mse(original, reconstructed):
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0: return float('inf')
    return 10 * math.log10((max_pixel ** 2) / mse)

# --- 1. FITUR UNGGAH GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar berwarna (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar yang diunggah
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)
    tinggi, lebar, _ = img_array.shape
    max_k = min(tinggi, lebar)

   # --- SIDESBAR UNTUK INPUT USER ---
    st.sidebar.header("⚙️ Pengaturan PCA")
    k = st.sidebar.number_input("Ketik Jumlah Komponen (k)", min_value=1, max_value=max_k, value=int(max_k * 0.15), step=1)
    
    # --- 2. TAMPILKAN EDA AWAL ---
    st.subheader("📊 Analisis EDA Awal (Sebelum Kompresi)")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Dimensi Gambar", f"{tinggi} x {lebar} px")
    with col_stat2:
        st.metric("Total Piksel", f"{img_array.size:,}")
    with col_stat3:
        st.metric("Mean Intensitas", f"{np.mean(img_array):.2f}")
    with col_stat4:
        st.metric("Std Deviation", f"{np.std(img_array):.2f}")

    # --- 3. PROSES KOMPRESI PCA RGB ---
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    pca_r = PCA(n_components=k)
    pca_g = PCA(n_components=k)
    pca_b = PCA(n_components=k)
    
    R_rec = pca_r.inverse_transform(pca_r.fit_transform(R))
    G_rec = pca_g.inverse_transform(pca_g.fit_transform(G))
    B_rec = pca_b.inverse_transform(pca_b.fit_transform(B))
    
    img_rec = np.dstack((R_rec, G_rec, B_rec))
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)

    # --- 4. PERHITUNGAN METRIK ---
    avg_ev = np.mean([sum(pca_r.explained_variance_ratio_), 
                      sum(pca_g.explained_variance_ratio_), 
                      sum(pca_b.explained_variance_ratio_)]) * 100
    mse_val = calculate_mse(img_array, img_rec)
    psnr_val = calculate_psnr(mse_val)
    ssim_val = ssim(img_array, img_rec, data_range=255, channel_axis=-1)
    
    ukuran_asli = tinggi * lebar * 3
    ukuran_terkompresi = 3 * ((tinggi * k) + (k * lebar))
    rasio_kompresi = ukuran_asli / ukuran_terkompresi

    # --- 5. VISUALISASI GAMBAR SIDE-BY-SIDE ---
    st.subheader("🖼️ Perbandingan Visual")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_array, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(img_rec, caption=f"Hasil Rekonstruksi PCA (k={k})", use_container_width=True)
    with col3:
        # Error image E = |X - X^|
        error_img = np.mean(np.abs(img_array.astype(np.float32) - img_rec.astype(np.float32)), axis=2).astype(np.uint8)
        st.image(error_img, caption="Error Image (Terang = Detail Hilang)", use_container_width=True)

    # --- 6. VISUALISASI HISTOGRAM ---
    st.subheader("📈 EDA Lanjutan: Perbandingan Histogram Distribusi Warna")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(img_array.ravel(), bins=256, color='blue', alpha=0.4, label='Asli', density=True)
    ax.hist(img_rec.ravel(), bins=256, color='orange', alpha=0.4, label='Rekonstruksi', density=True)
    ax.set_title("Distribusi Intensitas Piksel (Asli vs Rekonstruksi)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # --- 7. TABEL EVALUASI DATA ---
    st.subheader("📋 Tabel Hasil Evaluasi Kompresi")
    eval_data = {
        "Metrik Evaluasi": ["Jumlah Komponen (k)", "Explained Variance (%)", "MSE", "PSNR (dB)", "SSIM Index", "Rasio Kompresi"],
        "Nilai Terhitung": [k, f"{avg_ev:.2f}%", f"{mse_val:.2f}", f"{psnr_val:.2f} dB", f"{ssim_val:.4f}", f"{rasio_kompresi:.2f} x lebih hemat"]
    }
    df_eval = pd.DataFrame(eval_data)
    st.table(df_eval)
