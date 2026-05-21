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

st.title("Aplikasi Kompresi Citra RGB dengan PCA dan EDA")
st.write("Unggah gambar berwarna Anda, ketik beberapa nilai komponen (k), dan bandingkan hasilnya.")

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

    # --- SIDEBAR UNTUK INPUT MULTI-VARIABEL ---
    st.sidebar.header("Pengaturan PCA")
    st.sidebar.write(f"Batas maksimal k: {max_k}")
    
    # Menggunakan text_input agar bisa menerima input koma
    k_input = st.sidebar.text_input("Masukkan variasi nilai k (pisahkan dengan koma)", "10, 50, 100")
    
    # Proses parsing input teks menjadi list angka (integer)
    try:
        # Memecah teks berdasarkan koma, menghapus spasi, dan mengubah ke integer
        k_values = [int(x.strip()) for x in k_input.split(',')]
        # Membuang angka yang melebihi batas maksimal gambar atau bernilai negatif/nol
        k_values = [k for k in k_values if 0 < k <= max_k]
        
        if not k_values:
            st.sidebar.error("Masukkan setidaknya satu angka yang valid.")
            st.stop() # Hentikan eksekusi ke bawah jika input tidak valid
    except ValueError:
        st.sidebar.error("Format salah! Masukkan angka yang dipisahkan koma. Contoh: 10, 50, 100")
        st.stop()
    
    # --- 2. TAMPILKAN EDA AWAL ---
    st.subheader("Analisis EDA Awal (Sebelum Kompresi)")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Dimensi Gambar", f"{tinggi} x {lebar} px")
    with col_stat2:
        st.metric("Total Piksel", f"{img_array.size:,}")
    with col_stat3:
        st.metric("Mean Intensitas", f"{np.mean(img_array):.2f}")
    with col_stat4:
        st.metric("Std Deviation", f"{np.std(img_array):.2f}")

    # --- 3. PROSES KOMPRESI DAN EVALUASI ---
    st.subheader("Perbandingan Visual dan Histogram")
    
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    eval_results = []
    
    # Membuat Tabs di Streamlit agar rapi (satu tab untuk satu nilai K)
    tabs = st.tabs([f"K = {k}" for k in k_values])
    
    # Melakukan perulangan untuk setiap nilai k yang dimasukkan user
    for idx, k in enumerate(k_values):
        
        # Proses PCA
        pca_r = PCA(n_components=k)
        pca_g = PCA(n_components=k)
        pca_b = PCA(n_components=k)
        
        R_rec = pca_r.inverse_transform(pca_r.fit_transform(R))
        G_rec = pca_g.inverse_transform(pca_g.fit_transform(G))
        B_rec = pca_b.inverse_transform(pca_b.fit_transform(B))
        
        img_rec = np.dstack((R_rec, G_rec, B_rec))
        img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)

        # Perhitungan Metrik
        avg_ev = np.mean([sum(pca_r.explained_variance_ratio_), 
                          sum(pca_g.explained_variance_ratio_), 
                          sum(pca_b.explained_variance_ratio_)]) * 100
        mse_val = calculate_mse(img_array, img_rec)
        psnr_val = calculate_psnr(mse_val)
    
        ssim_val = ssim(img_array, img_rec, data_range=255, channel_axis=-1)
        
        ukuran_asli = tinggi * lebar * 3
        ukuran_terkompresi = 3 * ((tinggi * k) + (k * lebar))
        rasio_kompresi = ukuran_asli / ukuran_terkompresi
        
        # Menyimpan data metrik ke list untuk dijadikan tabel nanti
        eval_results.append({
            "Jumlah Komponen (k)": k,
            "Explained Variance": f"{avg_ev:.2f}%",
            "MSE": round(mse_val, 2),
            "PSNR (dB)": round(psnr_val, 2),
            "SSIM Index": round(ssim_val, 4),
            "Rasio Kompresi": round(rasio_kompresi, 2)
        })

        # Memasukkan visualisasi ke dalam Tab yang sesuai
        with tabs[idx]:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_array, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(img_rec, caption=f"Hasil Rekonstruksi PCA (k={k})", use_container_width=True)
            with col3:
                error_img = np.mean(np.abs(img_array.astype(np.float32) - img_rec.astype(np.float32)), axis=2).astype(np.uint8)
                st.image(error_img, caption="Error Image (Terang = Detail Hilang)", use_container_width=True)

            # Histogram
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.hist(img_array.ravel(), bins=256, color='blue', alpha=0.4, label='Asli', density=True)
            ax.hist(img_rec.ravel(), bins=256, color='orange', alpha=0.4, label='Rekonstruksi', density=True)
            ax.set_title(f"Distribusi Intensitas Piksel (k={k})")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    # --- 4. TABEL EVALUASI KESELURUHAN ---
    st.subheader("Tabel Hasil Evaluasi Kompresi")
    df_eval = pd.DataFrame(eval_results)
    st.dataframe(df_eval, use_container_width=True)
