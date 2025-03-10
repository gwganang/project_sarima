import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from src.sarima_model import run_forecast

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Pengeluaran & Perencanaan ATK",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === FUNGSI PEMBANTU ===
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
    default_data_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(default_data_path)
        
        # Validasi dataset
        if 'Bulan' not in df.columns or len(df.columns) < 2:
            st.sidebar.error("Dataset harus memiliki kolom 'Bulan' dan minimal 1 kolom barang!")
            return pd.DataFrame(), []
        
        # Proses tanggal
        df['Bulan'] = pd.to_datetime(df['Bulan'], format='%b-%y', errors='coerce')
        df = df.dropna(subset=['Bulan']).set_index('Bulan').asfreq('MS')
        
        # Daftar barang
        items = [col for col in df.columns if col != 'Bulan']
        
        return df, items
    
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return pd.DataFrame(), []

# === LOAD DATA ===
df, items = load_data()

if not df.empty and items:
    # Parameter pengadaan
    st.sidebar.subheader("Parameter Pengadaan")
    selected_item = st.sidebar.selectbox("Pilih Barang:", items)
    harga_per_unit = st.sidebar.number_input(
        f"Harga {selected_item} per Unit (Rp)", 
        min_value=0, 
        value=5000, 
        step=500
    )
    buffer_stok = st.sidebar.slider("Buffer Stok (%)", 0, 50, 10)

    # Parameter SARIMA
    with st.sidebar.expander("Parameter Model SARIMA"):
        p = st.slider("p (AR)", 0, 5, 1)
        d = st.slider("d (Differencing)", 0, 2, 1)
        q = st.slider("q (MA)", 0, 5, 1)
        seasonal = st.checkbox("Komponen Musiman", True)

    # Halaman utama
    page = st.sidebar.radio("Navigasi", ["Data Historis", "Prediksi & SPK", "Dokumentasi"])

    if page == "Data Historis":
        st.header(f"Data Historis {selected_item}")
        
        # Plot data
        fig = px.line(
            df.reset_index(),
            x='Bulan',
            y=selected_item,
            title=f'Data Pengeluaran {selected_item}',
            labels={selected_item: 'Jumlah Pengeluaran'},
            template='plotly'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistik
        with st.expander("Statistik Lengkap"):
            st.write(df[[selected_item]].describe())

    elif page == "Prediksi & SPK":
        st.header(f"Prediksi & Perencanaan {selected_item}")
        
        if st.button("Mulai Prediksi dan SPK"):
            with st.spinner("Memproses model..."):
                try:
                    # Cek panjang data
                    if len(df[selected_item]) < 12:
                        st.error("Data terlalu sedikit! Minimal 12 bulan data")
                        raise ValueError("Data insufficient")
                    
                    # Jalankan prediksi
                    forecast_df, results, seasonal_status = run_forecast(
                        df[selected_item], p, d, q, seasonal
                    )
                    
                    # Tampilkan peringatan berdasarkan status
                    if seasonal_status == "diff_only":
                        st.warning("⚠️ Hanya differencing musiman yang digunakan (data < 36 bulan)")
                    elif seasonal_status is False:
                        st.warning("⚠️ Komponen musiman dinonaktifkan karena data < 24 bulan")
                    elif isinstance(seasonal_status, str):
                        st.error(f"Model gagal: {seasonal_status}")

                    # Generate tanggal
                    forecast_dates = pd.date_range(
                        start=df.index[-1] + pd.DateOffset(months=1),
                        periods=12,
                        freq='MS'
                    )
                    forecast_df.index = forecast_dates
                    
                    # Hitung SPK
                    spk_df = forecast_df.copy()
                    spk_df['Buffer'] = (spk_df['Prediksi'] * buffer_stok/100).round(2)
                    spk_df['Jumlah Pengadaan'] = (spk_df['Prediksi'] + spk_df['Buffer']).round(2)
                    spk_df['Biaya (Rp)'] = (spk_df['Jumlah Pengadaan'] * harga_per_unit).astype(int)
                    
                    # Total
                    total_pengadaan = spk_df['Jumlah Pengadaan'].sum()
                    total_biaya = spk_df['Biaya (Rp)'].sum()
                    rata_pengadaan = total_pengadaan / 12
                    rata_biaya = total_biaya / 12

                    # Tampilkan hasil
                    st.subheader("Hasil Prediksi & SPK")
                    st.dataframe(spk_df.style.format({
                        "Prediksi": "{:.2f}",
                        "Buffer": "{:.2f}",
                        "Jumlah Pengadaan": "{:.2f}",
                        "Biaya (Rp)": "Rp {:,.0f}"
                    }))
                    
                    # Plot
                    fig = px.line(
                        df.reset_index(),
                        x='Bulan',
                        y=selected_item,
                        title=f'Prediksi Pengeluaran {selected_item}',
                        labels={selected_item: 'Jumlah'},
                        template='plotly'
                    )
                    fig.add_scatter(
                        x=spk_df.index,
                        y=spk_df['Prediksi'],
                        mode='lines',
                        name='Prediksi',
                        line=dict(color='red', dash='dash')
                    )
                    fig.add_bar(
                        x=spk_df.index,
                        y=spk_df['Jumlah Pengadaan'],
                        name='Rekomendasi Pengadaan',
                        marker_color='green'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ringkasan
                    st.subheader("Ringkasan Perencanaan")
                    st.write(f"Total Pengadaan 1 Tahun: **{total_pengadaan:,.2f} unit**")
                    st.write(f"Total Biaya: **Rp {total_biaya:,.0f}**")
                    st.write(f"Rata-rata Pengadaan/Bulan: **{rata_pengadaan:.2f} unit**")
                    st.write(f"Rata-rata Biaya/Bulan: **Rp {rata_biaya:,.0f}**")
                    
                    # Download
                    st.download_button(
                        "Unduh SPK",
                        spk_df.to_csv(index=True),
                        f"spk_{selected_item.lower().replace(' ', '_')}.csv",
                        "text/csv"
                    )
                    
                except np.linalg.LinAlgError:
                    st.error("Model tidak konvergen! Coba parameter lebih sederhana.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")

    elif page == "Dokumentasi":
        st.header(f"Dokumentasi Analisis {selected_item}")
        
        # Uji stasioneritas
        st.subheader("Uji ADF")
        adf_result = sm.tsa.adfuller(df[selected_item])
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"p-value: {adf_result[1]:.4f}")
        
        # Decomposisi
        st.subheader("Decomposisi Data")
        decomposition = sm.tsa.seasonal_decompose(df[selected_item], model='additive', period=12)
        fig = decomposition.plot()
        st.pyplot(fig)
else:
    st.warning("Upload dataset yang valid atau periksa dataset default!")