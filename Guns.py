import streamlit as st
import pandas as pd
import joblib

# CSS styling untuk judul aplikasi dan sidebar
st.markdown("""
<style>
h1 {
    color: #DC143C;
    text-align: center;
    font-size: 40px;
    margin-top: -30px; /* Menurunkan judul sedikit ke atas */
}
.sidebar .sidebar-content {
    background-color: #e0f7fa; /* Warna latar belakang sidebar */
    padding: 20px;
    margin-top: 20px; /* Menambahkan margin atas untuk sidebar */
}
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title('Aplikasi Deteksi Dini DBD')

# Menu navigasi di sidebar
st.sidebar.title('Menu Navigasi')
menu_selection = st.sidebar.radio(
    '',
    ['Beranda', 'Pre Procces Data', 'Smote Data', 'Klasifikasi SVM', 'Uji Coba', 'Hubungi Kami']
)

# Mapping antara teks menu dan tautan halaman
pages = {
    'Beranda': '',
    'Pre Procces Data': 'preproccess.py',
    'Smote Data': 'smote.py',
    'Klasifikasi SVM': 'svm.py',
    'Uji Coba': 'uji_coba.py',
    'Hubungi Kami': 'kontak.py'
}

# Halaman Uji Coba
if menu_selection == 'Uji Coba':
    st.write("### Input Data Pasien untuk Uji Coba")
    nama = st.text_input('Nama')
    trombosit = st.number_input('Jumlah Trombosit Pasien', min_value=0, max_value=500000, step=1)
    hct = st.number_input('Jumlah Hematokrit Pasien', min_value=0, max_value=500000, step=1)
    igg = st.selectbox('IgG', ['+', '-'])
    igm = st.selectbox('IgM', ['+', '-'])
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
    umur = st.number_input('Umur', min_value=0, max_value=150, step=1)
    ruam = st.selectbox('Ruam', ['YA', 'TIDAK'])
    nyeri_kepala = st.selectbox('Nyeri Kepala', ['YA', 'TIDAK'])
    nyeri_otot = st.selectbox('Nyeri Otot', ['YA', 'TIDAK'])
    demam = st.selectbox('Demam', ['YA', 'TIDAK'])
    submitted = st.button('Submit')

    # Menampilkan hasil prediksi setelah data pasien diinput
    if submitted:
        # Konversi input menjadi format DataFrame yang sesuai
        data = {
            'TROMBOSIT': [trombosit],
            'HCT': [hct],
            'IgG': [1 if igg == '+' else 0],
            'IgM': [1 if igm == '+' else 0],
            'JENIS KELAMIN': [0 if jenis_kelamin.lower() == 'perempuan' else 1],
            'UMUR': [umur],
            'RUAM': [1 if ruam.lower() == 'ya' else 0],
            'NYERI KEPALA': [1 if nyeri_kepala.lower() == 'ya' else 0],
            'NYERI OTOT': [1 if nyeri_otot.lower() == 'ya' else 0],
            'DEMAM': [1 if demam.lower() == 'ya' else 0]
        }
        new_data = pd.DataFrame(data)

        # Baca X_test.csv
        datatest = pd.read_csv('Data/X_test.csv')

        # Gabungkan new_data ke datatest
        datatest = pd.concat([datatest, new_data], ignore_index=True)
        datanorm = joblib.load('models/standard_scaler.pkl').transform(datatest)  # Ubah fit_transform ke transform
        # print(datanorm)
        datapredict = joblib.load('models/lala.pkl').predict(datanorm)

        # Tampilkan datatest
        # st.write(datatest)
        # st.write(datanorm)
        # st.write('### Data yang Diinput:')
        # st.write(f'- Nama: {nama}')
        # st.write(f'- Jumlah Trombosit Pasien: {trombosit}')
        # st.write(f'- Jumlah Hematokrit Pasien: {hct}')
        # st.write(f'- IgG: {igg}')
        # st.write(f'- IgM: {igm}')
        # st.write(f'- Jenis Kelamin: {jenis_kelamin}')
        # st.write(f'- Umur: {umur}')
        # st.write(f'- Ruam: {ruam}')
        # st.write(f'- Nyeri Kepala: {nyeri_kepala}')
        # st.write(f'- Nyeri Otot: {nyeri_otot}')
        # st.write(f'- Demam: {demam}')

        # # Tampilkan hasil prediksi
        result = 'Positif DBD' if datapredict[-1] == 1 else 'Negatif DBD'
        st.write(f'### Hasil Prediksi: {result}')

# Tautan ke halaman lain
else:
    link = pages[menu_selection]
    if link:
        st.write(f"Untuk melanjutkan ke {menu_selection}, silakan klik tautan berikut: [{menu_selection}]({link})")
