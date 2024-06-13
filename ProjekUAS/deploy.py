import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

st.title("Gold Price Prediction")

st.subheader("Data Table")

gold_data = pd.read_csv('gld_price_data.csv')

col1, col2 = st.columns(2)

with col1:
    st.write("Data ")
    st.write(gold_data)
with col2:
    st.write("5 data terakhir")
    st.write(gold_data.tail(10))

st.markdown("""
## Penjelasan data

Table tersebut berisi data historis untuk beberapa aset keuangan dan nilai tukar mata uang. Berikut adalah analisis awal dan penjelasan untuk masing-masing kolom:

1. Date: Tanggal data diambil, dalam format MM/DD/YYYY.
2. SPX: Harga penutupan harian untuk indeks S&P 500 (pasar saham AS).
3. GLD: Harga penutupan harian untuk SPDR Gold Shares ETF (representasi harga emas).
4. USO: Harga penutupan harian untuk United States Oil Fund (representasi harga minyak mentah).
5. SLV: Harga penutupan harian untuk iShares Silver Trust (representasi harga perak).
6. EUR/USD: Kurs penutupan harian untuk pasangan mata uang Euro terhadap Dolar AS.

""")

st.subheader("Informasi atau Deskripsi Data Table")

st.write(gold_data.describe())

st.markdown("""
## Korelasi antar kolom

Korelasi adalah ukuran statistik yang menyatakan sejauh mana dua variabel berhubungan secara linier 
(artinya keduanya berubah bersama-sama dengan laju konstan). Ini adalah alat umum untuk menggambarkan 
hubungan sederhana tanpa membuat pernyataan tentang sebab dan akibat.

Berikut ini penggambaran korelasi antar table yang digambarkan dalam grafik:
""")

st.write("Correlation heatmap:")
gold_data['date_column'] = pd.to_datetime(gold_data['Date'], errors='coerce')
gold_data = gold_data.select_dtypes(include=[np.number])
correlation = gold_data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
st.pyplot(plt)

st.markdown("""
## Analisis Korelasi tersebut: 

- Harga emas (GLD) memiliki korelasi positif yang sangat kuat dengan harga perak (SLV), yang menunjukkan bahwa pergerakan harga emas cenderung sejalan dengan pergerakan harga perak.
- Indeks S&P 500 (SPX) memiliki korelasi negatif dengan harga minyak (USO) dan nilai tukar EUR/USD, menunjukkan bahwa ketika indeks S&P 500 naik, harga minyak dan nilai tukar EUR/USD cenderung turun, dan sebaliknya.
- Nilai tukar EUR/USD memiliki korelasi positif kuat dengan harga minyak (USO), menunjukkan hubungan yang signifikan antara nilai tukar ini dengan harga minyak.
""")

st.subheader("Model Learning")

st.markdown("""
## Random Forest 

Random Forest adalah algoritma machine learning yang kuat dan serbaguna untuk tugas klasifikasi dan regresi. Ini termasuk dalam kategori algoritma ensemble learning, yang berarti algoritma ini menggabungkan beberapa 
model pembelajaran mesin untuk menghasilkan hasil yang lebih baik. Random Forest secara khusus menggunakan sejumlah besar pohon keputusan dan menggabungkannya untuk mendapatkan prediksi yang lebih akurat dan stabil.
            
#### Model dalam program
Dalam system ini, data akan dibagi menjadi data trainging dan data testing yang dimana dibagi menjadi 20% 
data test dan 80% training. Lalu data tersebut dilakuakn training dengan metode random forest tersebut
Dan pelatihan tersebut jadilah sebuah model learning.

## Analisa System harga emas 
Berikut ini data yang dapat mempengaruhi harga emas, inputkan data tersebut:
            
""")
st.write("Masukkan data untuk diprediksi:")
    
spx = st.number_input('SPX', value=0.0)
uso = st.number_input('USO', value=0.0)
slv = st.number_input('SLV', value=0.0)
eur_usd = st.number_input('EUR/USD', value=0.0)

# Reshape input data
input_data = np.array([[spx, uso, slv, eur_usd]])
model = joblib.load('model.sav')

# Prediction button
if st.button('Predict'):
    # Load the model
    model = joblib.load('model.sav')
    
    # Prediction
    prediction = model.predict(input_data)
    
    st.markdown("""
    ## Analisis data input 
            
    """)
    st.write(f"Nilai SPX: {spx}")
    st.write(f"Nilai USO: {uso}")
    st.write(f"Nilai SLV: {slv}")
    st.write(f"Nilai EUR/USD: {eur_usd}")
    st.subheader(f"Prediksi harga emas: {prediction[0]:.5f}")