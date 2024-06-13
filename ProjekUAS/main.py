#!/usr/bin/env python
# coding: utf-8

# GOLD PRICE PREDICTION 
# Kelompok 1 :
# 1. Zaki Ahmad Basyary
# 2. Alih Bangun Wicaksono
# 3. Leo Fetri Hendli
# 4. Ahmad Mauluddin

# Importing the Libraries

# In[2]:


# Install library seaborn
get_ipython().system('pip install seaborn')


# In[3]:


# install library sciklearn
get_ipython().system('pip install scikit-learn')


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib


# Data Collection and Processing

# In[2]:


# Mengambil data pada file gld_prce_data.csv
gold_data = pd.read_csv('gld_price_data.csv')


# File "gld_price_data.csv" berisi data harga historis dari berbagai instrumen keuangan. Data ini terdiri dari beberapa kolom, masing-masing merepresentasikan variabel berbeda. Berikut adalah penjelasan detail tentang kolom-kolom dalam data ini:
# 
# 1. Date: Tanggal data diambil, dalam format bulan/hari/tahun (MM/DD/YYYY).
# 2. SPX: Nilai penutupan indeks S&P 500.
# 3. GLD: Harga penutupan ETF SPDR Gold Shares, yang melacak harga emas.
# 4. USO: Harga penutupan United States Oil Fund, LP, yang melacak harga minyak mentah.
# 5. SLV: Harga penutupan iShares Silver Trust, yang melacak harga perak.
# 6. EUR/USD: Nilai tukar EUR/USD, yaitu nilai tukar antara Euro dan Dolar AS.

# In[3]:


# Menampilkan 5 data pertama
gold_data.head()


# In[4]:


# menampilkan 5 data terakhir
gold_data.tail()


# In[5]:


# Menampilkan informasi jumlah baris dan kolom
gold_data.shape


# In[6]:


# Menampilkan informasi kolom dengan lebih detail
gold_data.info()


# In[7]:


# Mengecek jumlah nilai yang kosong (NULL)
gold_data.isnull().sum()


# In[8]:


# Menampilkan deskrispi dari data tersebut
gold_data.describe()


# Correlation:
# Korelasi digunakan untuk memahami hubungan antara dua variabel, yang bisa membantu dalam analisis data dan pengambilan keputusan.
# 
# 1. Positive Correlation, Ketika dua variabel bergerak dalam arah yang sama. Jika satu variabel meningkat, variabel lainnya juga meningkat, dan jika satu variabel menurun, variabel lainnya juga menurun. Korelasi positif sempurna adalah +1.
# 2. Negative Correlation. Ketika dua variabel bergerak dalam arah yang berlawanan. Jika satu variabel meningkat, variabel lainnya menurun, dan sebaliknya. Korelasi negatif sempurna adalah -1.
# 3. Zero Correlation,Tidak ada hubungan linear antara dua variabel. Perubahan dalam satu variabel tidak berhubungan dengan perubahan dalam variabel lainnya. 

# In[9]:


gold_data['date_column'] = pd.to_datetime(gold_data['Date'], errors='coerce')
gold_data = gold_data.select_dtypes(include=[np.number])


# In[10]:


correlation = gold_data.corr()


# In[11]:


# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')


# Analisis:
# - Harga emas (GLD) memiliki korelasi positif yang sangat kuat dengan harga perak (SLV), yang menunjukkan bahwa pergerakan harga emas cenderung sejalan dengan pergerakan harga perak.
# - Indeks S&P 500 (SPX) memiliki korelasi negatif dengan harga minyak (USO) dan nilai tukar EUR/USD, menunjukkan bahwa ketika indeks S&P 500 naik, harga minyak dan nilai tukar EUR/USD cenderung turun, dan sebaliknya.
# - Nilai tukar EUR/USD memiliki korelasi positif kuat dengan harga minyak (USO), menunjukkan hubungan yang signifikan antara nilai tukar ini dengan harga minyak.

# In[12]:


# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='green')


# 
# Plot distribusi menunjukkan bahwa harga GLD (SPDR Gold Shares) dalam dataset memiliki distribusi yang tidak sepenuhnya simetris, dengan adanya variasi dalam data. Puncak utama terletak di sekitar 120, menunjukkan bahwa harga GLD paling sering berada di sekitar nilai ini. Namun, terdapat juga puncak kedua di sekitar 160, menunjukkan adanya dua kelompok utama dalam data. Rentang harga GLD berkisar dari sekitar 60 hingga 200. 

# In[13]:


print(gold_data.columns)


# In[14]:


X = gold_data.drop(['GLD'],axis=1)
Y = gold_data['GLD']


# In[15]:


print(X)


# In[16]:


print(Y)


# Splitting into Training data and Test Data

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# Model Training: Random Forest Regressor

# In[18]:


regressor = RandomForestRegressor(n_estimators=100)


# In[19]:


# training the model
regressor.fit(X_train,Y_train)


# In[20]:


joblib.dump(regressor, 'main.sav')


# Model Evaluation

# In[21]:


# prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[22]:


print(test_data_prediction)


# In[51]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# Compare the Actual Values and Predicted Values in a Plot

# In[54]:


Y_test = list(Y_test)


# In[55]:


plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

