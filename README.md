# Customer Personality Prediction To Boost Marketing Campaign By Using Machine Learning

# Business Understanding 

Analisis Kepribadian Pelanggan adalah analisis rinci terhadap pelanggan dari suatu perusahaan. Analisis ini dapat membantu bisnis untuk lebih memahami pelanggannya sehingga dapat mencari inovasi produk yang sesuai dengan perilaku dan kebutuhan yang lebih spesifik dari berbagai segmentasi pelanggan. Hal ini akan menjadi lebih efektif dan efisien daripada perusahaan harus mengeluarkan banyak uang yang kurang berguna untuk memasarkan produk baru ke setiap pelanggan yang ada di database perusahaan. Dengan adanya analisis terlebih dahulu maka perusahaan dapat memasarkannya kepada pelanggan yang paling mungkin membeli produk tersebut sesuai dengan segmentasinya.

Dengan memahami perilaku pelanggan dapat membantu meningkatkan pertumbuhan perusahaan dengan lebih cepat melalui peningkatan layanan bagi calon pelanggan setia. Selain itu, menganalisis data historis dari marketing campign juga dapat membantu perusahaan untuk meningkatkan kinerja dan juga menargetkan pelanggan yang tepat. Sehingga diharapkan dengan mengembangkan cluster prediction model, dapat semakin menyederhanakan proses pengambilan keputusan untuk bisnis.

## Problem Statement

Ada sebuah supermarket retail yang menjual berbagai jenis produk seperti Ikan, Daging, Buah-buahan, Produk Manis, Coke, dan Produk Emas. Dalam 6 bulan terakhir, tim marketing telah melakukan promosi dengan memberikan voucher diskon kepada seluruh pelanggan melalui Broadcast Message. Namun mengalami beberapa permasalahan sebagai berikut:

- Rendahnya respon dari pelanggan
- Biaya yang dikeluarkan untuk marketing campaign tidak efisien 
- Keuntungan yang didapat tidak sebanding dengan biaya yang dikeluarkan untuk marketing campaign

Berdasarkan permasalahan yang dialami, Tim marketing telah meminta tim data untuk menganalisis permasalahan tersebut. Dengan mengolah data historis yang ada diharapkan hasil analisis tersebut nantinya dapat digunakan oleh perusahaan untuk Menyusun startgi marketing yang baru sehingga marketing campaign selanjutnya dapat menargetkan sesuai dengan karakteristik pelanggan dan dapat meningkatkan kinerja bisnis dari perusahaan tersebut.

## **Goals**

Proyek ini diinisiasi untuk memprediksi customer personality dengan menggunakan machine learning dan digunakan untuk meningkatkan kinerja dari marketing campaign. Hal ini akan membantu perusahaan untuk menganalisis dan memahami perilaku pelanggannya, sehingga dapat menyusun strategi dan menargetkan ulang objek marketingnya berdasarkan segmentasi pelanggan.

## **Objectives**

Mengembangkan clustering model untuk memprediksi segmentasi pelanggan dalam analisis customer personality untuk membantu bisnis menyesuaikan produk sesuai dengan target berdasarkan beberapa segmentasi pelanggan.

## Work Environment

- **Tools**

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/jupyter%20logo.png" height="100"/>

- **Programming Language**

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Python.png" height="100"/>

- **Library**

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/pandas%20logo.png" height="100"/>

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/logo%20matplotlib.png" height="100"/>

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/scikit%20learn.png" height="100"/>

- **Dataset**

[lihat disini](https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/dataset/marketing_campaign_data.csv) 

## Data Description 

**About Dataset:**

Dataset ini berisi `2,240 samples` dan `29 features` :

Promosi yang mendapat respon :
- AcceptedCmp1 - 1 jika pelanggan menerima tawaran pada promosi yang pertama, 0 sebaliknya
- AcceptedCmp2 - 1 jika pelanggan menerima tawaran pada promosi yang kedua, 0 sebaliknya
- AcceptedCmp3 - 1 jika pelanggan menerima tawaran pada promosi yang ketiga, 0 sebaliknya
- AcceptedCmp4 - 1 jika pelanggan menerima tawaran pada promosi yang keempat, 0 sebaliknya
- AcceptedCmp5 - 1 jika pelanggan menerima tawaran pada promosi yang kelima, 0 sebaliknya
- Response - 1 jika pelanggan menerima tawaran pada promosi yang terakhir, 0 sebaliknya
- Complain - 1 jika pelanggan complain dalam 2 tahun terakhir

Informasi terkait Pelanggan :
- ID - ID Pelanggan
- Year_Birth - Tanggal ulang tahun pelanggan
- Education - Tingkat Pendidikan pelanggan
- Marital_Status - Status pernikahan pelanggan
- Income - Pendapatan rumah tangga tahunan pelanggan
- Kidhome - Jumlah anak kecil yang dimiliki pelanggan
- Teenhome - Jumlah remaja yang dimiliki pelanggan
- DtCustomer - Tanggal pendaftaran pelanggan di perusahaan
- Recency - Jumlah hari sejak pembelian terakhir oleh pelanggan

Jenis Produk Penjualan :
- MntCoke - Jumlah yang dibelanjakan untuk produk Coke dalam 2 tahun terakhir
- MntFruits - Jumlah yang dibelanjakan untuk produk buah-buahan dalam 2 tahun terakhir
- MntMeatProducts - Jumlah yang dibelanjakan untuk produk daging dalam 2 tahun terakhir
- MntFishProducts - Jumlah yang dibelanjakan untuk produk ikan dalam 2 tahun terakhir
- MntSweetProducts - Jumlah yang dibelanjakan untuk produk manis dalam 2 tahun terakhir
- MntGoldProds - Jumlah yang dibelanjakan untuk produk emas dalam 2 tahun terakhir


Jumlah Pembelian per Jenis :
- NumDealsPurchases - Jumlah pembelian yang dilakukan dengan diskon
- NumWebPurchases - Jumlah pembelian yang dilakukan melalui situs web perusahaan
- NumCatalogPurchases - Jumlah pembelian yang dilakukan menggunakan katalog
- NumStorePurchases - Jumlah pembelian yang dilakukan langsung di toko
- NumWebVisitsMonth - Jumlah kunjungan ke situs web perusahaan dalam sebulan terakhir

Biaya dan Pendapatan
- Z_CostContact = Biaya untuk menghubungi pelanggan
- Z_Revenue = Pendapatan setelah klien menerima marketing campaign


# Data Understanding

### Dataset Information, Checking Duplicate Rows, dan Missing Values
- Dataset terdiri dari 29 kolom dan 2240 baris data.
- Ada 3 tipe data: int64, objek, float64.
- Kolom Pendapatan memiliki 2216 nilai bukan null dan 24 nilai null.

Karena data yang tersedia tidak ekstensif, null values pada kolom "Income" akan dilakukan imputasi pada tahap data preprocessing :
- Imputasi dengan median karena distribusinya yang sangat miring
- Dengan Multivariate (Imputasi MICE, Imputer KNN, dll)

### Statistical Summary

**Numerical**

**Year_Birth**
- Tahun lahir (min) tertua adalah 1893, namun dari data tersebut menunjukkan adanya kemungkinan kesalahan entri data sehingga memerlukan proses lebih lanjut untuk mendeteksi Outlier.
- Ekstraksi fitur kemungkinan akan dilakukan untuk menghitung Usia berdasarkan tahun berjalan (2014) untuk mendapatkan data.

**Income**
- Rata-rata dari Income adalah 52247251.354 dan Median/nilai tengahnya adalah 51381500. Nilai rata-rata yang lebih tinggi daripada median menunjukkan bahwa distribusi datanya akan sedikit condong ke kanan.
- Data juga menunjukkan rentang minimum  memiliki rentang 1730000 hingga maksimum 666666000, yang menunjukkan kemungkinan adanya outlier. Oleh karena itu, perlu diproses melalui Transformasi Log/Normalisasi sebelum lanjut ke modeling.

**Recency, Kidhome, Teenhome**
- Kesamaan antara nilai mean dan median menunjukkan bahwa kemungkinan distribusinya normal-skewed atau distribusi bimodal. Namun, untuk memastikan perlu dilakukan analisis univariat.
- Untuk Kidhome dan Teenhome, perlu untuk menambahkan fitur baru yaitu 'Dependents' yang berguna untuk lebih menunjukkan berapa jumlah anggota keluarga yang menjadi tanggungan.

**Type of Products**
- MntCoke: mean = 303935.714, median = 173500
- MntFruits: mean = 26302.232, median = 8000
- MntMeatProducts: mean = 166950, median = 67000
- MntFishProducts: mean = 37525.446, median = 12000
- MntSweetProducts: mean = 27062.946, median = 8000
- MntGoldProds: mean = 44021.875, median = 24000

Melihat adanya perbedaan yang signifikan antara beberapa nilai mean dan median, menunjukkan kemungkinan terdapat jumlah outlier yang tinggi dan distribusi yang miring. Oleh karena itu, diperlukan Data Preprocessing dengan Log Transformation agar data dapat sesuai dengan normalitas.

**Moderately Positively Skewed (Slightly skewed to the right)**
- NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth

Pada kolom-kolom tersebut, nilai mean dan median mempunyai selisih yang relatif kecil, namun masih ada kemungkinan distribusinya miring ke kanan. Diperlukan visualisasi lebih detail untuk memastikan hal ini. Selain itu juga Preprocessing data dengan Log Transformation diperlukan agar data sesuai dengan normalitas.

**Z_CostContact, Z_Revenue**
Kedua kolom tersebut hanya memiliki 1 nilai unik, sehingga dalam proses modelling tidak akan digunakan karena tidak memberikan informasi yang signifikan untuk prediction modelling.

**Categorical**

- Pada data ID, semua nilai hanya muncul satu kali, menandakan tidak ada duplikat ID pada data.
- Kebanyakan nasabah lahir pada tahun 1976 yang berarti berumur 38 tahun, dengan total 89 orang.
- Pada data Education, mayoritas nasabah bergelar Sarjana (S1) dengan total  1.127 orang, sehingga jauh lebih besar dibandingkan kategori Pendidikan lainnya.
- Pada data Marital_Status, mayoritas pelanggannya adalah Menikah sebanyak 864 orang, dan beberapa kategori lain dapat disederhanakan menjadi:
   - Gabungan kategori Cerai, Janda dan Duda menjadi Lajang.
   - Mengubah kategori Bertunangan menjadi Menikah.
- Pada data AcceptedCmp(1-5), mayoritas pelanggan tidak merespon marketing campaign.
- Pada data Complain, mayoritas pelanggan tidak melakukan komplain terhadap kampanye.
- Pada data Response terjadi ketidakseimbangan data atau Imbalanced Data:
   - Tidak merespons = 1906
   - Meresponse = 334

Langkah-langkah untuk Preprocessing data:
- Melakukan pengelompokan data pada kolom Marital_Status untuk menyederhanakan data yang mempunyai arti serupa.
- Feature Encoding pada kolom Education dan Marital_Status untuk modeling, karena saat ini representasi data numeriknya masih kurang.

# Exploratory Data Analysis (EDA)

### Univariate Analysis

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150105.png" height="200"/>

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150123.png" height="200"/>

Terdapat outliers pada kolom : Year_Birth, Income, MntCoke, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, and NumWebVisitMonth.

- Pada kolom Year_Birth column, outliers terjauh berada di bawah 1900.
- Pada kolom Income, outliers terjauh berada di atas 600M.
- Pada kolom MntCoke, outliers berada di atas 1.2M.
- Pada kolom MntFruits, outliers berada pada antara 80k to 200k.
- Pada kolom MntMeatProducts, outliers terjauh berada di antara 1.75M.
- Pada kolom MntFishProducts, outliers di antara 125k hingga di atas 250k.
- Pada kolom MntSweetProducts, outliers terjauh berada di antara 250k.
- Pada kolom MntGoldProds, outliers terjauh berada di antara 350k.
- Pada kolom NumDealsPurchases, outliers terjauh adalah 15.
- Pada kolom NumWebPurchases, outliers berada di antara 25.
- Pada kolom NumCatalogPurchases, outliers terjauh berada di atas 25.
- Pada kolom NumWebVisitsMonth, outliers terjauh berada di angka 20.

Langkah-langkah untuk PreProcessing Data:
- Gunakan Log Transformation untuk Feature Scaling dan Outliers Handling, karena berguna untuk meminimalkan outlier dan berpotensi membantu mendapatkan distribusi data berbentuk lonceng/normal. Pilihan ini digunakan karena data yang tersedia terbatas yaitu hanya 2240 baris, dan menghindari drop data.
- Alternatif yang juga dapat diambil yaitu membersihkan data dengan menghilangkan outlier berdasarkan IQR atau Z-score, namun dapat mengurangi jumlah data yang tersedia.

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150248.png" height="200"/>

Berdasarkan diagram Boxplot, ViolinPlot, dan HistPlot di atas, kita dapat mengidentifikasi beberapa variabel yang memiliki outlier dan ada juga yang menunjukkan Skewed Distribution.

**1. Normal distribution**
- Recency: Berdistribusi Normal
- Year_Birth: Berdistribusi Cukup Normal
- NumWebVisitsMonth: Berdistribusi Cukup Normal

**2. Uniform distribution**
- Z_CostContact: Uniform Distribution - Hanya ada 1 nilai
- Z_Revenue: Uniform Distribution - Hanya ada 1 nilai

**3. Positive skewed distribution**
- Income
- MntCoke
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds
- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases

**4. Bimodal distribution**
- Kidhome
- Teenhome

**Rekomendasi untuk data preprocessing**
Data dengan Positive Skewed Distribution harus dilakukan Log Transformation agar menjadi normal distribution. Sedangkan untuk variable yang lain, feature scaling akan dilakukan dengan menggunakan Standardization 

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150440.png" height="200"/>

**Observations**
- Terlalu banyak kategori pada kolom ID and Year_Birth
- Kategori Pendidikan tertinggi adalah Bachelor's Degree (S1)
- Pada kategori Marital Status, mayoritas pelanggan sudah Menikah sebanyak 864 orang, dan beberapa kategori lain dapat disederhanakan menjadi:
   - Penggabungan kategori Cerai, Janda dan Duda menjadi Lajang.
   - Mengubah kategori Bertunangan menjadi Menikah.
- Kolom Kidhome dan Teenhome mayoritas pelanggannya tidak memiliki anak-anak dan remaja.
- Kolom AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Complain, dan Respon didominasi nilai 0 yang berarti No Response / No Complaint.
- Kolom Respon untuk pelanggan yang merespon memiliki ketidakseimbangan yang sangat tinggi:
   - Tidak merespons = 1906
   - Menanggapi = 334

### Multivariate Analysis

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150529.png" height="200"/>

**MntMeatProduct** berkorelasi kuat positif dengan NumCatalogPurchases. Menunjukkan bahwa Sebagian besar pembelian produk melalui katalog (koefisien korelasi : 0.72).

**Income**
- Pelanggan dengan Income lebih tinggi cenderung melakukan pembelian lebih banyak.
- Kolom Income berkorelasi positif siginifikan terhadap MntCoke, MntFruits, MntMeatProduct, MntFishProduct, MntSweetProduct, dan MntGoldProduct.
- Kolom Income memiliki berkorelasi positif signifikan dengan NumWebPurchases, NumCatalogPurchases, dan NumStorePurchases, sedangkan berkorelasi negatif yang relatif besar dengan NumWebVisitsMonth. Hal ini menunjukkan bahwa pelanggan yang memiliki Income lebih tinggi, sebagian besar lebih memilih pembelian melalui Web, Catalog, dan Toko. Sedangkan pelanggan yang memiliki income lebih rendah cenderung lebih banyak melakukan pembelian melalui website atau lebih sering mengunjungi website.

**Product**
- Kolom MntCoke berkorelasi positif yang relatif besar dengan MntMeatProduct, sehingga menunjukkan bahwa pelanggan yang membeli Coke kemungkinan besar juga akan membeli produk daging.
- Kolom MntFruits berkorelasi positif signifikan dengan MntMeatProduct, MntFishProduct, dan MntSweetProduct, sehingga menunjukkan bahwa pelanggan cenderung membeli produk-produk tersebut secara bersamaan.
- MntMeatProduct berkorelasi positif yang relatif besar dengan NumCatalogPurchases, sehingga menunjukkan bahwa sebagian besar pembelian produk daging dilakukan melalui katalog.
- MntCoke sebagian besar dibeli melalui katalog dibandingkan metode pembelian lainnya.

**Purchases**
- Kolom NumDealsPurchases berkorelasi positif terhadap NumWebVisitMonth, sehingga menunjukkan bahwa selama periode diskon, jumlah kunjungan pelanggan ke situs web meningkat.
 - Kolom NumWebVisitMonth berkorelasi negatif signifikan dengan NumCatalogPurchases dan NumStorePurchases, sehingga menunjukkan bahwa ketika pelanggan lebih sering mengunjungi situs web, pembelian melalui katalog dan toko akan menurun.

**Additional**
- Pelanggan yang membeli Coke (MntCoke) cenderung membeli Daging (MntMeatProducts) dan lebih memilih membeli langsung dari Toko, Katalog, dan Website.
 - Beberapa kombinasi Buah yang banyak disukai antara lain Buah dan Daging, Buah dan Ikan, atau Buah dan Manis. Pelanggan juga lebih memilih membeli langsung dari Toko atau melalui Katalog dibandingkan menggunakan melalui Website.
- Kombinasi kelima kolom produk menunjukkan nilai korelasi yang relatif tinggi. Oleh karena itu, pelanggan cenderung membeli lebih dari 1 produk dalam sekali belanja.
- Produk Coke dan Gold sebagian besar dibeli melalui Website, sedangkan produk Buah-buahan, Daging, Ikan, dan Manis dibeli melalui Toko dan Katalog.
- Produk yang ditawarkan menggunakan Deals (diskon) tidak terlalu menarik bagi pelanggan karena menunjukkan korelasi yang sangat rendah dengan kolom produk lainnya.
- Pelanggan yang menggunakan Deals cenderung melakukan lebih banyak pembelian melalui Website.

### Korelasi antar variable terhadap variable Response

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150605.png" height="200"/>

- 10 fitur teratas yang memiliki korelasi tertinggi terhadap Respon:
   - AcceptedCmp5 - 0.32 - Positif
   - AcceptedCmp1 - 0.29 - Positif
   - AcceptedCmp3 - 0.25 - Positif
   - MntCoke - 0.24 - Positif
   - MntMeatProducts - 0.23 - Positif
   - NumCatalogPurchases - 0.22 - Positif
   - Recency - 0.19 - Negatif
   - AcceptedCmp4 - 0.17 - Positif
   - AcceptedCmp2 - 0.16 - Positif
   - Teenhome - 0.15 - Negative
- Korelasi antara kolom Respon dengan kolom yang lain cenderung rendah, hanya sekitar 0,00 hingga 0,33.
- Korelasi kolom Response dengan kolom AcceptedCmp5/1/3 paling tinggi dibandingkan kolom campaign sebelumnya.
- Customer lebih tertarik pada produk Coke dan Daging, yang ditunjukkan dari korelasi antara kolom Response dengan kolom MntCoke dan MntMeatProducts.
- Pelanggan juga lebih memilih melakukan pembelian melalui Katalog (kolom NumCatalogPurchases) dibandingkan metode lainnya.
- Kolom Complain, Z_CostContact, dan Z_Revenue tidak memiliki korelasi dengan kolom lainnya.

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150700.png" height="200"/>

Berdasarkan analisis dengan menggunakan visualisasi Boxplot diketahui:

Terdapat beberapa variabel yang menunjukkan perbedaan signifikan dalam hal median antara pelanggan yang merespons dengan yang tidak merespons. Variabel-variabel tersebut yaitu Pendapatan, Kekinian, MntCoke, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProducts, NumberWebPurchases, NumberCatalogPurchases, dan NumberStorePurchases.

- Income, pelanggan yang merespon cenderung memiliki income yang lebih tinggi, dengan rata-rata income sekitar 65M, sedangkan pelanggan yang tidak merespon hanya memiliki rata-rata income sekitar 50M.
- Recency, pelanggan yang merespon cenderung lebih aktif melakukan pembelian, dengan rata-rata jumlah hari yang dihitung sejak pembelian terakhir sekitar 30 hari, sedangkan pelanggan yang tidak merespon hanya memiliki rata-rata lebih dari 50 hari sejak pembelian terakhirnya.
- MntCoke, pelanggan yang merespons cenderung membeli lebih banyak produk Coke, dengan rata-rata pembelian sekitar 450rb selama 2 tahun, dibandingkan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 200rb selama 2 tahun.
- MntFruits, pelanggan yang merespons cenderung membeli lebih banyak produk buah-buahan, dengan rata-rata pembelian di atas 20rb selama 2 tahun, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 5rb selama 2 tahun.
- MntMeatProducts, pelanggan yang merespons cenderung membeli lebih banyak produk Daging, dengan rata-rata pembelian di atas 190 ribu selama 2 tahun, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 50 ribu selama 2 tahun.
- MntFishProducts, pelanggan yang merespons cenderung membeli lebih banyak produk Ikan, dengan rata-rata pembelian sekitar 25rb selama 2 tahun, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 5rb selama 2 tahun.
- MntSweetProducts, pelanggan yang merespons cenderung membeli lebih banyak produk Manis, dengan rata-rata pembelian sekitar 20rb selama 2 tahun, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 5rb selama 2 tahun.
- MntGoldProducts, pelanggan yang merespons cenderung membeli lebih banyak produk Emas, dengan rata-rata pembelian sekitar 40rb selama 2 tahun, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata pembelian kurang dari 20rb selama 2 tahun.
- NumberWebPurchases, pelanggan yang merespons cenderung lebih banyak melakukan pembelian melalui Web, dengan rata-rata 5 kali pembelian, dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata 3 kali pembelian melalui Web.
- NumberCatalogPurchases, pelanggan yang merespons cenderung lebih banyak melakukan pembelian melalui Katalog dengan rata-rata 4 kali pembelian dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata 1 kali pembelian melalui Katalog.
- NumberStorePurchases, pelanggan yang merespons cenderung lebih banyak melakukan pembelian melalui Toko dengan rata-rata 6 kali pembelian dibandingkan dengan pelanggan yang tidak merespons hanya memiliki rata-rata 5 kali pembelian melalui Toko.

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150754.png" height="200"/>

- Education:
S1, S2, dan S3 mempunyai jumlah respon yang tinggi, masing-masing dengan rasio respon > 13,4% (maks 20%).

- Status pernikahan:
Menikah, Lajang, dan Cerai mempunyai jumlah respon yang tinggi, namun rasio response dibandingkan dengan yang tidak merespon adalah < 22.4% (maks 33%).

- Rumah Anak & Rumah Remaja:
Semakin tinggi jumlah children/teenager yang dimiliki pelanggan, maka semakin kecil pula peluang pelanggan merespons kampanye pemasaran (yang terbaru). Dengan demikian, sebaiknya perusahaan menargetkan kepada pelanggan yang tidak memiliki children/teenager. Demikian pula rasio respon dibandingkan dengan yang tidak merespon menurun seiring dengan bertambahnya jumlah children/teenager.

# Feature Engineering

Melakukan penghitungan dan ekstraksi terhadap fitur-fitur berikut:

- Age Customer: Hitung usia dari tiap pelanggan berdasarkan tahun lahirnya kemudian ekstrak sebagai fitur baru.
- Age Group: Kelompokkan pelanggan ke dalam beberapa kategori usia berdasarkan usia mereka.
- Has Child: Membuat fitur baru yang menunjukkan apakah pelanggan memiliki anak atau tidak.
- Dependents: Hitung jumlah tanggungan dalam rumah tangga pelanggan dengan cara menjumlahkan nilai dari kolom Kidhome dan Teenhome.
- Month Customer (Lifetime): Hitung jumlah bulan pelanggan yang sudah bergabung sejak pembelian pertama mereka.
- Spending: Buat fitur baru dengan menjumlahkan total pengeluaran pelanggan dengan menambahkan jumlah yang dibelanjakan untuk berbagai kategori produk.
- Primer & Tersier: Buat fitur baru dengan cara menjumlahkan total pembelian untuk setiap pelanggan dalam kelompok produk primer dan tersier.
- Total of Purchases: Hitung jumlah total pembelian yang dilakukan oleh setiap pelanggan.
- Total_AccCmpgn (Campaign yang Diterima 1-5): Jumlahkan total campaign (1-5) yang diterima oleh tiap pelanggan.
- Ever_Acc (Kampanye yang Diterima minimal 1): Buat fitur biner untuk menunjukkan apakah pelanggan pernah menerima kampanye apa pun (1-5).
- Total Revenue: Hitung total pendapatan yang dihasilkan dari setiap pembelian pelanggan.
- Income Segmentation: Kelompokkan pelanggan ke dalam beberapa segmen pendapatan berdasarkan tingkat pendapatan mereka.
- Conversion Rate Web: Menghitung tingkat konversi untuk pembelian yang dilakukan melalui web.
- Joined Year & Joined Month: Ekstrak bulan dan tahun dimana setiap pelanggan bergabung sebagai pelanggan baru.
- is_Married: Membuat fitur baru untuk menunjukkan apakah pelanggan sudah menikah atau belum.
- Recency Segmentation: Kelompokkan recency ke dalam beberapa segmen berdasarkan total bulan sejak mereka melakukan pembelian terakhir.

# Business Insight

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20150930.png" height="200"/>

<img src="https://github.com/MukhlishHaq/Customer-Personality-Prediction-To-Boost-Marketing-Campaign-Using-Machine-Learning/blob/master/assets/Screenshot%202024-06-30%20151001.png" height="200"/>

Dari beberapa analisis yang telah dilakukan diatas, dapat diidentifikasi beberapa jenis pengguna yang berpotensi lebih tinggi untuk merespons campaign yang dilakukan dengan cara melihat korelasi antara conversion Rate Web terhadap kolom lainnya.

- NumWebVisitsMonth memiliki korelasi negatif yang tinggi dan Total_Purchases memiliki korelasi positif yang tinggi dengan Conversion Rate Web karena Conversion Rate Web berasal dari kedua kolom tersebut.
- NumCatalogPurchases dan NumStorePurchases berkorelasi positif, artinya semakin banyak pelanggan yang melakukan pembelian melalui Catalog/Store maka Conversion Rate Webnya akan semakin tinggi.
- Spending memiliki korelasi positif yang tinggi sebesar 0,63, yang menunjukkan bahwa pelanggan dengan total spending lebih tinggi cenderung akan merespons campaign.
- MntMeatProducts memiliki korelasi positif, menunjukkan bahwa pelanggan yang berhasil memiliki tingkat konversi yang tinggi cenderung melakukan lebih banyak pembelian produk Daging.
- Income memiliki korelasi positif yang tinggi sebesar 0,54, menunjukkan bahwa pelanggan dengan income yang lebih tinggi lebih cenderung akan merespons campaign.
