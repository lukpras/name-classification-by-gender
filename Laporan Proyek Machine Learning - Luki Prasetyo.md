# Klasifikasi Nama Berdasarkan Jenis Kelamin Menggunakan Machine Learning - Luki Prasetyo

## Domain Proyek

Proyek ini merupakan sebuah proyek untuk melakukan klasifikasi dari nama berdasarkan jenis kelamin, hal tersebut dilakukan karena untuk membuat prediksi berdasarkan nama seseorang apakah dia termasuk dalam kategori Pria atau Wanita.
Karena di era data-driven-solution atau data-driven decision-making, klasifikasi jenis kelamin dapat digunakan untuk menentukan demografis dari pasar atau kostumer, yang dapat berperan penting dalam membuat instansi atau perusahaan dalam meningkatkan ketepatan penawaran terhadap jasa/produk mereka dan memilih target yang tepat. Sehingga hal klasifikasi dari jenis kelamin atau umur sangatlah penting,
Permasalah tersebut dapat diatasi dengan melakukan pembuatan machine learning yang dapat melakukan klasifikasi jenis kelamin dari berbagai kategori, menggunakan pendekatan classification, dengan berbagai metode klasifikasi seperti K-Means, Fuzzy C-Means dan Naive Bayes, Decission Tree, Logistic Regression dan Random Forest, pada proyek kali algoritma klasifikasi yang digunakan adalah **Naive Bayes, Decission Tree, dan Logisitic Regression**.
Variabel yang digunakan pada klasifikasi kali ini adalah Nama, dan terdapat referensi yang digunakan dalam pembuatan proyek ini yaitu:
- Jurnal: **Predicting customer’s gender and age depending on mobile phone data**
- Authors: Ibrahim Mousa Al-Zuabi et al.
- Journal of Big Data
- Di Terbitkan pada 19 Februari 2019

Referensi dapat dikases di [Journal](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0180-9)

## Business Understanding

Pada era saat ini dimana data mulai banyak digunakan, mudah didapatkan, dan mudah diakses, Data menjadi salah satu resource dalam menentukan keputusan atau juga dapat disebut sebagai data-driven decision-making, maka dperlukan alat untuk mengolah data tersebut sehingga menghasilkan nilai, salah satunya dengan menggunakan algoritma-algoritma di machine learning.

### Problem Statements

Seperti yang telah disebutkan dalam latar belakang permasalahan, maka terdapat beberapa masalah yang harus dipecahkan, yaitu:
- Bagaimana cara melakukan klasifikasi agar kita dapat memprediksi jenis kelamin hanya dengan nama
- Mencari algoritma yang tepat untuk melakukan klasifikasi nama berdasarkan jenis kelamin.
- Model yang dibuat harus menghasilkan akurasi yang baik dan model dapat digunakan dalam memprediksi jenis kelamin berdasarkan nama.

### Goals

Setelah permasalahan diketahui, maka harus memiliki tujuan dalam menyelesaikan permasalah tersebut, dan permasalah tersebut dapat diselesaikan dengan:
- Menggunakan klasifikasi menggunakan machine learning dengan bahasa pemrograman python.
- Memilih beberapa algoritma untuk klasifikasi, dan memilih algoritma dengan akurasi terbaik. 
- Menentukan parameter terbaik dari algoritma yang terbaik untuk menghasilkan akurasi yang lebih tinggi.


### Solution statements
- Menggunakan algoritma Naive Bayes dan Decission Tree dalam penyelesaian permasalahan yang ada.
- Menggunakan beberapa algoritma dalam penyelesaian klasifikasi untuk mencari nilai terbaik, dengan menggunakan Naive Bayes, Decission Tree dan Logistic Regression.
- Melakukan hyperparameters tuning dalam penyelesaian masalah sehingga menghasilkan akurasi terbaik.

## Data Understanding
Data set yang digunakan merupakan data yang berada pada UCI Machine Learning Repository, dengan nama data set yaitu **"Gender By Name Data Set"**.
Data set tersebut merupakan data yang dibuat oleh Arun Rao dari Berkeley Skydeck University of California, dan di donasikan ke UCI pada 15 Maret 2020, dimana data set ini memiliki 142720 baris data. Data ini didapatkan dari beberapa negara seperti US, UK, Canada, dan Australia.

Informasi data sebagai berikut:
- Kumpulan data ini menggabungkan jumlah untuk nama depan laki-laki dan perempuan dalam periode waktu tertentu, dan kemudian menghitung probabilitas untuk nama yang diberikan
- Kumpulan data sumber berasal dari otoritas pemerintah:
    - AS: Nama Bayi dari Aplikasi Kartu Jaminan Sosial - Data Nasional, 1880 hingga 2019
    - UK: Nama-nama bayi di Buletin Statistik Inggris dan Wales, 2011 hingga 2018
    - Kanada: British Columbia 100 Years of Popularity Baby names, 1918 hingga 2018
    - Australia: Nama Bayi Populer, Departemen Kejaksaan Agung, 1944 hingga 2019

Sumber data dapat diakses pada [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Gender+by+Name).

### Variabel-variabel pada Gender by Name UCI dataset adalah sebagai berikut:
- Name: String
- Gender: M/F (category/string)
- Count: Integer
- Probability: Float

### Visualisasi Data
- Melakukan Exploratory Data Anlysis untuk melihat struktur data dan mencari nilai NULL.
- Melakukan visualisasi data menggunakan library matplotlib dan seaborn untuk memahami data lebih jauh.

## Data Preparation
Tahapan yang dilakukan dalam penyelesaian masalah atau pembuatan klasifikasi menggunakan bahasa pemrograman python, dan dilakukan beberapa tahap preparasi data, yaitu sebagai berikut:
- Membuat pivot dengan membuat tabel kategori untuk melihat frekuensi dari jenis kelamin Pria dan Wanita. hal ini dilakukan agar klasifikasi mudah untuk dilakukan
- Mengubah data text menjadi vector agar bisa dilakukan klasifikasi. Hal ini dilakukan untuk memudahkan dalam training, karena data yang tadinya string diubah menjadi bigram blocks dari karakter.
- Memisah data menjadi Train dan Validation. Hal ini dilakukan karena data set belum memiliki Train dan Validation data.


## Modeling

Model yang digunakan menggunakan library scikit-learn dengan algoritma sebagai berikut:
- Naive Bayes dengan menggunakan Multinomial Naive Bayes
    - Menggunaakan Model Multinomial Naive Bayes dengan parameter defaults, dan nilai alpha sam dengan 1 (satu).
    - Kelebihan Naive Bayes yaitu akan menghasilkan model yang tidak overfit karena tidak mengguanakn fitur yang tidak relevan, akan tetapi hal ini juga menjadi pisau bermata dua karena akan membuat salah dalam klasifikasi dengan contoh ketika kita ingin melakukan prediksi Class Ci, akan tetapi fitur tidak mengenal Class Ci sehingga dapat membuat salah dalam klasifikasi.
- Decission Tree
    - Menggunakan parameter defaults
    - Kelebihan dari Decission Tree adalah model yang simple dan mudah dipahami oleh orang-orang yang mungkin kurang paham mengenai machine learning. Decission Tree pun sangat cepat, efisien dan dapat bekerja dengan berbagai macam data seperti numerik, kategori, diskrit, dan kontinyu. Akan tetapi ia pun memiliki kekurangan berupa hasil dari training dan validation mudah sekali menimbulkan overfitting karena feature yang tidak penting memiliki kemungkinan ikut dalam pelatihan model.
- Logistic Regression
    - Menggunakan parameter defauls pada awal pembuatan model.
    - Kelebihan Logistic Regression yaitu model simple seperti Decisiion Tree dan mudah dipahami, dan dapat digunakan untuk multiclass (multinomial regression), dan dapat melakukan klasifikasi untuk nilai yang tidak diketahui dan nilai akurasinya pun cukup besar, akan tetapi Logistic Regression pun memiliki kekurangan yang cukup signifikan yaitu memiliki asumsi bahwa data linear antara independent dan dependent varibale, nyatanya dalam dunia nyata, data tidak selalu linear.

Dalam proses modelling pun melakukan hyperparameters tuning dalam pemilihan parameter dari default model yang terbaik. Dimana pada kasus ini model yang terbaik adalah Logistic Regression.
Dalam melaukan hyperparameters tuning dilakukan dengan menggunakan GridSearchCV yang berada pada library scikit-learn. Parameter yang dipilih untuk melakukan tuning yaitu:
- Penalty, berupa nilai yang melakukan penalty terhadap varibale, sehingga dapat menghasilkan koefisien terhadap variabel mendekati 0  (nol), atau dapat juga disebut dengan regulasi
    - Pemilihan parameters penalty yaitu:
        - l1, nilai absolut terhadap magnitude dari koefisien.
        - ;2, jumlah kuadrat magnitude dari koefisien.
    - Pemilihan parameter solver yaitu:
        - newton-cg, baik untuk permasalahan multiclass, karena multinomial loss.
        - lbfgs, baik untuk permasalahn multiclass, karena dapat menagnani multinomial loss.
        - liblinear, baik untuk data set yang kecil.


## Evaluation
Pada model klasifikasi yang dibuat, menggunakan metric mean accuracy.
- Mean Accuracy merupakan nilai proporsi dari prediksi yang benar dari total prediksi.
- Rumus matematis Mean Accuracy adalah
    - Accuracy = Prediksi yang Benar / Total Prediksi.
- Hasil Accuracy dari beberapa algoritma yaitu sebagai berikut:
    - Multinomial Naive Bayes, Train Accuracy **72,72%**, Validation Accuracy **72.1%**
    - Logistic Regression, Train Accuracy **79,73%**, Validation Accuracy **79.19%**
    - Decission Tree, Train Accuracy **74,19%**, Validation Accuracy **72.24%**

Dari hasil akurasi tersebut maka diplih Logistic Regression sebagai algoritma yang digunakan untuk klasifikasi dan dilakukan hyperparameters tuning, dan menghasilkan nilai parameter terbaik yaitu:
- Penalty: ;2
- Solver: lbfgs
Parameter ini merupakan default parameter dari Logistic Regression menggunakan scikit-learn, parameter dpat ditambah saat melakukan tuning dengan beberapa parameter lain seperti pada penalty dapat menggunakan nilai 'none' dan 'elasticnet', serta pada parameter solver dapat juga ditambah dengan 'saga' dan 'sag'. Lalu dapat juga ditambah dengan parameter lain seperti C