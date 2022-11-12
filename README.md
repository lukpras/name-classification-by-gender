# Klasifikasi Nama Berdasarkan Jenis Kelamin Menggunakan Machine Learning - Luki Prasetyo

## Domain Proyek

Proyek ini merupakan sebuah proyek untuk melakukan klasifikasi dari nama berdasarkan jenis kelamin, hal tersebut dilakukan karena untuk membuat prediksi berdasarkan nama seseorang apakah dia termasuk dalam kategori Pria atau Wanita.
<br />
<br />
Karena di era ***data-driven-solution*** atau ***data-driven decision-making***, klasifikasi jenis kelamin dapat digunakan untuk menentukan demografis dari pasar atau kostumer, yang dapat berperan penting dalam membuat instansi atau perusahaan dalam meningkatkan ketepatan penawaran terhadap jasa/produk mereka dan memilih target yang tepat. Sehingga hal klasifikasi dari jenis kelamin atau umur sangatlah penting.
<br />
<br />
Permasalah tersebut dapat diatasi dengan melakukan pembuatan *machine learning* yang dapat melakukan klasifikasi jenis kelamin dari berbagai kategori, menggunakan pendekatan *classification*, dengan berbagai metode klasifikasi seperti K-Means, Fuzzy C-Means, Naive Bayes, Decission Tree, Logistic Regression dan Random Forest, pada proyek kali algoritma klasifikasi yang digunakan adalah **Naive Bayes, Decission Tree, dan Logisitic Regression**.
<br />
<br />
Variabel yang digunakan pada klasifikasi kali ini adalah Nama, dan terdapat referensi yang digunakan dalam pembuatan proyek ini yaitu:
- Jurnal: **Predicting customer’s gender and age depending on mobile phone data**
- Authors: Ibrahim Mousa Al-Zuabi et al.
- *Journal of Big Data*
- Di Terbitkan pada 19 Februari 2019

<br/><br/>

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

<br/><br/>

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
- Variabel dari data set
    - Name: String (meruapakan variabel tentang nama depan seseorang dari berbagai negara)
    - Gender: M/F (*category/string*)
    - Count: Integer (merupakan variabel jumlah dari setiap nama)
    - Probability: Float (merupakan variabel probabilitas dari setiap nama muncul dari total semua nama)

### Visualisasi Data
Melakukan *Exploratory Data Anlysis* menggunakan library pandas, yaitu berupa
- Melihat sturuktur data dengan contoh data sebagai berikut:

|        |   Name | Gender | Count |  Probability |
|-------:|-------:|-------:|------:|-------------:|
| 114150 |      A |      M |     2 | 5.473480e-09 |
| 112246 |      A |      F |     2 | 5.473480e-09 |
| 115618 |  A'Aff |      F |     1 | 2.736740e-09 |
| 133954 | A'Aron |      M |     1 | 2.736740e-09 |
| 115619 | A'Dele |      F |     1 | 2.736740e-09 |

<br /><br />
- Mencari nilai statsitik dari data set

|       |        Count |  Probability |
|------:|-------------:|-------------:|
| count | 1.472690e+05 | 1.472690e+05 |
|  mean | 2.481161e+03 | 6.790295e-06 |
|  std  | 4.645472e+04 | 1.271345e-04 |
|  min  | 1.000000e+00 | 2.736740e-09 |
|  25%  | 5.000000e+00 | 1.368370e-08 |
|  50%  | 1.700000e+01 | 4.652460e-08 |
|  75%  | 1.320000e+02 | 3.612500e-07 |
|  max  | 5.304407e+06 | 1.451679e-02 |

<br /><br />
- Melihat struktur data 

RangeIndex: 147269 entries, 0 to 147268
Data columns (total 4 columns):
| # |      Column |  Non-Null Count | Dtype   |
|--:|------------:|----------------:|---------|
| 0 |        Name | 147269 non-null | object  |
| 1 |      Gender | 147269 non-null | object  |
| 2 |       Count | 147269 non-null | int64   |
| 3 | Probability | 147269 non-null | float64 |

dtypes: float64(1), int64(1), object(2)
memory usage: 4.5+ MB

<br /><br />
Melakukan visualisasi data menggunakan library matplotlib dan seaborn untuk memahami data lebih jauh, dengan visualisasi sebagai berikut:
- Bar Chart untuk banyaknya jenis kelamin berdasrkan jumlah nama di data set.

![Barchart1](https://github.com/lukpras/name-classification-by-gender/blob/main/assets/barchart1.png)

- Pie Chart untuk banyaknya jenis kelamin berdasrkan jumlah nama di data set.

![Piechart1](https://github.com/lukpras/name-classification-by-gender/blob/main/assets/piechart1.png)

- Bar Chart untuk banyaknya jenis kelamin berdasrkan jumlah nama di data set dengan mempertimbangkan variabel jumlah (*Count*) untuk setiap nama.

![Barchart2](https://github.com/lukpras/name-classification-by-gender/blob/main/assets/barchart2.png)

- Pie Chart untuk banyaknya jenis kelamin berdasrkan jumlah nama di data set dengan mempertimbangkan variabel jumlah (*Count*) untuk setiap nama.

![Piechart2](https://github.com/lukpras/name-classification-by-gender/blob/main/assets/piechart2.png)


Berdasarkan dari visual diatas dapat disimpulkan bahwa jika kita tidak menggunakan variabel jumlah (*Count*) untuk mencari perbandingan antara sebaran jenis kelamin di setiap nama, maka terlihat bahwa "Nama untuk Pria paling banyak muncul, atau memiliki variasi yang tinggi bila dibandingkan dengan nama wanita"
<br /><br />
Tetapi jika kita melihat dari jumlah perbandingan antara Pria dan Wanita maka proporsi anatara Pria dan Wanita tidak signifikan berbeda

## Data Preparation
Tahapan yang dilakukan dalam penyelesaian masalah atau pembuatan klasifikasi menggunakan bahasa pemrograman python, dan dilakukan beberapa tahap preparasi data, yaitu sebagai berikut:
- Membuat pivot dengan membuat tabel kategori untuk melihat frekuensi dari jenis kelamin Pria dan Wanita. hal ini dilakukan agar klasifikasi mudah untuk dilakukan
- Mengubah data text menjadi vector agar bisa dilakukan klasifikasi. Hal ini dilakukan untuk memudahkan dalam training, karena data yang tadinya string diubah menjadi bigram blocks dari karakter.
- Memisah data menjadi Train dan Validation. Hal ini dilakukan karena data set belum memiliki Train dan Validation data.
    - Data set dipisahkan karena masih dalam satu set, dan harus kita pisah menggunakan train_test_split di library scikit-learn.
    - Data set dipsah dengan data untuk *training* sebesar 70% dari total data, dan data untuk validasi sebesar 30%. Dengan rincian sebagai berikut:
        - Total Train Data 93737
        - Total Test Data 40173
        - Total Data set 133910
<br/><br/>

## Modeling

Model yang digunakan menggunakan library scikit-learn dengan algoritma sebagai berikut:
- Naive Bayes dengan menggunakan Multinomial Naive Bayes
    - Menggunaakan Model Multinomial Naive Bayes dengan parameter defaults, dan nilai alpha sam dengan 1 (satu).
    - Multinomial Naive Bayes bekerja dengan memiliki asumsi bahwa setiap fitur dibuat dengan distribusi multinomial. Distribusi multinomial mendeskripsikan probabilitas dari jumlah observasi di setiap kategori.
    - Kelebihan Naive Bayes yaitu akan menghasilkan model yang tidak overfit karena tidak mengguanakn fitur yang tidak relevan, akan tetapi hal ini juga menjadi pisau bermata dua karena akan membuat salah dalam klasifikasi dengan contoh ketika kita ingin melakukan prediksi Class Ci, akan tetapi fitur tidak mengenal Class Ci sehingga dapat membuat salah dalam klasifikasi.
- Decission Tree
    - Menggunakan parameter defaults
    - Cara kerja Decission Tree adalah dimana algoritma ini akan membagi sebuah keputusan kedalam 2 atau lebih pilihan, atau juga dapat disebut dari satu node akan membagi ke 2 atau lebih sub-nodes. Decission Tree menggunakan flowchart seperti layaknya struktur pohon untuk memperlihatkan hasil predisiksi dari beberapa pilihan, dimulai dari akar (*root*) dan bercabang ke daun (*leaf*).
    - Kelebihan dari Decission Tree adalah model yang simple dan mudah dipahami oleh orang-orang yang mungkin kurang paham mengenai machine learning. Decission Tree pun sangat cepat, efisien dan dapat bekerja dengan berbagai macam data seperti numerik, kategori, diskrit, dan kontinyu. Akan tetapi ia pun memiliki kekurangan berupa hasil dari training dan validation mudah sekali menimbulkan overfitting karena feature yang tidak penting memiliki kemungkinan ikut dalam pelatihan model.
- Logistic Regression
    - Menggunakan parameter defauls pada awal pembuatan model.
    - Logistic Regression merupakan model klasifikasi berbasis parameter, dimana ia memiliki parameter yang telah ditetapkan, Logistic Regression hampir sama dengan Linear Regression, akan tetapi jika pada Linear Regression ia menarik garis lurus, pada Logistic Regression ia menarik garis dengan bentuk huruf S, atau biasa disebut ***Sigmoid***. Logistic Regression bekerja dengan menghitung probabilitas secara binary (Ya/Tidak).
    - Kelebihan Logistic Regression yaitu model simple seperti Decisiion Tree dan mudah dipahami, dan dapat digunakan untuk multiclass (multinomial regression), dan dapat melakukan klasifikasi untuk nilai yang tidak diketahui dan nilai akurasinya pun cukup besar, akan tetapi Logistic Regression pun memiliki kekurangan yang cukup signifikan yaitu memiliki asumsi bahwa data linear antara independent dan dependent varibale, nyatanya dalam dunia nyata, data tidak selalu linear.

Dalam proses modelling pun melakukan hyperparameters tuning dalam pemilihan parameter dari default model yang terbaik. Dimana pada kasus ini model yang terbaik adalah Logistic Regression.
Dalam melaukan hyperparameters tuning dilakukan dengan menggunakan GridSearchCV yang berada pada library scikit-learn. Parameter yang dipilih untuk melakukan tuning yaitu:
- Penalty, berupa nilai yang melakukan penalty terhadap varibale, sehingga dapat menghasilkan koefisien terhadap variabel mendekati 0  (nol), atau dapat juga disebut dengan regulasi
    - Pemilihan parameters penalty yaitu:
        - l1, nilai absolut terhadap magnitude dari koefisien.
        - l2, jumlah kuadrat magnitude dari koefisien.
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

- Evaluasi dari akurasi
    - Dari hasil tersebut dapat disimpulkan bahwa Logistic Regression sudah cocok untuk persoalan ini, karena kita akan melakukan klasifikasi dari 2 kategori yaitu Pria dan Wanita.
    - Nilai akurasi tidak terlalu tinggi dikarenakan daftar nama memiliki nilai yang ambigu jika jumlah Pria dan Wanita sama, jadi ia diklasifikasikan ke Wanita.
    - Lalu dari hasil prediksi pun ketika kita mencoba dengan nama yang populer dari keempat negara yang berbasis bahasa inggris yaitu AS, Kanada, UK dan Australia, maka hasil prediksi akan menghaslikan kesalahan, karena basis data hanya berasal dari keempat negara tersebut.

Dari hasil akurasi tersebut maka diplih Logistic Regression sebagai algoritma yang digunakan untuk klasifikasi dan dilakukan hyperparameters tuning, dan menghasilkan nilai parameter terbaik yaitu:
- Penalty: l2
- Solver: lbfgs
Parameter ini merupakan default parameter dari Logistic Regression menggunakan scikit-learn, parameter dpat ditambah saat melakukan tuning dengan beberapa parameter lain seperti pada penalty dapat menggunakan nilai 'none' dan 'elasticnet', serta pada parameter solver dapat juga ditambah dengan 'saga' dan 'sag'. Lalu dapat juga ditambah dengan parameter lain seperti C

Refernsi Bacaan:
- [Predicting customer’s gender and age depending on mobile phone data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0180-9)
