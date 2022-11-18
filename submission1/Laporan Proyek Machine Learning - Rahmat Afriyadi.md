# Laporan Proyek Machine Learning - Rahmat Afriyadi

## Domain Proyek
Berkembangnya teknologi berperan besar dalam kehidupan sehari hari, termasuk contohnya penggunaan *Machine Learning* untuk membantu manusia dalam menyelesaikan permasalahan yang mempunyai komputasi rumit. Dalam penggunaan *Machine Learning* kali ini kita akan melakukan prediksi harga rumah di Negara USA.

Rumah merupakan kebutuhan yang diperlukan bagi manusia sebagai tempat tinggal. Dalam kebutuhan membeli rumah, beberapa aspek dapat menjadi pertimbangan untuk menentukan harga jual rumah. Dengan menggunakan teknologi untuk memprediksi harga rumah, orang dapat menghitung korelasi berbagai aspek rumah, yang dapat memberikan informasi tentang harga rumah berdasarkan keadaan.

Berdasarkan dataset, data latih Model *Machine Learning* yang mampu prediksi harga rumah di Negara USA. Penulis akan menyelesaikan permasalahan prediksi harga rumah dengan 3 model yaitu *KNN, Random Forest* dan *Boost* yang nantinya akan menghasiljkan prediksi harga rumah dengan akurasi tinggi berdasrkan data yang sudah di bagi menjadi data latih dan data uji.
## Business Understanding

### Problem Statements
- Bagaimana cara membuat model untuk prediksi harga rumah di Negara USA dengan akurasi tertinggi
- Bagaimana memilih fitur apa saja yang memiliki korelasi terbaik

### Goals
- Mengetahui model yang mempunyai akurasi tinggi untuk prediksi harga rumah di Negara USA

    ### Solution statements
    - Melakukan proses *Exploratory Data Analysis* untuk melihat data yang berkolerasi dan memiliki pengaruh terhadap harga rumah.
    - Menggunakan model *Machine Learning* regresi. Untuk menemukan hasil prediksi harga rumah yang terbaik. Berikut model-model yang akan digunakan:
    - *Random Forest Regressor*
    - *K-Neighbors Regressor*
    - *AdaBoost Regressor*

## Data Understanding
Dataset yang digunakan untuk prediksi harga rumah diambil dari platfrom kaggle.com yang di publikasikan oleh GANTALA SWETHA. Dataset ini terdiri dari 1 file csv dan berisi data harga rumah yang berada di Negara USA. Berikut akses datasetnya https://www.kaggle.com/code/gantalaswetha/usa-housing-dataset-linear-regression/data. Dataset ini memiliki 5000 data dan 6 kolom, dengan penejelasan kolumnya sebagai berikut:
- Rows : nomor data.
- Address : alamat rumah.
- Avg. Area Income : Rata Rata Pendapatan Daerah ($).
- Price	 : harga rumah.
- Avg. Area House Age : Rata-rata usia rumah di area tersebut.
- Avg. Area Number of Rooms : Rata-rata jumlah ruangan di area tersebut.
- Avg. Area Number of Bedrooms : Rata-rata jumlah kamar tidur di area tersebut.
- Area Population : Jumlah populusi di area tersebut

Tahap selanjutmnya melakukan *Exploratory Data Analysis* (EDA) yang bertujuan untuk menghilangkan outliers, serta menampilkan korelasi antar data baik data kategorikal maupun data numerik.

Berikut merupakan visualisasi boxplot dari data numerik dari area_income, house_age, numberof_room,numberof_bedrooms dan population.

![Visualisasi BoxPlot Area Income (JT)](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/are_income.png)

![Visualisasi BoxPlot House Age Area](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/house_age.png)

![Visualisasi BoxPlot Number of Bedrooms Lon](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/number_of_bedrooms.png)

![Visualisasi BoxPlot Number of Room Lat](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/number_of_rooms.png)

![Visualisasi BoxPlot Population Lat](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/population.png)

Bisa kita lihat dari ke empat gambar bahwa semua fitur memiliki outliers. Oleh karena itu digunakan lah metode *Interquartile Range* (IQR) untuk mengatasi outliers. Yang hasilnya data tersebut akan direduksi dan dieleminasi untuk mengatasi outliers.

Proses selanjutnya melakukan *univariate analysis* untuk data kategorikal dan data numerik. 

![Visualisasi Data Kategorikal Room](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/count_room.png)

Dari gasdmbar diatas dapat disimpulkan bahwa pada fitur jumlah ruangan, data terbanyak yaitu total 6 ruangan dalam satu rumah dan paling sedikit 8 ruangan dalam satu rumah.

![Visualisasi Data Kategorikal district](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/count_bedroom.png)

Dari gambar diatas dapat disimpulkan bahwa pada fitur number_of_bedroom, data terbanyak yaitu pada total 3 kamar tidur dalam satu rumah dan data paling sedikit yaitu total 5 kamar tidur .

Visualisasi data Price dilakukan dengan menggunakan plot histogram.
Sebelum melakukan ploting fitur price terlebih dahulu aku bagi dengan 10_000 agar datanya mudah dilihat

![Visualisasi Data Price](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/price.png)

Dari gambar diatas, dapat ditarik kesimpulan, yaitu:
- Pada data "Price", data harga rumah kebanyakan terdapat direntang 1.000.000$ hingga 1.500.000$
- Distribusi data fokus ditengah artinya nilai rata rata harga sangat berpengaruh.

Tahap selanjutnya proses *multivariate analysis* untuk data kategorikal dan data numerik.

![Visualisasi Data Kategorikal](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/price_with_room.png)

Dari data diatas, dapat disimpulkan:
- Data pada Room (Jumlah Ruangan), jumlah Ruangan 8 memiliki nilai yang tinggi, sehingga dapat disimpulkan fitur number_room memiliki pengaruh dampak yang tinggi terhadap rata-rata harga.
- Data pada Room (Jumlah Kamar Tidur), jumlah kamar tidur 5 memiliki nilai yang tinggi, sehingga dapat disimpulkan fitur number_of_bedrooms memiliki pengaruh dampak yang tinggi terhadap rata-rata harga.

Pada data numerik, digunakan heatmap yang bertujuan untuk memvisualisasikan korelasi antara fitur "Area Income", "House Age", "Number of Room", "Number of Bedroom" dan "Population" dengan data "Price" agar lebih mudah untuk dilihat dan dipahami.

![Visualisasi Data Matrix](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/heatmap.png)

## Data Preparation
- Mengatasi outliers dengan menggunakan metode *Interquartile Range* (IQR) yang akan berdampak pada pengurangan data pada dataset.
- Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 4086 dibagi dengan ratio(80 Train, 20 Test) menjadi 3268 untuk data Train dan 818 untuk data Test.
- Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.MinMaxScaler
- Melakukan standarisasi untuk fitur numerik dengan melakukan kalkualasi pada setiap nilai dengan rumus x - x-min / x-max - x-min.


## Modeling

- Berikut penjelasan beberapa algoritma yang membantu dalam pembuatan model *Machine Learning*, dimana algoritma yang diambil merupakan algoritma bertipe regresi.
    - **Random Forest**, merupakan salah satu algoritma populer yang digunakan karena kesederhanaannya dan memiliki stabilitas yang baik. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
     `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
     `max_depth` = Kedalaman maksimum setiap tree.
     `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.
    - **AdaBoost**, merupakan singkatan dari Adaptive Boosting. Algoritma ini bertujuan untuk memberikan bobot lebih pada observasi yang tidak tepat atau disebut weak classification. Proyek ini menggunakan [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
     `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
     `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
     `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.
    - **K-Neighbors Regressor**, K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.. 

## Model Development

HyperParameter Tuning adalah kegiatan untuk menentukan parameter apa yang terbaik untuk setiap model machine learning. Pada project ini saya menggunakan Grid Search
***Grid search*** adalah proses memindai data untuk melakukan konfigurasi parameter optimal pada model. Ia bekerja dengan menentukan ruang pencarian sebagai kisi nilai hyperparameter dan mengevaluasi kombinasi nilai parameter pada kisi. Tujuannya adalah untuk menemukan kombinasi nilai hyperparameter terbaik. Kelebihan grid search adalah ia mampu menemukan kombinasi optimal dari parameter yang disediakan. Kekurangannya adalah proses ini cukup memakan waktu dan mahal secara komputasi.

Untuk melakukan tuning menggunakan metode Grid Search dapat menggunakan library sebagai berikut: 
-   - from sklearn.model_selection import GridSearchCV
    - from sklearn.model_selection import ShuffleSplit

![Visualisasi Table Grid Search ](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/best_params.png)
Dari gambar di atas kita telah menemukan parameter terbaik untuk setiap model yang akan kita gunakan mari kita konfigurasikan params tersebut kepada model kita.

- Berikut merupakan tahapan pembuatan model dengan beberapa algoritma yang berbeda.
    1. Sebelum membuat model, dilakukan dulu pembuatan DataFrame yang akan diisi dengan hasil MSE data train dan data test pada setiap algoritma. 
    2. Selanjutnya, dilakukan pembuatan model Random Forest dengan melakukan import library pada sklearn.ensemble yang mengambil fungsi RandomForestRegressor. Setelah itu membuat model dengan diisikan beberapa parameter seperti n_estimators=100, max_depth=16, dan random_state=55.
    3. Pada algoritma Boosting, melakukan import library sklearn.ensemble yang mengambil fungsi AdaBoostRegressor. Digunakan beberapa parameter seperti n_estimators=100, learning_rate=0.1, dan random_state=11.
    4. Pada tahapan ini, dilakukan import library sklearn.neighbors yang mengambil fungsi *KNeighborsRegressor*. Pada algoritma K-Neighbors Regressor*, digunakan parameter n_neighbors=15.

## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan *mean squared error* (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE

![MSE Formula](https://www.gstatic.com/education/formulas2/472522532/en/mean_squared_error.svg)

MSE	=	mean squared error
n	=	jumlah dataset
Yi	=	nilai sebenarnya
Ŷi	=	nilai prediksi

Berikut merupakan hasil dari MSE yang dilakukan oleh ketiga model *Machine Learning*.

![Visualisasi Data mse](https://raw.githubusercontent.com/Rahmat-Afriyadi/Machine-Learning-Terapan/main/submission1/image/mse_error.png)


Dan berikut merupakan hasil akurasi dari tiga model *Machine Learning*.
| model    | accuracy |
  |----------|----------|
  | K-Nearest Neighbor  | 83.333167 |
  | Boosting            | 69.221075 |
  | Random Forest       | 84.898115 |

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma *Random Forest* memiliki akurasi lebih tinggi dan tingkat *error* lebih kecil dibandingkan algoritma lainnya dalam proyek ini.

Hasil predict

  | y_true    | prediksi_KNN | prediksi_RF | prediksi_Boosting |
  |----------|----------|----------|----------|
| 126.0 | 112.3 | 115.6 | 125.5 |
| 132.0 | 139.1 | 131.4 | 132.1 |
| 107.0 | 126.1 | 111.2 | 97.3 |
| 78.0 | 74.3 | 71.4 | 75.7 |
| 62.0 | 78.6 | 67.6 | 80.8 |
| 134.0 | 120.1 | 120.6 | 138.0 |
| 102.0 | 98.2 | 103.3 | 111.4 |
 
## Kesimpulan
Dapat ditarik kesimpulan dari proyek prediksi harga rumah di kota Amsterdam dengan menggunakan tiga model regresi *Machine Learning*, yaitu bahwa diantara *Random Forest, K-Neighbors Regressor*, dan *AdaBoost*, algoritma *Random Forest* lebih baik dibandingkan yang lainnya. Hal ini dapat dilihat dari nilai Mean Squared Error (MSE) yang dihasilkan lebih kecil dan mempunyai akurasi yang tinggi dibandingkan algoritma yang lainnya.



