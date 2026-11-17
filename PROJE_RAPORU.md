# COMPUTER VISION – DERİN ÖĞRENME UYGULAMASI PROJE RAPORU

---

## 1. Projenin Amacı
Bu çalışmanın amacı, açık kaynaklı **Pascal VOC 2012** veri setini kullanarak çok sınıflı bir **Nesne Tespiti ve Konumlandırma (Object Detection & Localization)** problemi için uçtan uca bir Evrişimli Sinir Ağı (**CNN**) modeli geliştirmek, modelde kullanılan mimari ve hiperparametre seçimlerinin gerekçelerini ortaya koymak, modelin başarımını hem sınıflandırma hem de konumlandırma metrikleriyle değerlendirmek, eğitilen modeli **Gradio** ile çalışan bir web uygulamasına dönüştürmek ve cep telefonuyla çekilen gerçek dünya fotoğrafları üzerinde test etmektir.

---

## 2. Görev (Task) ve Seçim Gerekçesi
* **Seçilen Görev:** Nesne Tespiti ve Konumlandırma (Object Detection – Sınıf Tahmini + Bounding Box Regresyonu).
* **Neden Bu Görev Seçildi?** Salt görüntü sınıflandırma yalnızca görüntüde ne olduğunu söylerken, nesne tespiti nesnenin görüntü içerisindeki uzamsal konumunu da (`xmin, ymin, xmax, ymax`) bulmayı gerektirir. Gerçek hayat uygulamalarında (otonom sürüş, güvenlik kameraları, akıllı ulaşım) hem nesnenin türünü (`araba, insan, otobüs, motosiklet, kedi, köpek`) tanımak hem de konumunu kutu içine almak çok daha işlevsel olduğu için bu görev tercih edilmiştir.

---

## 3. Model Mimarisi (CNN) ve Seçim Gerekçeleri
Problemin çözümü için **TensorFlow / Keras** kullanılarak 5 evrişim bloğuna ve çift çıkışlı (Dual-Head / Multi-Output) yapıya sahip özel bir **CNN** mimarisi (`model_mimarisi.py`) tasarlanmıştır:

* **Neden CNN (Evrişimli Sinir Ağı) Seçildi?** Görüntü verilerinde pikseller arasındaki yerel komşuluk ilişkilerini ve uzamsal hiyerarşiyi (ilk katmanlarda kenar/köşe, orta katmanlarda doku/parça, son katmanlarda bütünsel nesne şekli) ağırlık paylaşımı sayesinde en verimli öğrenen derin öğrenme mimarisi CNN'dir.
* **Neden 5 Katmanlı Evrişim Bloğu (`32 -> 64 -> 128 -> 256 -> 512` Filtre)?** Görüntü boyutu `MaxPooling2D(2x2)` ile her adımda yarıya inerken (`224 -> 112 -> 56 -> 28 -> 14`), filtre sayısının iki katına çıkarılması modelin düşük seviyeli kenar bilgilerinden yüksek seviyeli karmaşık nesne temsillerine geçişini dengeli bir kapasiteyle sağlamaktadır.
* **Neden `BatchNormalization` ve `GlobalAveragePooling2D` Kullanıldı?** `BatchNormalization` katmanlar arası içsel ortak değişinti kaymasını azaltarak eğitimi hızlandırmak ve stabilize etmek için; `GlobalAveragePooling2D` ise klasik `Flatten` katmanına kıyasla parametre sayısını büyük ölçüde düşürüp aşırı öğrenmeyi (overfitting) engellemek ve uzamsal özetleme sağlamak için seçilmiştir.
* **Neden Çift Çıkışlı (Dual-Head) Mimari Seçildi?** Nesne tespiti hem ayrık bir sınıflandırma problemi (nesne hangi sınıfa ait?) hem de sürekli bir regresyon problemi (kutunun 4 koordinatı nedir?) içerir. Ortak CNN omurgasından çıkan özelliklerin iki ayrı uzmanlaşmış dala (`class_output` ve `bbox_output`) ayrılması, her iki görevin eşzamanlı (multi-task learning) olarak öğrenilmesini sağlamıştır.

---

## 4. Veri Seti (Dataset) ve Bölünme Oranı Gerekçeleri
* **Seçilen Veri Seti:** Pascal VOC 2012 (https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)
* **Neden Pascal VOC 2012 ve Bu 6 Sınıf Seçildi?** Pascal VOC 2012, bilgisayarlı görü literatüründe nesne tespiti için standartlaşmış, her görüntü için XML formatında hassas Bounding Box etiketleri barındıran güvenilir bir veri setidir. İçerisinden seçilen 6 sınıf (`car, cat, dog, motorbike, bus, person`) günlük yaşamda cep telefonuyla kolayca fotoğraflanıp gerçek dünya testi yapılabilen nesneler olduğu için özellikle tercih edilmiştir.
* **Neden %80 Eğitim (3200) – %10 Doğrulama (400) – %10 Test (400) Ayrımı Seçildi?** Toplam **4000** görüntülü veri setinde, sıfırdan eğitilen 5 katmanlı CNN modelinin hem sınıf ayrımlarını hem de kutu koordinatlarını genelleştirebilmesi için verinin büyük çoğunluğuna (**%80 - 3200 görüntü**) eğitimde ihtiyacı vardır. Geriye kalan %20'lik kısım ise sınıf dağılımları korunarak (stratified) **%10 Doğrulama (400 görüntü** - eğitim sırasında `EarlyStopping` ve hiperparametre takibi için) ve **%10 Test (400 görüntü** - modelin eğitimde hiç görmediği bağımsız performans ölçümü için) olarak iki eşit parçaya ayrılmıştır.

---

## 5. Model Ayarları, Hiperparametreler ve Seçim Gerekçeleri
Modelin eğitiminde kullanılan tüm hiperparametreler, aktivasyon/çıkış fonksiyonları, kayıp fonksiyonları ve optimizasyon algoritması aşağıda seçim nedenleriyle birlikte açıklanmıştır:

* **Girdi Boyutu (`Input Size = 224 x 224 x 3`):** Çok küçük boyutlar (örn. 64x64) nesne sınırlarının ve küçük detayların kaybolmasına yol açarken, çok büyük boyutlar (örn. 512x512) bellek ve işlem maliyetini aşırı artırmaktadır. `224x224x3` boyutu, nesne konumlandırma hassasiyeti ile eğitim/tahmin hızı arasında en ideal dengeyi sağladığı için seçilmiştir.
* **Öğrenme Oranı (`Learning Rate = 0.001 / 1e-3`) ve `ReduceLROnPlateau`:** `1e-3` başlangıç öğrenme oranı, ağırlıkların başlangıçta hızlı yakınsamasını sağlar. Doğrulama kaybı (`val_loss`) 2 epoch boyunca iyileşmediğinde öğrenme oranını yarıya indiren `ReduceLROnPlateau (factor=0.5, patience=2)` kullanılarak yerel minimumlara daha hassas inilmesi sağlanmıştır.
* **Yığın Boyutu (`Batch Size = 32`):** 32'lik yığın boyutu, gradyan tahminlerinin yeterince kararlı olmasını sağlarken aynı zamanda bellek kullanımını optimize eder ve genelleme başarımını yüksek tutar.
* **Epoch Sayısı (`Epochs = 25`) ve `EarlyStopping (patience = 5)`:** Modelin yeterince öğrenmesi için üst sınır 25 epoch olarak belirlenmiş; ancak aşırı öğrenmeyi (overfitting) önlemek amacıyla doğrulama kaybı 5 epoch boyunca düşmediğinde eğitimi durdurup en iyi ağırlıkları geri yükleyen `EarlyStopping` kullanılmıştır.
* **Seyreltme (`Dropout = 0.4`):** Sınıflandırma başlığındaki tam bağlantılı (`Dense 256`) katmandan sonra nöronların %40'ını rastgele devre dışı bırakarak modelin eğitim verisini ezberlemesini önlemek için seçilmiştir.
* **Optimizer Seçimi – Neden `Adam`?** Klasik `SGD` tüm parametreler için tek bir öğrenme oranı kullanırken, `Adam (Adaptive Moment Estimation)` Momentum ve RMSProp avantajlarını birleştirerek her parametre için adaptif öğrenme oranı hesaplar. Hem sınıflandırma hem de Bounding Box regresyonunu aynı anda öğrenen çift çıkışlı mimarilerde `SGD`'ye kıyasla çok daha hızlı ve kararlı yakınsama sağladığı için `Adam` tercih edilmiştir.
* **Aktivasyon ve Çıkış Fonksiyonu Seçimleri – Neden `ReLU`, `Softmax` ve `Sigmoid`?**
  1. **Gizli Katmanlarda `ReLU`:** Pozitif bölgede türevi 1 olduğu için kaybolan gradyan (vanishing gradient) problemini önler ve doğrusal olmayan özellikleri yüksek hızda öğrenir.
  2. **Sınıf Çıkışında (`class_output`) `Softmax`:** 6 nesne sınıfı birbirini dışlayan kategoriler olduğu için çıktıları toplamı 1.0 olan olasılık dağılımına dönüştürmek amacıyla seçilmiştir.
  3. **Kutu Çıkışında (`bbox_output`) `Sigmoid`:** Görüntü üzerindeki kutu koordinatları `[0, 1]` aralığına normalize edildiği için, çıktıyı doğal olarak `[0, 1]` aralığında sınırlandıran `Sigmoid` fonksiyonu tercih edilmiştir.
* **Loss (Kayıp) Fonksiyonu Seçimleri – Neden `Categorical Crossentropy` ve `MSE` (`loss_weights = [1.0, 10.0]`)?**
  1. **Sınıf Tahmini için `Categorical Crossentropy`:** Çok sınıflı one-hot etiketli sınıflandırma problemlerinde gerçek dağılım ile tahmin edilen olasılık dağılımı arasındaki farkı en iyi cezalandıran kayıp fonksiyonudur.
  2. **Kutu Koordinatları için `MSE (Mean Squared Error)`:** Sürekli `[0, 1]` değerlerinden oluşan 4 koordinatın (`xmin, ymin, xmax, ymax`) regresyonunda büyük sapmaları karesel olarak cezalandırarak kutunun nesneye tam oturmasını sağlar. MSE değerleri sayısal olarak `[0, 1]` aralığında küçük kaldığı için `loss_weights=[1.0, 10.0]` seçilerek iki kayıp fonksiyonunun gradyan büyüklükleri dengelenmiştir.

### Eğitim ve Doğrulama Grafikleri
![Eğitim Grafikleri](egitim_grafikleri.png)

---

## 6. Performans Değerlendirmesi (%10 Test Kümesi Sonuçları)
Eğitilen model, eğitim ve doğrulama aşamalarında hiç görmediği **%10 Test Kümesi (400 görüntü)** üzerinde değerlendirilmiştir:

* **Ortalama IoU (Mean IoU):** `0.3243`
* **Test Accuracy (Doğruluk):** `0.7025 (%70.25)`
* **Test Precision (Kesinlik):** `0.6896`
* **Test Recall (Duyarlılık):** `0.7025`
* **Test F1-Score:** `0.6767`
* **Toplam Eğitim Süresi:** `845.40 saniye (14.09 dakika)`
* **Ortalama Tahmin Süresi (Inference Time):** `8.09 ms / görüntü`

### Test Kümesi Karmaşıklık Matrisi (Confusion Matrix)
![Test Confusion Matrix](test_confusion_matrix.png)

---

## 7. Uygulama (Deployment – Gradio) ve Seçim Gerekçesi
* **Seçilen Teknoloji:** Gradio Web Arayüzü (`uygulama_app.py`).
* **Neden Gradio Seçildi?** Bilgisayarlı görü ve derin öğrenme modellerini hızlı, interaktif ve sürükle-bırak (drag & drop) destekli bir web arayüzüne dönüştürmek için en uygun Python kütüphanesidir. Ekstra HTML/JS yazmaya gerek kalmadan hem tekli nesne tespiti hem de 5'li fotoğraf karşılaştırma modülünü tarayıcı üzerinde doğrudan çalıştırmayı sağlamıştır.

---

## 8. Gerçek Görüntülerle Test ve Değerlendirme (5 Telefon Fotoğrafı)
Modelin gerçek dünya performansını ölçmek amacıyla cep telefonu ile 5 farklı fotoğraf (`dog, bus, person, motorbike, car`) çekilmiş ve geliştirilen uygulamaya girdi olarak verilmiştir.

### 5 Fotoğrafın Kutulu (Bounding Box) Tahmin Görseli
![Telefonla Çekilen 5 Fotoğraf Sonucu](telefon_5_fotograf_sonuc.png)

### Gerçek Sonuçlar ve Model Tahminlerinin Karşılaştırılması

| Fotoğraf No | Gerçek Nesne Sınıfı | Modelin Tahmin Ettiği Sınıf | Güven Oranı (%) | Tahmin Süresi (ms) | Sonuç |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `dog` | `dog` | %84.9 | 5140.7 ms | **BAŞARILI** |
| **2** | `bus` | `bus` | %42.9 | 4872.0 ms | **BAŞARILI** |
| **3** | `person` | `person` | %80.8 | 4816.3 ms | **BAŞARILI** |
| **4** | `motorbike` | `motorbike` | %74.9 | 4823.6 ms | **BAŞARILI** |
| **5** | `car` | `car` | %76.1 | 5004.0 ms | **BAŞARILI** |

### Modelin Başarılı ve Başarısız Olduğu Durumların Yorumlanması
* **Başarılı Olduğu Durumlar:** Telefonla çekilen 5 gerçek fotoğrafın tamamında (5/5) model nesne sınıfını doğru tahmin etmiştir (`dog %84.9, person %80.8, car %76.1, motorbike %74.9, bus %42.9`). Özellikle motosiklet, araba ve insan fotoğraflarında olduğu gibi nesnenin kadrajın merkezinde yer aldığı ve sınırlarının belirgin olduğu görüntülerde hem sınıf tahmini yüksek güven oranıyla yapılmış hem de Bounding Box kutusu nesneyi başarıyla çerçevelemiştir.
* **Zorlandığı / Geliştirilebilecek Durumlar:** Otobüs fotoğrafında olduğu gibi kadrajda birden fazla aracın/otobüsün bulunduğu ve arka planın karmaşık olduğu sahnelerde model sınıfı doğru bilmesine rağmen güven skoru (`%42.9`) diğer sınıflara göre daha düşük kalmıştır. Ayrıca köpek fotoğrafında nesnenin kadrajın sol üst köşesinde yer alması ve zemindeki yoğun çim dokusu nedeniyle sınıf doğru tahmin edilse de (`%84.9`) sınır kutusu merkeze doğru kayma göstermiştir.

---

## 9. Proje Raporu Şablonu

| Alan | Açıklama / Öğrenci Yanıtı |
| :--- | :--- |
| **Project Title** | TensorFlow CNN Mimarisi ile Çok Sınıflı Nesne Tespiti ve Konumlandırma Uygulaması |
| **Problem / Task** | Nesne Tespiti (Object Detection – Sınıf Tahmini + Bounding Box Regresyonu) |
| **Dataset** | Pascal VOC 2012 (6 Sınıf: `car, cat, dog, motorbike, bus, person` – **%80 Eğitim / %10 Doğrulama / %10 Test**) |
| **Dataset URL** | https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012 |
| **Model Architecture** | 5 Katmanlı Özel CNN (`Conv2D + BatchNormalization + MaxPooling2D + GlobalAveragePooling2D`) + Çift Çıkışlı Başlık (`class_output` & `bbox_output`) |
| **Input Size** | `224 x 224 x 3` (RGB) |
| **Hyperparameters** | `Split = %80/%10/%10`, `Learning Rate = 0.001`, `Batch Size = 32`, `Epochs = 25`, `Dropout = 0.4`, `Loss Weights = [1.0, 10.0]` |
| **Optimizer** | `Adam` (Adaptif öğrenme oranı ile sınıflandırma ve regresyonda hızlı yakınsama için) |
| **Activation / Output Function** | Gizli Katmanlar: `ReLU`, Sınıf Çıkışı: `Softmax`, Bounding Box Çıkışı: `Sigmoid` |
| **Loss Function** | Sınıf için `Categorical Crossentropy`, Kutu Koordinatları için `MSE (Mean Squared Error)` |
| **Evaluation Metrics** | Mean IoU (Intersection over Union), Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Training Environment** | Windows 11, VS Code, Python 3, TensorFlow / Keras |
| **Best Performance** | `Test Accuracy: %70.25, Precision: 0.6896, Recall: 0.7025, F1-Score: 0.6767, Mean IoU: 0.3243` |
| **Inference Time** | `8.09 ms / görüntü` |
| **Deployment Technology** | Gradio Web Arayüzü |
| **Repository** | `https://github.com/AhmetFurkanTAsDEMIR/cnn-nesne-tespiti` |
