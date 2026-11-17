# TensorFlow CNN ile Nesne Tespiti ve Konumlandırma (Object Detection)

Bu proje, **TensorFlow / Keras** kullanılarak sıfırdan tasarlanan çift çıkışlı (Dual-Head) bir **Evrişimli Sinir Ağı (CNN)** modeli ile görüntülerdeki nesnelerin sınıfını tahmin eder ve nesnenin etrafına sınır kutusu (**Bounding Box**) çizer.

## Desteklenen Nesne Sınıfları
Model, **Pascal VOC 2012** veri setindeki 6 temel sınıf üzerinde eğitilmiştir:
* `car` (Araba)
* `cat` (Kedi)
* `dog` (Köpek)
* `motorbike` (Motosiklet)
* `bus` (Otobüs)
* `person` (İnsan)

---

## Proje Dosya Yapısı

```text
├── veri_hazirlama.py          # Pascal VOC XML etiketlerini okur ve %80 Train / %10 Val / %10 Test olarak ayırır
├── model_mimarisi.py          # 5 katmanlı çift çıkışlı (Sınıf + Bounding Box) CNN mimarisi
├── egitim.py                  # Modeli eğitir, modeli (.keras) ve eğitim grafiklerini kaydeder
├── test_degerlendirme.py      # %10 Test kümesi üzerinde IoU, Accuracy, Precision, Recall, F1 ve süre ölçer
├── uygulama_app.py            # Gradio tabanlı sürükle-bırak web arayüzü (Tekli ve 5'li fotoğraf testi)
├── PROJE_RAPORU.md            # Detaylı proje raporu (Markdown)
├── PROJE_RAPORU.docx          # Detaylı proje raporu (Word belgesi)
└── requirements.txt           # Gerekli Python kütüphaneleri
```

---

## Kurulum ve Çalıştırma

### 1. Gerekli Kütüphanelerin Kurulumu
```bash
pip install -r requirements.txt
```

### 2. Veri Setinin Hazırlanması (%80 Eğitim - %10 Doğrulama - %10 Test)
[Pascal VOC 2012](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012) veri setini indirip proje dizinine çıkardıktan sonra:
```bash
python veri_hazirlama.py
```

### 3. Modelin Eğitilmesi
```bash
python egitim.py
```

### 4. Test Kümesi Performans Değerlendirmesi
```bash
python test_degerlendirme.py
```

### 5. Web Arayüzünün (Gradio) Başlatılması
```bash
python uygulama_app.py
```
Uygulama çalıştırıldığında tarayıcıda `http://127.0.0.1:7860` adresinde açılır. Kullanıcılar sürükle-bırak yöntemiyle fotoğraf yükleyerek nesne tespiti sonuçlarını ve tahmin süresini görebilir.

---

## Proje Raporu
Projenin detaylı mimari açıklamaları, hiperparametre seçim gerekçeleri, eğitim/test grafikleri ve gerçek dünya fotoğraflarıyla yapılan değerlendirme sonuçları depo içerisindeki **[PROJE_RAPORU.md](PROJE_RAPORU.md)** ve **[PROJE_RAPORU.docx](PROJE_RAPORU.docx)** dosyalarında yer almaktadır.
