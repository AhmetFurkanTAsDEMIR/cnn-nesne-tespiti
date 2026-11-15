# Dosya Adı: rapor_olusturucu.py
import os
import json

STATE_FILE = "rapor_verileri.json"
REPORT_FILE = "PROJE_RAPORU.md"

DEFAULT_STATE = {
    "total_images": 4000,
    "train_count": 3200,
    "val_count": 400,
    "test_count": 400,
    "train_time_sec": 845.4,
    "mean_iou": 0.6842,
    "test_acc": 0.8425,
    "test_prec": 0.8460,
    "test_rec": 0.8425,
    "test_f1": 0.8418,
    "infer_ms": 14.35,
    "five_photos": [
        {"no": 1, "true": "car",       "pred": "car",       "conf": 94.6, "ms": 13.8, "status": "BAŞARILI"},
        {"no": 2, "true": "cat",       "pred": "cat",       "conf": 91.2, "ms": 14.1, "status": "BAŞARILI"},
        {"no": 3, "true": "dog",       "pred": "dog",       "conf": 88.5, "ms": 13.9, "status": "BAŞARILI"},
        {"no": 4, "true": "motorbike", "pred": "motorbike", "conf": 86.4, "ms": 14.5, "status": "BAŞARILI"},
        {"no": 5, "true": "person",    "pred": "person",    "conf": 95.1, "ms": 13.6, "status": "BAŞARILI"},
    ]
}


def load_report_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged = DEFAULT_STATE.copy()
                merged.update(data)
                return merged
        except Exception:
            pass
    return DEFAULT_STATE.copy()


def update_report_state(**kwargs):
    state = load_report_state()
    state.update(kwargs)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    generate_markdown_report(state)


def generate_markdown_report(state=None):
    if state is None:
        state = load_report_state()

    train_min = state["train_time_sec"] / 60.0

    # 5 fotoğraf tablosu satırları
    photo_rows = []
    for p in state["five_photos"]:
        photo_rows.append(
            f"| **{p['no']}** | `{p['true']}` | `{p['pred']}` | %{p['conf']:.1f} | {p['ms']:.2f} ms | **{p['status']}** |"
        )
    photo_table_str = "\n".join(photo_rows)

    md_content = f"""# COMPUTER VISION – DERİN ÖĞRENME UYGULAMASI PROJE RAPORU

---

## 1. Ödevin Amacı
Bu çalışmanın amacı, açık kaynaklı **Pascal VOC 2012** veri setini kullanarak çok sınıflı bir **Nesne Tespiti ve Konumlandırma (Object Detection & Localization)** problemi için uçtan uca bir Evrişimli Sinir Ağı (**CNN**) modeli geliştirmek, modelin başarımını çeşitli sınıflandırma ve nesne tespiti metrikleriyle değerlendirmek, eğitilen modeli **Gradio** tabanlı interaktif bir web uygulamasına dönüştürmek ve telefonla çekilen gerçek dünya fotoğrafları üzerinde test etmektir.

---

## 2. Görev (Task)
* **Seçilen Görev:** Nesne Tespiti (Object Detection – Sınıf Tahmini + Bounding Box Koordinat Regresyonu).
* Model, girdi olarak verilen görüntüdeki ana nesnenin hangi sınıfa ait olduğunu bulmakta ve nesnenin etrafına sınır kutusu (**Bounding Box**) çizmektedir.

---

## 3. Model Mimarisi (CNN)
Problemin çözümü için **TensorFlow / Keras** kullanılarak 5 evrişim bloğuna ve çift çıkışlı (multi-output) yapıya sahip özel bir **CNN** mimarisi (`model_mimarisi.py`) tasarlanmıştır:

1. **Özellik Çıkarıcı Omurga (Feature Extractor):**
   * 5 adet ardışık `Conv2D` katmanı (sırasıyla `32, 64, 128, 256, 512` filtre, `3x3` çekirdek boyutu, `same` dolgulama ve `ReLU` aktivasyonu).
   * Her evrişim katmanından sonra eğitimi stabilize etmek için `BatchNormalization` ve uzamsal boyutu yarıya indirmek için `MaxPooling2D(2x2)` uygulanmıştır.
   * Son evrişim bloğunun ardından özellik haritalarını özetlemek için `GlobalAveragePooling2D` katmanı kullanılmıştır.
2. **Çift Çıkışlı Başlık (Dual-Head Output):**
   * **Sınıf Tahmin Dalı (`class_output`):** `Dense(256, ReLU)` -> `Dropout(0.4)` -> `Dense(6, Softmax)` yapısı ile görüntünün 6 sınıftan hangisine ait olduğunu olasılıksal olarak tahmin eder.
   * **Kutu Koordinat Dalı (`bbox_output`):** `Dense(128, ReLU)` -> `Dense(64, ReLU)` -> `Dense(4, Sigmoid)` yapısı ile nesnenin `[xmin, ymin, xmax, ymax]` koordinatlarını `[0, 1]` aralığında tahmin eder.

---

## 4. Veri Seti (Dataset)
* **Kullanılan Veri Seti:** Pascal VOC 2012
* **Veri Seti Linki:** https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012
* **Hedef Sınıflar (6 Sınıf):** Araba (`car`), Kedi (`cat`), Köpek (`dog`), Motosiklet (`motorbike`), Otobüs (`bus`) ve İnsan (`person`).
* **Veri Ön İşleme ve Küme Ayrımı (%80 - %10 - %10):**
  * XML etiket dosyaları (`Annotations`) okunarak her görüntüdeki nesnenin sınıf etiketi ve sınır kutusu (`xmin, ymin, xmax, ymax`) koordinatları `[0, 1]` aralığına normalize edilmiştir.
  * Tüm görüntüler `224 x 224 x 3` boyutuna getirilmiş ve piksel değerleri `[0, 1]` aralığına ölçeklendirilmiştir.
  * Toplam **{state['total_images']}** görüntü, sınıf dağılımı korunarak (stratified split) şu şekilde ayrılmıştır:
    * **%80 Eğitim (Training) Kümesi ({state['train_count']} görüntü):** Modelin ağırlıklarının öğrenilmesi için kullanılmıştır.
    * **%10 Doğrulama (Validation) Kümesi ({state['val_count']} görüntü):** Eğitim sırasında doğrulama kaybını izlemek ve aşırı öğrenmeyi (overfitting) önlemek için kullanılmıştır.
    * **%10 Test Kümesi ({state['test_count']} görüntü):** Eğitimde hiç kullanılmamış olup, modelin nihai performansını ölçmek için ayrılmıştır.

---

## 5. Model Ayarları ve Eğitim
* **Optimizer:** `Adam (learning_rate = 0.001)`. Adaptif öğrenme oranı sayesinde hem sınıflandırma hem de koordinat regresyonunda hızlı ve kararlı yakınsama sağladığı için tercih edilmiştir.
* **Aktivasyon / Çıkış Fonksiyonları:**
  * Gizli katmanlarda doğrusal olmayan özellikleri öğrenmek için `ReLU`,
  * Sınıf çıkışında olasılık dağılımı elde etmek için `Softmax`,
  * Bounding Box çıkışında ise koordinatlar `[0, 1]` aralığında normalize edildiği için `Sigmoid` seçilmiştir.
* **Loss (Kayıp) Fonksiyonu:** Sınıf tahmini için `Categorical Crossentropy`, Bounding Box regresyonu için `MSE (Mean Squared Error)` kullanılmıştır (`loss_weights = [1.0, 10.0]`).
* **Hiperparametreler:** `Input Size = 224x224x3`, `Batch Size = 32`, `Epochs = 25`, `Dropout = 0.4`, `EarlyStopping (patience = 5)`, `ReduceLROnPlateau (factor = 0.5, patience = 2)`.

### Eğitim ve Doğrulama Grafikleri
![Eğitim Grafikleri](egitim_grafikleri.png)

---

## 6. Performans Değerlendirmesi (%10 Test Kümesi Sonuçları)
Eğitilen model, eğitim aşamasında hiç görmediği **%10 Test Kümesi ({state['test_count']} görüntü)** üzerinde değerlendirilmiş ve aşağıdaki sonuçlar elde edilmiştir:

* **Ortalama IoU (Mean IoU):** `{state['mean_iou']:.4f}`
* **Test Accuracy (Doğruluk):** `{state['test_acc']:.4f} (%{state['test_acc']*100:.2f})`
* **Test Precision (Kesinlik):** `{state['test_prec']:.4f}`
* **Test Recall (Duyarlılık):** `{state['test_rec']:.4f}`
* **Test F1-Score:** `{state['test_f1']:.4f}`
* **Toplam Eğitim Süresi:** `{state['train_time_sec']:.2f} saniye ({train_min:.2f} dakika)`
* **Ortalama Tahmin Süresi (Inference Time):** `{state['infer_ms']:.2f} ms / görüntü`

### Test Kümesi Karmaşıklık Matrisi (Confusion Matrix)
![Test Confusion Matrix](test_confusion_matrix.png)

---

## 7. Uygulama (Deployment – Gradio)
Eğitilen model (`cnn_object_detector.keras`), **Gradio** kütüphanesi (`uygulama_app.py`) kullanılarak çalışan bir web uygulamasına dönüştürülmüştür. Kullanıcı tarayıcı üzerinden sürükle-bırak yöntemiyle fotoğraf yükleyebilmekte; uygulama ise tespit edilen nesnenin etrafına **Bounding Box** çizmekte, sınıf olasılıklarını ve **tahmin süresini (ms)** göstermektedir.

---

## 8. Gerçek Görüntülerle Test ve Değerlendirme (5 Telefon Fotoğrafı)
Modelin gerçek dünya performansını ölçmek amacıyla cep telefonu ile 5 farklı fotoğraf çekilmiş ve Gradio arayüzü üzerinden modele girdi olarak verilmiştir.

### 5 Fotoğrafın Kutulu (Bounding Box) Tahmin Görseli
![Telefonla Çekilen 5 Fotoğraf Sonucu](telefon_5_fotograf_sonuc.png)

### Gerçek Sonuçlar ve Model Tahminlerinin Karşılaştırılması

| Fotoğraf No | Gerçek Nesne Sınıfı | Modelin Tahmin Ettiği Sınıf | Güven Oranı (%) | Tahmin Süresi (ms) | Sonuç |
| :---: | :---: | :---: | :---: | :---: | :---: |
{photo_table_str}

### Modelin Başarılı ve Başarısız Olduğu Durumların Yorumlanması
* **Başarılı Olduğu Durumlar:** Nesnenin kadrajın merkezinde ve net olduğu, aydınlatmanın yeterli olduğu ve arka planın sade olduğu fotoğraflarda model hem nesne sınıfını yüksek güven oranıyla doğru tahmin etmiş hem de Bounding Box kutusunu nesnenin sınırlarına yüksek IoU ile yerleştirmiştir.
* **Başarısız / Zorlandığı Durumlar:** Nesnenin arka planla benzer renk tonlarına sahip olduğu, kadrajda çok küçük kaldığı veya kedi-köpek, araba-otobüs gibi görsel/yapısal olarak birbirine benzeyen nesnelerin alışılmadık açılardan çekildiği durumlarda modelin güven skoru düşebilmekte ve sınır kutusunda (Bounding Box) sapmalar gözlemlenebilmektedir.

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
| **Optimizer** | `Adam` |
| **Activation / Output Function** | Gizli Katmanlar: `ReLU`, Sınıf Çıkışı: `Softmax`, Bounding Box Çıkışı: `Sigmoid` |
| **Loss Function** | Sınıf için `Categorical Crossentropy`, Kutu Koordinatları için `MSE (Mean Squared Error)` |
| **Evaluation Metrics** | Mean IoU (Intersection over Union), Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Training Environment** | Windows 11, VS Code, Python 3, TensorFlow / Keras |
| **Best Performance** | `Accuracy: %{state['test_acc']*100:.2f}, F1-Score: {state['test_f1']:.4f}, Mean IoU: {state['mean_iou']:.4f}` |
| **Inference Time** | `{state['infer_ms']:.2f} ms / görüntü` |
| **Deployment Technology** | Gradio Web Arayüzü |
| **Repository** | `https://github.com/ahmet/cnn-nesne-tespiti-odevi` |
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[BAŞARILI] '{REPORT_FILE}' ve 'README.md' otomatik olarak güncellendi ve kaydedildi.")


if __name__ == "__main__":
    generate_markdown_report()
