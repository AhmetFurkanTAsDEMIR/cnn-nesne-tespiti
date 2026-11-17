# Dosya Adı: rapor_olusturucu.py
import os
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

STATE_FILE = "rapor_verileri.json"
REPORT_MD_FILE = "PROJE_RAPORU.md"
REPORT_DOCX_FILE = "PROJE_RAPORU.docx"

DEFAULT_STATE = {
    "total_images": 4000,
    "train_count": 3200,
    "val_count": 400,
    "test_count": 400,
    "train_time_sec": 845.4,
    "mean_iou": 0.3243,
    "test_acc": 0.7025,
    "test_prec": 0.6896,
    "test_rec": 0.7025,
    "test_f1": 0.6767,
    "infer_ms": 8.09,
    "five_photos": [
        {"no": 1, "true": "dog",       "pred": "dog",       "conf": 84.9, "ms": 5140.7, "status": "BAŞARILI"},
        {"no": 2, "true": "bus",       "pred": "bus",       "conf": 42.9, "ms": 4872.0, "status": "BAŞARILI"},
        {"no": 3, "true": "person",    "pred": "person",    "conf": 80.8, "ms": 4816.3, "status": "BAŞARILI"},
        {"no": 4, "true": "motorbike", "pred": "motorbike", "conf": 74.9, "ms": 4823.6, "status": "BAŞARILI"},
        {"no": 5, "true": "car",       "pred": "car",       "conf": 76.1, "ms": 5004.0, "status": "BAŞARILI"},
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
    generate_reports(state)


def add_bullet_with_bold_prefix(doc, prefix, text):
    p = doc.add_paragraph(style="List Bullet")
    run_bold = p.add_run(prefix)
    run_bold.bold = True
    p.add_run(text)


def generate_docx_report(state):
    doc = Document()

    # Yazı tipi ayarları
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Ana Başlık
    title = doc.add_heading("COMPUTER VISION – DERİN ÖĞRENME UYGULAMASI PROJE RAPORU", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    train_min = state["train_time_sec"] / 60.0

    # 1. BÖLÜM
    doc.add_heading("1. Projenin Amacı", level=1)
    doc.add_paragraph(
        "Bu çalışmanın amacı, açık kaynaklı Pascal VOC 2012 veri setini kullanarak çok sınıflı bir "
        "Nesne Tespiti ve Konumlandırma (Object Detection & Localization) problemi için uçtan uca bir "
        "Evrişimli Sinir Ağı (CNN) modeli geliştirmek, modelde kullanılan mimari ve hiperparametre "
        "seçimlerinin gerekçelerini ortaya koymak, modelin başarımını hem sınıflandırma hem de konumlandırma "
        "metrikleriyle değerlendirmek, eğitilen modeli Gradio ile çalışan bir web uygulamasına dönüştürmek "
        "ve cep telefonuyla çekilen gerçek dünya fotoğrafları üzerinde test etmektir."
    )

    # 2. BÖLÜM
    doc.add_heading("2. Görev (Task) ve Seçim Gerekçesi", level=1)
    add_bullet_with_bold_prefix(
        doc,
        "Seçilen Görev: ",
        "Nesne Tespiti ve Konumlandırma (Object Detection – Sınıf Tahmini + Bounding Box Regresyonu)."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden Bu Görev Seçildi? ",
        "Salt görüntü sınıflandırma yalnızca görüntüde ne olduğunu söylerken, nesne tespiti nesnenin görüntü "
        "içerisindeki uzamsal konumunu da (xmin, ymin, xmax, ymax) bulmayı gerektirir. Gerçek hayat "
        "uygulamalarında (otonom sürüş, güvenlik kameraları, akıllı ulaşım) hem nesnenin türünü (araba, insan, "
        "otobüs, motosiklet, kedi, köpek) tanımak hem de konumunu kutu içine almak çok daha işlevsel olduğu için "
        "bu görev tercih edilmiştir."
    )

    # 3. BÖLÜM
    doc.add_heading("3. Model Mimarisi (CNN) ve Seçim Gerekçeleri", level=1)
    doc.add_paragraph(
        "Problemin çözümü için TensorFlow / Keras kullanılarak 5 evrişim bloğuna ve çift çıkışlı (Dual-Head / "
        "Multi-Output) yapıya sahip özel bir CNN mimarisi (model_mimarisi.py) tasarlanmıştır:"
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden CNN (Evrişimli Sinir Ağı) Seçildi? ",
        "Görüntü verilerinde pikseller arasındaki yerel komşuluk ilişkilerini ve uzamsal hiyerarşiyi (ilk "
        "katmanlarda kenar/köşe, orta katmanlarda doku/parça, son katmanlarda bütünsel nesne şekli) ağırlık "
        "paylaşımı sayesinde en verimli öğrenen derin öğrenme mimarisi CNN'dir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden 5 Katmanlı Evrişim Bloğu (32 -> 64 -> 128 -> 256 -> 512 Filtre)? ",
        "Görüntü boyutu MaxPooling2D(2x2) ile her adımda yarıya inerken (224 -> 112 -> 56 -> 28 -> 14), filtre "
        "sayısının iki katına çıkarılması modelin düşük seviyeli kenar bilgilerinden yüksek seviyeli karmaşık "
        "nesne temsillerine geçişini dengeli bir kapasiteyle sağlamaktadır."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden BatchNormalization ve GlobalAveragePooling2D Kullanıldı? ",
        "BatchNormalization katmanlar arası içsel ortak değişinti kaymasını (internal covariate shift) azaltarak "
        "eğitimi hızlandırmak ve stabilize etmek için; GlobalAveragePooling2D ise klasik Flatten katmanına kıyasla "
        "parametre sayısını büyük ölçüde düşürüp aşırı öğrenmeyi (overfitting) engellemek ve uzamsal özetleme "
        "sağlamak için seçilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden Çift Çıkışlı (Dual-Head) Mimari Seçildi? ",
        "Nesne tespiti hem ayrık bir sınıflandırma problemi (nesne hangi sınıfa ait?) hem de sürekli bir regresyon "
        "problemi (kutunun 4 koordinatı nedir?) içerir. Ortak CNN omurgasından çıkan özelliklerin iki ayrı uzmanlaşmış "
        "dala (class_output ve bbox_output) ayrılması, her iki görevin eşzamanlı (multi-task learning) olarak "
        "öğrenilmesini sağlamıştır."
    )

    # 4. BÖLÜM
    doc.add_heading("4. Veri Seti (Dataset) ve Bölünme Oranı Gerekçeleri", level=1)
    add_bullet_with_bold_prefix(
        doc,
        "Seçilen Veri Seti: ",
        "Pascal VOC 2012 (https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)"
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden Pascal VOC 2012 ve Bu 6 Sınıf Seçildi? ",
        "Pascal VOC 2012, bilgisayarlı görü literatüründe nesne tespiti için standartlaşmış, her görüntü için XML "
        "formatında hassas Bounding Box etiketleri barındıran güvenilir bir veri setidir. İçerisinden seçilen 6 sınıf "
        "(car, cat, dog, motorbike, bus, person) günlük yaşamda cep telefonuyla kolayca fotoğraflanıp gerçek dünya "
        "testi yapılabilen nesneler olduğu için özellikle tercih edilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        f"Neden %80 Eğitim ({state['train_count']}) – %10 Doğrulama ({state['val_count']}) – %10 Test ({state['test_count']}) Ayrımı Seçildi? ",
        f"Toplam {state['total_images']} görüntülü veri setinde, sıfırdan eğitilen 5 katmanlı CNN modelinin hem "
        "sınıf ayrımlarını hem de kutu koordinatlarını genelleştirebilmesi için verinin büyük çoğunluğuna (%80 - 3200 "
        "görüntü) eğitimde ihtiyacı vardır. Geriye kalan %20'lik kısım ise sınıf dağılımları korunarak (stratified) "
        "%10 Doğrulama (400 görüntü - eğitim sırasında EarlyStopping ve hiperparametre takibi için) ve %10 Test "
        "(400 görüntü - modelin eğitimde hiç görmediği bağımsız performans ölçümü için) olarak iki eşit parçaya ayrılmıştır."
    )

    # 5. BÖLÜM
    doc.add_heading("5. Model Ayarları, Hiperparametreler ve Seçim Gerekçeleri", level=1)
    doc.add_paragraph(
        "Modelin eğitiminde kullanılan tüm hiperparametreler, aktivasyon/çıkış fonksiyonları, kayıp fonksiyonları ve "
        "optimizasyon algoritması aşağıda seçim nedenleriyle birlikte açıklanmıştır:"
    )
    add_bullet_with_bold_prefix(
        doc,
        "Girdi Boyutu (Input Size = 224 x 224 x 3): ",
        "Çok küçük boyutlar (örn. 64x64) nesne sınırlarının ve küçük detayların kaybolmasına yol açarken, çok büyük "
        "boyutlar (örn. 512x512) bellek ve işlem maliyetini aşırı artırmaktadır. 224x224x3 boyutu, nesne "
        "konumlandırma hassasiyeti ile eğitim/tahmin hızı arasında en ideal dengeyi sağladığı için seçilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Öğrenme Oranı (Learning Rate = 0.001 / 1e-3) ve ReduceLROnPlateau: ",
        "1e-3 başlangıç öğrenme oranı, ağırlıkların başlangıçta hızlı yakınsamasını sağlar. Doğrulama kaybı (val_loss) "
        "2 epoch boyunca iyileşmediğinde öğrenme oranını yarıya indiren ReduceLROnPlateau (factor=0.5, patience=2) "
        "kullanılarak yerel minimumlara daha hassas inilmesi sağlanmıştır."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Yığın Boyutu (Batch Size = 32): ",
        "32'lik yığın boyutu, gradyan tahminlerinin yeterince kararlı olmasını sağlarken aynı zamanda bellek "
        "kullanımını optimize eder ve genelleme başarımını yüksek tutar."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Epoch Sayısı (Epochs = 25) ve EarlyStopping (patience = 5): ",
        "Modelin yeterince öğrenmesi için üst sınır 25 epoch olarak belirlenmiş; ancak aşırı öğrenmeyi (overfitting) "
        "önlemek ve gereksiz zaman kaybını engellemek amacıyla doğrulama kaybı 5 epoch boyunca düşmediğinde eğitimi "
        "durdurup en iyi ağırlıkları geri yükleyen EarlyStopping kullanılmıştır."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Seyreltme (Dropout = 0.4): ",
        "Sınıflandırma başlığındaki tam bağlantılı (Dense 256) katmandan sonra nöronların %40'ını rastgele devre dışı "
        "bırakarak modelin eğitim verisini ezberlemesini (overfitting) önlemek için seçilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Optimizer Seçimi – Neden Adam? ",
        "Klasik SGD tüm parametreler için tek bir öğrenme oranı kullanırken, Adam (Adaptive Moment Estimation) "
        "Momentum (1. moment) ve RMSProp (2. moment) avantajlarını birleştirerek her parametre için adaptif öğrenme "
        "oranı hesaplar. Hem sınıflandırma hem de Bounding Box regresyonunu aynı anda öğrenen çift çıkışlı "
        "mimarilerde SGD'ye kıyasla çok daha hızlı ve kararlı yakınsama sağladığı için Adam tercih edilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Aktivasyon ve Çıkış Fonksiyonu Seçimleri – Neden ReLU, Softmax ve Sigmoid? ",
        "\n  1) Gizli Katmanlarda ReLU: Pozitif bölgede türevi 1 olduğu için kaybolan gradyan (vanishing gradient) "
        "problemini önler ve doğrusal olmayan özellikleri yüksek hızda öğrenir.\n"
        "  2) Sınıf Çıkışında (class_output) Softmax: 6 nesne sınıfı birbirini dışlayan kategoriler olduğu için "
        "çıktıları toplamı 1.0 olan olasılık dağılımına dönüştürmek amacıyla seçilmiştir.\n"
        "  3) Kutu Çıkışında (bbox_output) Sigmoid: Görüntü üzerindeki kutu koordinatları [0, 1] aralığına normalize "
        "edildiği için, çıktıyı doğal olarak [0, 1] aralığında sınırlandıran Sigmoid fonksiyonu tercih edilmiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Loss (Kayıp) Fonksiyonu Seçimleri – Neden Categorical Crossentropy ve MSE (Ağırlık 1.0 ve 10.0)? ",
        "\n  1) Sınıf Tahmini için Categorical Crossentropy: Çok sınıflı one-hot etiketli sınıflandırma "
        "problemlerinde gerçek dağılım ile tahmin edilen olasılık dağılımı arasındaki farkı (KL-divergence) en iyi "
        "cezalandıran kayıp fonksiyonudur.\n"
        "  2) Kutu Koordinatları için MSE (Mean Squared Error): Sürekli [0, 1] değerlerinden oluşan 4 koordinatın "
        "(xmin, ymin, xmax, ymax) regresyonunda büyük sapmaları karesel olarak cezalandırarak kutunun nesneye tam "
        "oturmasını sağlar. MSE değerleri sayısal olarak [0, 1] aralığında çok küçük kaldığı için loss_weights=[1.0, 10.0] "
        "seçilerek iki kayıp fonksiyonunun gradyan büyüklükleri dengelenmiştir."
    )

    if os.path.exists("egitim_grafikleri.png"):
        doc.add_heading("Eğitim ve Doğrulama Grafikleri", level=2)
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.add_run().add_picture("egitim_grafikleri.png", width=Inches(6.0))

    # 6. BÖLÜM
    doc.add_heading("6. Performans Değerlendirmesi (%10 Test Kümesi Sonuçları)", level=1)
    doc.add_paragraph(
        f"Eğitilen model, eğitim ve doğrulama aşamalarında hiç görmediği %10 Test Kümesi ({state['test_count']} görüntü) "
        "üzerinde değerlendirilmiştir. Sınıflandırma başarımı için Accuracy, Precision, Recall ve F1-Score; nesne "
        "konumlandırma başarımı için ise gerçek kutu ile tahmin edilen kutunun kesişim/birleşim oranını ölçen "
        "IoU (Intersection over Union) metriği kullanılmıştır:"
    )
    add_bullet_with_bold_prefix(doc, "Ortalama IoU (Mean IoU): ", f"{state['mean_iou']:.4f}")
    add_bullet_with_bold_prefix(doc, "Test Accuracy (Doğruluk): ", f"{state['test_acc']:.4f} (%{state['test_acc']*100:.2f})")
    add_bullet_with_bold_prefix(doc, "Test Precision (Kesinlik): ", f"{state['test_prec']:.4f}")
    add_bullet_with_bold_prefix(doc, "Test Recall (Duyarlılık): ", f"{state['test_rec']:.4f}")
    add_bullet_with_bold_prefix(doc, "Test F1-Score: ", f"{state['test_f1']:.4f}")
    add_bullet_with_bold_prefix(doc, "Toplam Eğitim Süresi: ", f"{state['train_time_sec']:.2f} saniye ({train_min:.2f} dakika)")
    add_bullet_with_bold_prefix(doc, "Ortalama Tahmin Süresi (Inference Time): ", f"{state['infer_ms']:.2f} ms / görüntü")

    if os.path.exists("test_confusion_matrix.png"):
        doc.add_heading("Test Kümesi Karmaşıklık Matrisi (Confusion Matrix)", level=2)
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.add_run().add_picture("test_confusion_matrix.png", width=Inches(4.8))

    # 7. BÖLÜM
    doc.add_heading("7. Uygulama (Deployment – Gradio) ve Seçim Gerekçesi", level=1)
    add_bullet_with_bold_prefix(
        doc,
        "Seçilen Teknoloji: ",
        "Gradio Web Arayüzü (uygulama_app.py)."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Neden Gradio Seçildi? ",
        "Bilgisayarlı görü ve derin öğrenme modellerini hızlı, interaktif ve sürükle-bırak (drag & drop) destekli bir "
        "web arayüzüne dönüştürmek için en uygun Python kütüphanesidir. Ekstra HTML/JS yazmaya gerek kalmadan hem "
        "tekli nesne tespiti hem de 5'li fotoğraf karşılaştırma modülünü tarayıcı üzerinde doğrudan çalıştırmayı sağlamıştır."
    )

    # 8. BÖLÜM
    doc.add_heading("8. Gerçek Görüntülerle Test ve Değerlendirme (5 Telefon Fotoğrafı)", level=1)
    doc.add_paragraph(
        "Modelin gerçek dünya performansını ölçmek amacıyla cep telefonu ile 5 farklı fotoğraf (köpek, otobüs, "
        "insan, motosiklet ve araba) çekilmiş ve geliştirilen uygulamaya girdi olarak verilmiştir."
    )

    if os.path.exists("telefon_5_fotograf_sonuc.png"):
        doc.add_heading("5 Fotoğrafın Kutulu (Bounding Box) Tahmin Görseli", level=2)
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.add_run().add_picture("telefon_5_fotograf_sonuc.png", width=Inches(6.3))

    doc.add_heading("Gerçek Sonuçlar ve Model Tahminlerinin Karşılaştırılması", level=2)
    t_photos = doc.add_table(rows=1, cols=6)
    t_photos.style = "Table Grid"
    t_photos.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = t_photos.rows[0].cells
    headers = ["Fotoğraf No", "Gerçek Sınıf", "Model Tahmini", "Güven Oranı (%)", "Tahmin Süresi (ms)", "Sonuç"]
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True

    for p in state["five_photos"]:
        row_cells = t_photos.add_row().cells
        row_cells[0].text = str(p["no"])
        row_cells[1].text = p["true"]
        row_cells[2].text = p["pred"]
        row_cells[3].text = f"%{p['conf']:.1f}"
        row_cells[4].text = f"{p['ms']:.1f} ms"
        row_cells[5].text = p["status"]

    doc.add_heading("Modelin Başarılı ve Başarısız Olduğu Durumların Yorumlanması", level=2)
    add_bullet_with_bold_prefix(
        doc,
        "Başarılı Olduğu Durumlar: ",
        "Telefonla çekilen 5 gerçek fotoğrafın tamamında (5/5) model nesne sınıfını doğru tahmin etmiştir "
        "(dog %84.9, person %80.8, car %76.1, motorbike %74.9, bus %42.9). Özellikle motosiklet, araba ve insan "
        "fotoğraflarında olduğu gibi nesnenin kadrajın merkezinde yer aldığı ve sınırlarının belirgin olduğu "
        "görüntülerde hem sınıf tahmini yüksek güven oranıyla yapılmış hem de Bounding Box kutusu nesneyi "
        "başarıyla çerçevelemiştir."
    )
    add_bullet_with_bold_prefix(
        doc,
        "Zorlandığı / Geliştirilebilecek Durumlar: ",
        "Otobüs fotoğrafında olduğu gibi kadrajda birden fazla aracın/otobüsün bulunduğu ve arka planın karmaşık "
        "olduğu sahnelerde model sınıfı doğru bilmesine rağmen güven skoru (%42.9) diğer sınıflara göre daha düşük "
        "kalmıştır. Ayrıca köpek fotoğrafında nesnenin kadrajın sol üst köşesinde yer alması ve zemindeki yoğun çim "
        "dokusu nedeniyle sınıf doğru tahmin edilse de (%84.9) sınır kutusu merkeze doğru kayma göstermiştir. "
        "Çoklu nesne içeren karmaşık sahnelerde YOLO veya Faster R-CNN gibi bölge öneri tabanlı mimariler kutu "
        "hassasiyetini daha da artıracaktır."
    )

    # 9. BÖLÜM
    doc.add_heading("9. Proje Raporu Şablonu", level=1)
    summary_rows = [
        ("Project Title", "TensorFlow CNN Mimarisi ile Çok Sınıflı Nesne Tespiti ve Konumlandırma Uygulaması"),
        ("Problem / Task", "Nesne Tespiti (Object Detection – Sınıf Tahmini + Bounding Box Regresyonu)"),
        ("Dataset", "Pascal VOC 2012 (6 Sınıf: car, cat, dog, motorbike, bus, person – %80 Eğitim / %10 Doğrulama / %10 Test)"),
        ("Dataset URL", "https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012"),
        ("Model Architecture", "5 Katmanlı Özel CNN (Conv2D + BatchNormalization + MaxPooling2D + GlobalAveragePooling2D) + Çift Çıkışlı Başlık (class_output & bbox_output)"),
        ("Input Size", "224 x 224 x 3 (RGB)"),
        ("Hyperparameters", "Split = %80/%10/%10, Learning Rate = 0.001, Batch Size = 32, Epochs = 25, Dropout = 0.4, Loss Weights = [1.0, 10.0]"),
        ("Optimizer", "Adam (Adaptif öğrenme oranı ile sınıflandırma ve regresyonda hızlı yakınsama için)"),
        ("Activation / Output Function", "Gizli Katmanlar: ReLU, Sınıf Çıkışı: Softmax, Bounding Box Çıkışı: Sigmoid"),
        ("Loss Function", "Sınıf için Categorical Crossentropy, Kutu Koordinatları için MSE (Mean Squared Error)"),
        ("Evaluation Metrics", "Mean IoU (Intersection over Union), Accuracy, Precision, Recall, F1-Score, Confusion Matrix"),
        ("Training Environment", "Windows 11, VS Code, Python 3, TensorFlow / Keras"),
        ("Best Performance", f"Test Accuracy: %{state['test_acc']*100:.2f}, Precision: {state['test_prec']:.4f}, Recall: {state['test_rec']:.4f}, F1-Score: {state['test_f1']:.4f}, Mean IoU: {state['mean_iou']:.4f}"),
        ("Inference Time", f"{state['infer_ms']:.2f} ms / görüntü"),
        ("Deployment Technology", "Gradio Web Arayüzü"),
        ("Repository", "https://github.com/AhmetFurkanTAsDEMIR/cnn-nesne-tespiti"),
    ]

    t_summary = doc.add_table(rows=1, cols=2)
    t_summary.style = "Table Grid"
    t_summary.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_summary.rows[0].cells
    hdr[0].text = "Alan"
    hdr[1].text = "Açıklama / Öğrenci Yanıtı"
    for c in hdr:
        for r in c.paragraphs[0].runs:
            r.bold = True

    for field, val in summary_rows:
        row = t_summary.add_row().cells
        row[0].text = field
        row[0].paragraphs[0].runs[0].bold = True
        row[1].text = val

    doc.save(REPORT_DOCX_FILE)
    print(f"[BAŞARILI] Word raporu '{REPORT_DOCX_FILE}' olarak oluşturuldu ve kaydedildi.")


def generate_reports(state=None):
    if state is None:
        state = load_report_state()
    generate_docx_report(state)


if __name__ == "__main__":
    generate_reports()
