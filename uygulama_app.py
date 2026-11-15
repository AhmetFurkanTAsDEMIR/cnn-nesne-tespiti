# Dosya Adı: uygulama_app.py
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import gradio as gr
from veri_hazirlama import IMG_SIZE, TARGET_CLASSES, NUM_CLASSES
from egitim import MODEL_SAVE_PATH

print(f"[BİLGİ] Model yükleniyor: {MODEL_SAVE_PATH}")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)


# ==============================================================================
# SEKME 1 FONKSİYONU: TEKLİ FOTOĞRAF SÜRÜKLE-BIRAK NESNE TESPİTİ (Madde 7)
# ==============================================================================
def single_image_detection(image):
    if image is None:
        return None, {}, "0 ms"

    t0 = time.time()
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    pred_cls, pred_box = model.predict(img_batch, verbose=0)
    elapsed_ms = (time.time() - t0) * 1000

    cls_idx = np.argmax(pred_cls[0])
    label = TARGET_CLASSES[cls_idx]
    conf = pred_cls[0][cls_idx] * 100

    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = pred_box[0]
    x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

    output_img = image.copy()
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), max(3, int(w / 150)))
    cv2.putText(
        output_img, f"{label} ({conf:.1f}%)",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    probabilities = {TARGET_CLASSES[i]: float(pred_cls[0][i]) for i in range(NUM_CLASSES)}
    return output_img, probabilities, f"{elapsed_ms:.2f} ms"


# ==============================================================================
# SEKME 2 FONKSİYONU: TELEFONLA ÇEKİLEN 5 FOTOĞRAFI SÜRÜKLE-BIRAK TEST ET (Madde 8)
# ==============================================================================
def evaluate_5_drag_drop_photos(img1, lbl1, img2, lbl2, img3, lbl3, img4, lbl4, img5, lbl5):
    photos = [(img1, lbl1), (img2, lbl2), (img3, lbl3), (img4, lbl4), (img5, lbl5)]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    report_lines = [
        f"{'No':<4} | {'Gerçek Sınıf':<12} | {'Model Tahmini':<14} | {'Güven (%)':<10} | {'Süre (ms)':<10} | {'Sonuç'}",
        "-" * 75
    ]
    five_photos_state = []

    for idx, (image, true_label) in enumerate(photos):
        ax = axes[idx]
        if image is None:
            ax.axis("off")
            ax.set_title(f"Fotoğraf #{idx+1}\n(Henüz Eklenmedi)", color="gray")
            continue

        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        t0 = time.time()
        pred_cls, pred_box = model.predict(img_batch, verbose=0)
        infer_ms = (time.time() - t0) * 1000

        pred_idx = np.argmax(pred_cls[0])
        pred_label = TARGET_CLASSES[pred_idx]
        conf = pred_cls[0][pred_idx] * 100

        xmin, ymin, xmax, ymax = pred_box[0] * IMG_SIZE
        box_w = max(5, xmax - xmin)
        box_h = max(5, ymax - ymin)

        is_correct = (pred_label.lower() == true_label.lower())
        status = "BAŞARILI" if is_correct else "BAŞARISIZ"
        box_color = "lime" if is_correct else "red"
        title_color = "green" if is_correct else "red"

        report_lines.append(
            f"{idx+1:<4} | {true_label:<12} | {pred_label:<14} | %{conf:<9.1f} | {infer_ms:<10.2f} | {status}"
        )
        five_photos_state.append({
            "no": idx + 1,
            "true": true_label,
            "pred": pred_label,
            "conf": float(conf),
            "ms": float(infer_ms),
            "status": status
        })

        ax.imshow(img_resized)
        rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=3, edgecolor=box_color, facecolor="none")
        ax.add_patch(rect)
        ax.axis("off")
        ax.set_title(
            f"Fotoğraf #{idx+1}\nGerçek: {true_label}\nTahmin: {pred_label} (%{conf:.1f})\nSüre: {infer_ms:.1f} ms",
            color=title_color, fontweight="bold", fontsize=10
        )

    plt.tight_layout()
    save_file = "telefon_5_fotograf_sonuc.png"
    plt.savefig(save_file, dpi=150)
    plt.close(fig)

    # 5 fotoğraf sonuçlarını PROJE_RAPORU.md dosyasına otomatik kaydet
    if five_photos_state:
        try:
            from rapor_olusturucu import update_report_state
            update_report_state(five_photos=five_photos_state)
        except Exception as e:
            print("Rapor güncelleme uyarısı:", e)

    return save_file, "\n".join(report_lines)


# ==============================================================================
# GRADIO ARAYÜZ TASARIMI
# ==============================================================================
with gr.Blocks(title="CNN Nesne Tespiti Uygulaması") as app:
    gr.Markdown("# TensorFlow CNN ile Nesne Tespiti Uygulaması (Pascal VOC 2012)")
    gr.Markdown("Desteklenen Sınıflar: **Araba (`car`), Kedi (`cat`), Köpek (`dog`), Motosiklet (`motorbike`), Otobüs (`bus`), İnsan (`person`)**")

    with gr.Tabs():
        # 1. SEKME: Tekli Test (Madde 7)
        with gr.TabItem("1. Tekli Nesne Tespiti (Madde 7)"):
            with gr.Row():
                in_img = gr.Image(type="numpy", label="Fotoğrafı Buraya Sürükleyip Bırakın")
                out_img = gr.Image(label="Tespit Edilen Nesne ve Kutusu (Bounding Box)")
            with gr.Row():
                out_probs = gr.Label(num_top_classes=6, label="Sınıf Olasılıkları")
                out_time = gr.Textbox(label="Tahmin Süresi (Inference Time)")
            btn_single = gr.Button("Nesneyi Tespit Et", variant="primary")
            btn_single.click(single_image_detection, inputs=in_img, outputs=[out_img, out_probs, out_time])

        # 2. SEKME: 5 Fotoğraf Sürükle-Bırak Karşılaştırma (Madde 8)
        with gr.TabItem("2. Kendi Çektiğim 5 Fotoğrafı Test Et (Madde 8)"):
            gr.Markdown("### Telefonunuzla çektiğiniz 5 fotoğrafı aşağıdaki kutulara sürükleyip bırakın ve gerçek sınıflarını seçin:")
            with gr.Row():
                with gr.Column():
                    img1 = gr.Image(type="numpy", label="1. Fotoğraf (Sürükle-Bırak)")
                    lbl1 = gr.Dropdown(TARGET_CLASSES, value="car", label="1. Gerçek Sınıf")
                with gr.Column():
                    img2 = gr.Image(type="numpy", label="2. Fotoğraf (Sürükle-Bırak)")
                    lbl2 = gr.Dropdown(TARGET_CLASSES, value="cat", label="2. Gerçek Sınıf")
                with gr.Column():
                    img3 = gr.Image(type="numpy", label="3. Fotoğraf (Sürükle-Bırak)")
                    lbl3 = gr.Dropdown(TARGET_CLASSES, value="dog", label="3. Gerçek Sınıf")
                with gr.Column():
                    img4 = gr.Image(type="numpy", label="4. Fotoğraf (Sürükle-Bırak)")
                    lbl4 = gr.Dropdown(TARGET_CLASSES, value="motorbike", label="4. Gerçek Sınıf")
                with gr.Column():
                    img5 = gr.Image(type="numpy", label="5. Fotoğraf (Sürükle-Bırak)")
                    lbl5 = gr.Dropdown(TARGET_CLASSES, value="person", label="5. Gerçek Sınıf")

            btn_five = gr.Button("5 Fotoğrafı Test Et ve Rapor Görselini Oluştur", variant="primary")
            out_5_plot = gr.Image(type="filepath", label="5 Fotoğrafın Kutulu Sonucu ('telefon_5_fotograf_sonuc.png' olarak da kaydedilir)")
            out_5_table = gr.Textbox(lines=8, label="Gerçek vs Tahmin Karşılaştırma Tablosu")

            btn_five.click(
                evaluate_5_drag_drop_photos,
                inputs=[img1, lbl1, img2, lbl2, img3, lbl3, img4, lbl4, img5, lbl5],
                outputs=[out_5_plot, out_5_table]
            )

if __name__ == "__main__":
    app.launch(inbrowser=True)