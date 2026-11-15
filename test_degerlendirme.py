# Dosya Adı: test_degerlendirme.py
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from veri_hazirlama import create_tf_dataset, TARGET_CLASSES
from egitim import MODEL_SAVE_PATH


def calculate_iou(boxA, boxB):
    """Gerçek kutu ile tahmin edilen kutu arasındaki IoU değerini hesaplar."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate_on_test_set():
    data = np.load("split_data.npz", allow_pickle=True)
    test_p, test_b, test_l = data["test_p"], data["test_b"], data["test_l"]
    test_ds = create_tf_dataset(test_p, test_b, test_l, is_train=False)

    print(f"[BİLGİ] '{MODEL_SAVE_PATH}' yükleniyor...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    y_true_cls, y_pred_cls, iou_scores = [], [], []
    start_infer = time.time()

    for imgs, (true_cls_batch, true_box_batch) in test_ds:
        pred_cls, pred_boxes = model.predict(imgs, verbose=0)

        y_true_cls.extend(np.argmax(true_cls_batch.numpy(), axis=1))
        y_pred_cls.extend(np.argmax(pred_cls, axis=1))

        for tb, pb in zip(true_box_batch.numpy(), pred_boxes):
            iou_scores.append(calculate_iou(tb, pb))

    infer_time_ms = ((time.time() - start_infer) / len(test_p)) * 1000
    mean_iou = np.mean(iou_scores)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="weighted", zero_division=0)

    print("\n========= %10 TEST KÜMESİ PERFORMANS METRİKLERİ =========")
    print(f"Test Görüntü Sayısı     : {len(test_p)}")
    print(f"Ortalama IoU (Mean IoU) : {mean_iou:.4f}")
    print(f"Test Accuracy           : {acc:.4f}")
    print(f"Test Precision          : {prec:.4f}")
    print(f"Test Recall             : {rec:.4f}")
    print(f"Test F1-Score           : {f1:.4f}")
    print(f"Tahmin Süresi (Görüntü) : {infer_time_ms:.2f} ms")
    print("=========================================================\n")
    print(classification_report(y_true_cls, y_pred_cls, target_names=TARGET_CLASSES, zero_division=0))

    # PROJE_RAPORU.md dosyasını test sonuçlarıyla otomatik güncelle
    try:
        from rapor_olusturucu import update_report_state
        update_report_state(
            test_count=int(len(test_p)),
            mean_iou=float(mean_iou),
            test_acc=float(acc),
            test_prec=float(prec),
            test_rec=float(rec),
            test_f1=float(f1),
            infer_ms=float(infer_time_ms)
        )
    except Exception as e:
        print("Rapor güncelleme uyarısı:", e)

    # Confusion Matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title("Test Kümesi (%10) Confusion Matrix")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png", dpi=150)
    print("[BAŞARILI] Confusion matrix 'test_confusion_matrix.png' olarak kaydedildi.")
    plt.show()


if __name__ == "__main__":
    evaluate_on_test_set()