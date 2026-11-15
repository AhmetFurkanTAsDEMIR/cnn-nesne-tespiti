# Dosya Adı: egitim.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from veri_hazirlama import create_tf_dataset, prepare_and_save_splits
from model_mimarisi import build_cnn_detector

EPOCHS = 25
MODEL_SAVE_PATH = "cnn_object_detector.keras"


def train_model():
    if not os.path.exists("split_data.npz"):
        prepare_and_save_splits()

    data = np.load("split_data.npz", allow_pickle=True)
    train_ds = create_tf_dataset(data["train_p"], data["train_b"], data["train_l"], is_train=True)
    val_ds = create_tf_dataset(data["val_p"], data["val_b"], data["val_l"], is_train=False)

    model = build_cnn_detector()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("\n[BİLGİ] Model eğitimi başlıyor...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    total_time = time.time() - start_time
    print(f"\n[BAŞARILI] Toplam Eğitim Süresi: {total_time:.2f} saniye ({total_time/60:.2f} dakika)")

    model.save(MODEL_SAVE_PATH)
    print(f"[BAŞARILI] Eğitilen model '{MODEL_SAVE_PATH}' olarak kaydedildi.")

    # Rapor dosyasındaki eğitim süresini otomatik güncelle
    try:
        from rapor_olusturucu import update_report_state
        update_report_state(
            total_images=int(len(data["train_p"]) + len(data["val_p"]) + len(data["test_p"])),
            train_count=int(len(data["train_p"])),
            val_count=int(len(data["val_p"])),
            test_count=int(len(data["test_p"])),
            train_time_sec=float(total_time)
        )
    except Exception as e:
        print("Rapor güncelleme uyarısı:", e)

    # Eğitim Grafiklerini Kaydet
    acc_key = [k for k in history.history.keys() if "acc" in k and not k.startswith("val_")][0]
    val_acc_key = "val_" + acc_key

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history[acc_key], label="Train Accuracy")
    plt.plot(history.history[val_acc_key], label="Validation Accuracy")
    plt.title("Sınıf Doğruluğu (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Toplam Kayıp (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("egitim_grafikleri.png", dpi=150)
    print("[BAŞARILI] Eğitim grafikleri 'egitim_grafikleri.png' olarak kaydedildi.")
    plt.show()


if __name__ == "__main__":
    train_model()