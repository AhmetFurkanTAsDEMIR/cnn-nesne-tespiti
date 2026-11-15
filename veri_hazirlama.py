# Dosya Adı: veri_hazirlama.py
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ==============================================================================
# AYARLAR VE SABİTLER
# ==============================================================================
# İndirdiğin VOC2012 klasörünün bilgisayarındaki yolunu buraya yazabilirsin.
# Eğer VOC2012 klasörünü proje klasörünün içine attıysan otomatik bulacaktır.
VOC_KLASOR_YOLU = r"VOC2012"

TARGET_CLASSES = ["car", "cat", "dog", "motorbike", "bus", "person"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

IMG_SIZE = 224
BATCH_SIZE = 32
MAX_SAMPLES = 4000


def find_voc_directory(custom_path):
    """Bilgisayardaki VOC2012 klasörünü (Annotations ve JPEGImages içeren) bulur."""
    if os.path.exists(os.path.join(custom_path, "Annotations")):
        return custom_path

    # Mevcut klasör ve alt klasörlerde otomatik ara
    candidates = glob.glob("**/Annotations", recursive=True)
    if candidates:
        return os.path.dirname(candidates[0])

    raise FileNotFoundError(
        f"HATA: '{custom_path}' içinde 'Annotations' ve 'JPEGImages' bulunamadı!\n"
        "Lütfen 'veri_hazirlama.py' içindeki VOC_KLASOR_YOLU değişkenine "
        "indirdiğin VOC2012 klasörünün tam yolunu (Örn: r'C:\\Users\\ahmet\\Downloads\\VOC2012') yaz."
    )


def prepare_and_save_splits():
    voc_dir = find_voc_directory(VOC_KLASOR_YOLU)
    annotations_dir = os.path.join(voc_dir, "Annotations")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    print(f"[BİLGİ] Pascal VOC 2012 Klasörü Bulundu: { os.path.abspath(voc_dir) }")

    image_paths, bboxes, labels = [], [], []
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue

        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        if width <= 0 or height <= 0:
            continue

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name in CLASS_TO_IDX:
                bndbox = obj.find("bndbox")
                xmin = max(0.0, min(1.0, float(bndbox.find("xmin").text) / width))
                ymin = max(0.0, min(1.0, float(bndbox.find("ymin").text) / height))
                xmax = max(0.0, min(1.0, float(bndbox.find("xmax").text) / width))
                ymax = max(0.0, min(1.0, float(bndbox.find("ymax").text) / height))

                if xmax > xmin and ymax > ymin:
                    image_paths.append(img_path)
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(CLASS_TO_IDX[cls_name])
                    break

    limit = min(len(image_paths), MAX_SAMPLES)
    image_paths = np.array(image_paths[:limit])
    bboxes = np.array(bboxes[:limit], dtype=np.float32)
    labels_int = np.array(labels[:limit])
    labels_cat = tf.keras.utils.to_categorical(labels_int, num_classes=NUM_CLASSES)

    # 1. Aşama: %80 Eğitim (Train) ve %20 Geçici (Val + Test)
    train_p, temp_p, train_b, temp_b, train_l, temp_l, _, temp_int = train_test_split(
        image_paths, bboxes, labels_cat, labels_int,
        test_size=0.20,
        random_state=42,
        stratify=labels_int
    )

    # 2. Aşama: %20'yi ikiye böl -> %10 Doğrulama (Validation) ve %10 Test
    val_p, test_p, val_b, test_b, val_l, test_l = train_test_split(
        temp_p, temp_b, temp_l,
        test_size=0.50,
        random_state=42,
        stratify=temp_int
    )

    print("\n================ VERİ SETİ DAĞILIMI ================")
    print(f"Toplam Görüntü  : {len(image_paths)} (%100)")
    print(f"Eğitim (Train)  : {len(train_p)} (%{len(train_p)/len(image_paths)*100:.1f})")
    print(f"Doğrulama (Val) : {len(val_p)} (%{len(val_p)/len(image_paths)*100:.1f})")
    print(f"Test Kümesi     : {len(test_p)} (%{len(test_p)/len(image_paths)*100:.1f})")
    print("====================================================\n")

    np.savez(
        "split_data.npz",
        train_p=train_p, train_b=train_b, train_l=train_l,
        val_p=val_p, val_b=val_b, val_l=val_l,
        test_p=test_p, test_b=test_b, test_l=test_l
    )
    print("[BAŞARILI] Veri kümeleri 'split_data.npz' dosyasına kaydedildi.")


def create_tf_dataset(paths, bboxes, labels, is_train=False):
    """Dosya yollarından TensorFlow Dataset oluşturur."""
    def _load(path, bbox, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        return img, (label, bbox)

    ds = tf.data.Dataset.from_tensor_slices((paths, bboxes, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(512)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    prepare_and_save_splits()