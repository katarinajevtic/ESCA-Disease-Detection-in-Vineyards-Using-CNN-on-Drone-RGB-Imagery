"""
CNN za detekciju ESCA bolesti na RGB dron snimcima.
- train/validation/test podela je na nivou scena
- samo 'eska' i 'meska' ulaze u klasu bolestan
- RGB slike
- balansira se samo train; validation i test ostaju prirodni
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import sys, json, glob, base64, argparse, random, csv
from io import BytesIO
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


CONFIG = {
    "dataset_json": ["dataset_json"],
    "patch_size": 64,
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "test_fraction": 0.20,
    "val_fraction": 0.15,
    "seed": 42,
    "disease_labels": {"eska", "meska"},
    "healthy_min_green": 0.70,
    "buffer_multiplier": 3.0,
    "step_multiplier": 2.0,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: ne učitava se {path}: {e}")
        return None


def load_rgb(data, json_path):
    """LabelMe imageData -> RGB"""
    image_data = data.get("imageData")
    if image_data:
        try:
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            return np.array(
                Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            )
        except Exception:
            pass

    image_path = data.get("imagePath")
    if image_path:
        candidate = Path(json_path).parent / image_path
        if candidate.exists():
            return np.array(Image.open(candidate).convert("RGB"))
    return None


def short_scene_name(scene_id):
    p = Path(scene_id)
    return f"{p.parent.name}/{p.stem.replace('_D', '')}"


# ================================================
# 1. PATCH PO SCENAMA
# ================================================
def extract_patches_by_scene(dataset_folders, patch_size=64):
    if isinstance(dataset_folders, str):
        dataset_folders = [dataset_folders]

    json_files = []
    for folder in dataset_folders:
        json_files += glob.glob(os.path.join(folder, "**", "*_D.json"), recursive=True)
    json_files = sorted(set(os.path.abspath(x) for x in json_files))

    print(f"Pronađeno RGB JSON scena: {len(json_files)}")
    if not json_files:
        sys.exit("GREŠKA: nema *_D.json fajlova.")

    scene_patches = defaultdict(lambda: {"bolestan": [], "zdrav": []})
    label_counts = Counter()

    for json_path in json_files:
        data = load_json(json_path)
        if data is None:
            continue

        img = load_rgb(data, json_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scene_id = os.path.abspath(json_path)  # jedinstven ID scene
        sick_boxes = []

        # Bolestan: samo eska/meska
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip().lower()
            label_counts[label] += 1
            if label not in CONFIG["disease_labels"]:
                continue

            pts = np.asarray(shape.get("points", []), dtype=np.float32)
            if len(pts) < 3:
                continue

            x1 = max(0, int(np.floor(pts[:, 0].min())))
            x2 = min(w, int(np.ceil(pts[:, 0].max())) + 1)
            y1 = max(0, int(np.floor(pts[:, 1].min())))
            y2 = min(h, int(np.ceil(pts[:, 1].max())) + 1)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patch = cv2.resize(
                patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
            )
            scene_patches[scene_id]["bolestan"].append(patch)
            sick_boxes.append((x1, y1, x2, y2))

        # Ne koristimo scene bez eska/meska anotacija kao izvor "zdravih"
        if not sick_boxes:
            scene_patches.pop(scene_id, None)
            continue

        avg_w = max(1, int(round(np.mean([x2 - x1 for x1, y1, x2, y2 in sick_boxes]))))
        avg_h = max(1, int(round(np.mean([y2 - y1 for x1, y1, x2, y2 in sick_boxes]))))
        buffer = int(round(max(avg_w, avg_h) * CONFIG["buffer_multiplier"]))

        # Zona isključenja oko bolesnih anotacija
        sick_mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in sick_boxes:
            cv2.rectangle(
                sick_mask,
                (max(0, x1 - buffer), max(0, y1 - buffer)),
                (min(w - 1, x2 + buffer), min(h - 1, y2 + buffer)),
                255, -1
            )

        # Zelena vegetacija
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (30, 50, 50), (85, 255, 255))
        healthy_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(sick_mask))

        step = max(1, int(round(max(avg_w, avg_h) * CONFIG["step_multiplier"])))
        for y in range(0, h - avg_h + 1, step):
            for x in range(0, w - avg_w + 1, step):
                region = healthy_mask[y:y + avg_h, x:x + avg_w]
                if region.size == 0:
                    continue

                green_ratio = (region > 0).mean()
                if green_ratio >= CONFIG["healthy_min_green"]:
                    patch = img[y:y + avg_h, x:x + avg_w]
                    patch = cv2.resize(
                        patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
                    )
                    scene_patches[scene_id]["zdrav"].append(patch)

    n_sick = sum(len(v["bolestan"]) for v in scene_patches.values())
    n_healthy = sum(len(v["zdrav"]) for v in scene_patches.values())

    print(f"Upotrebljive scene: {len(scene_patches)}")
    print(f"Bolesni patch-evi: {n_sick}")
    print(f"Zdravi kandidati: {n_healthy}")
    print(f"Labele pronađene u JSON: {dict(label_counts)}")

    if n_sick == 0 or n_healthy == 0:
        raise RuntimeError("Nisu formirane obe klase. Proveri labele/HSV kriterijum.")

    return scene_patches


# ============================================================
# 2. SCENE-WISE TRAIN/VALIDATION/TEST
# ============================================================
def split_scenes(scene_patches):
    scenes = np.array(sorted(scene_patches.keys()), dtype=object)
    rng = np.random.default_rng(CONFIG["seed"])
    rng.shuffle(scenes)

    n = len(scenes)
    n_test = max(1, int(round(n * CONFIG["test_fraction"])))
    n_val = max(1, int(round(n * CONFIG["val_fraction"])))
    if n_test + n_val >= n:
        raise RuntimeError("Nedovoljno scena za train/validation/test podelu.")

    test_scenes = list(scenes[:n_test])
    val_scenes = list(scenes[n_test:n_test + n_val])
    train_scenes = list(scenes[n_test + n_val:])

    assert set(train_scenes).isdisjoint(val_scenes)
    assert set(train_scenes).isdisjoint(test_scenes)
    assert set(val_scenes).isdisjoint(test_scenes)

    return train_scenes, val_scenes, test_scenes


def collect(scene_patches, scenes):
    X, y, scene_ids = [], [], []
    for scene in scenes:
        for patch in scene_patches[scene]["bolestan"]:
            X.append(patch); y.append(0); scene_ids.append(scene)
        for patch in scene_patches[scene]["zdrav"]:
            X.append(patch); y.append(1); scene_ids.append(scene)
    return X, np.asarray(y, dtype=np.int64), np.asarray(scene_ids, dtype=object)


def balance_train(X, y, scene_ids):
    """Undersampling train skupa."""
    sick = np.where(y == 0)[0]
    healthy = np.where(y == 1)[0]
    if len(sick) == 0 or len(healthy) == 0:
        raise RuntimeError("Train skup nema obe klase.")

    n = min(len(sick), len(healthy))
    rng = np.random.default_rng(CONFIG["seed"])
    sick = rng.choice(sick, n, replace=False)
    healthy = rng.choice(healthy, n, replace=False)
    keep = np.concatenate([sick, healthy])
    rng.shuffle(keep)

    return [X[i] for i in keep], y[keep], scene_ids[keep]


def class_counts(y):
    return int((y == 0).sum()), int((y == 1).sum())


def prepare_data(scene_patches):
    train_scenes, val_scenes, test_scenes = split_scenes(scene_patches)

    X_train_raw, y_train_raw, train_ids_raw = collect(scene_patches, train_scenes)
    X_val, y_val, val_ids = collect(scene_patches, val_scenes)
    X_test, y_test, test_ids = collect(scene_patches, test_scenes)

    X_train, y_train, train_ids = balance_train(X_train_raw, y_train_raw, train_ids_raw)

    if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Validation ili test nema obe klase. Promeni seed/proveri scene.")

    print("\nSCENE-WISE PODELA")
    for name, scenes, y in [
        ("TRAIN", train_scenes, y_train),
        ("VALIDATION", val_scenes, y_val),
        ("TEST", test_scenes, y_test),
    ]:
        a, b = class_counts(y)
        print(f"{name:10s}: {len(scenes)} scena | {len(y)} patch-eva | bolestan={a}, zdrav={b}")

    raw_a, raw_b = class_counts(y_train_raw)
    print(f"Train PRE balansiranja: bolestan={raw_a}, zdrav={raw_b}, ukupno={len(y_train_raw)}")

    with open("scene_split.txt", "w", encoding="utf-8") as f:
        for name, scenes in [("TRAIN", train_scenes), ("VALIDATION", val_scenes), ("TEST", test_scenes)]:
            f.write(f"[{name}]\n")
            for s in scenes:
                a = len(scene_patches[s]["bolestan"])
                b = len(scene_patches[s]["zdrav"])
                f.write(f"{short_scene_name(s)} | bolestan={a} | zdrav={b}\n")
            f.write("\n")

    X_train = np.asarray(X_train, dtype=np.float32) / 255.0
    X_val = np.asarray(X_val, dtype=np.float32) / 255.0
    X_test = np.asarray(X_test, dtype=np.float32) / 255.0

    return {
        "X_train": X_train, "y_train": y_train, "train_ids": train_ids,
        "X_val": X_val, "y_val": y_val, "val_ids": val_ids,
        "X_test": X_test, "y_test": y_test, "test_ids": test_ids,
        "train_scenes": train_scenes, "val_scenes": val_scenes, "test_scenes": test_scenes,
    }


# =======================================
# 3. CNN
# =======================================
def create_model(patch_size=64):
    model = Sequential([
        Input(shape=(patch_size, patch_size, 3)),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),

        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),

        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),

        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_architecture(model):
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    try:
        tf.keras.utils.plot_model(
            model, to_file="cnn_arhitektura.png",
            show_shapes=True, show_layer_names=True, dpi=180
        )
    except Exception as e:
        print(f"plot_model nije generisan (pydot/Graphviz): {e}")


# ========================================
# 4. TRENING + TEST EVALUACIJA
# =========================================
def find_best_disease_threshold(y_val, val_probs):
    thresholds = np.arange(0.10, 0.91, 0.01)
    results = []
    p_bolestan = val_probs[:, 0]

    for threshold in thresholds:
        pred = np.where(p_bolestan >= threshold, 0, 1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val,
            pred,
            labels=[0],
            average=None,
            zero_division=0,
        )

        precision = float(precision[0])
        recall = float(recall[0])
        f1 = float(f1[0])
        accuracy = float(accuracy_score(y_val, pred))

        # F2 = (1 + beta^2) * P * R / (beta^2 * P + R), beta=2
        denominator = 4.0 * precision + recall
        f2 = (5.0 * precision * recall / denominator) if denominator > 0 else 0.0

        results.append({
            "threshold": float(threshold),
            "accuracy": accuracy,
            "precision_bolestan": precision,
            "recall_bolestan": recall,
            "f1_bolestan": f1,
            "f2_bolestan": f2,
        })

    best = max(
        results,
        key=lambda x: (
            x["f2_bolestan"],
            x["recall_bolestan"],
            x["precision_bolestan"],
        ),
    )

    with open("threshold_validation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "accuracy",
            "precision_bolestan",
            "recall_bolestan",
            "f1_bolestan",
            "f2_bolestan",
        ])
        for row in results:
            writer.writerow([
                f'{row["threshold"]:.2f}',
                f'{row["accuracy"]:.6f}',
                f'{row["precision_bolestan"]:.6f}',
                f'{row["recall_bolestan"]:.6f}',
                f'{row["f1_bolestan"]:.6f}',
                f'{row["f2_bolestan"]:.6f}',
            ])

    print("\n" + "=" * 65)
    print("IZBOR PRAGA—VALIDATION SKUP")
    print("=" * 65)
    print(f"Najbolji threshold:  {best['threshold']:.2f}")
    print(f"Validation accuracy: {best['accuracy']:.4f}")
    print(f"Precision bolestan:  {best['precision_bolestan']:.4f}")
    print(f"Recall bolestan:     {best['recall_bolestan']:.4f}")
    print(f"F1 bolestan:         {best['f1_bolestan']:.4f}")
    print(f"F2 bolestan:         {best['f2_bolestan']:.4f}")

    return best["threshold"]


def train_and_evaluate(model, d):
    y_train_c = to_categorical(d["y_train"], 2)
    y_val_c = to_categorical(d["y_val"], 2)
    y_test_c = to_categorical(d["y_test"], 2)

    history = model.fit(
        d["X_train"],
        y_train_c,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(d["X_val"], y_val_c),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                "best_model.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ],
        shuffle=True,
        verbose=2,
    )

    best_loss_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_acc_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    # -------------------------------------------
    # VALIDATION
    # --------------------------------------------
    val_loss, val_acc_default = model.evaluate(d["X_val"], y_val_c, verbose=0)
    val_probs = model.predict(d["X_val"], verbose=0)

    best_threshold = find_best_disease_threshold(d["y_val"], val_probs)

    val_pred_tuned = np.where(
        val_probs[:, 0] >= best_threshold,
        0,
        1,
    )
    val_acc_tuned = accuracy_score(d["y_val"], val_pred_tuned)

    # --------------------------------------------------------
    # TEST: poređenje standardnog praga 0.50 i izabranog praga
    # --------------------------------------------------------
    test_loss, _ = model.evaluate(d["X_test"], y_test_c, verbose=0)
    probs = model.predict(d["X_test"], verbose=0)
    yt = d["y_test"]

    yp_default = np.where(probs[:, 0] >= 0.50, 0, 1)
    cm_default = confusion_matrix(yt, yp_default, labels=[0, 1])
    test_acc_default = accuracy_score(yt, yp_default)
    report_default = classification_report(
        yt,
        yp_default,
        target_names=["bolestan", "zdrav"],
        digits=4,
        zero_division=0,
    )

    yp_tuned = np.where(probs[:, 0] >= best_threshold, 0, 1)
    cm_tuned = confusion_matrix(yt, yp_tuned, labels=[0, 1])
    test_acc_tuned = accuracy_score(yt, yp_tuned)
    report_tuned = classification_report(
        yt,
        yp_tuned,
        target_names=["bolestan", "zdrav"],
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print("FINALNA EVALUACIJA")
    print("=" * 65)
    print(
        f"Najmanji val_loss: {min(history.history['val_loss']):.4f} "
        f"| epoha {best_loss_epoch}"
    )
    print(
        f"Najveći val_accuracy: {max(history.history['val_accuracy']):.4f} "
        f"| epoha {best_acc_epoch}"
    )

    print("\n--- DEFAULT THRESHOLD = 0.50 ---")
    print(f"Validation accuracy vraćenog modela: {val_acc_default:.4f}")
    print(f"TEST accuracy: {test_acc_default:.4f}")
    print("\nCLASSIFICATION REPORT — TEST — threshold 0.50")
    print(report_default)
    print("CONFUSION MATRIX—TEST— threshold 0.50")
    print(cm_default)

    print(f"\n--- TUNED THRESHOLD = {best_threshold:.2f} ---")
    print(f"Validation accuracy tuned: {val_acc_tuned:.4f}")
    print(f"TEST accuracy tuned: {test_acc_tuned:.4f}")
    print(f"\nCLASSIFICATION REPORT — TEST — threshold {best_threshold:.2f}")
    print(report_tuned)
    print(f"CONFUSION MATRIX—TEST— threshold {best_threshold:.2f}")
    print(cm_tuned)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write("DEFAULT THRESHOLD = 0.50\n")
        f.write("=" * 60 + "\n")
        f.write(report_default)
        f.write("\nCONFUSION MATRIX\n")
        f.write(np.array2string(cm_default))

        f.write("\n\n")
        f.write(f"TUNED THRESHOLD = {best_threshold:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(report_tuned)
        f.write("\nCONFUSION MATRIX\n")
        f.write(np.array2string(cm_tuned))

    model.save("model_final.keras")

    return {
        "history": history,
        "probs": probs,
        "yt": yt,
        "yp_default": yp_default,
        "yp_tuned": yp_tuned,
        "cm_default": cm_default,
        "cm_tuned": cm_tuned,
        "val_acc_default": float(val_acc_default),
        "val_acc_tuned": float(val_acc_tuned),
        "test_acc_default": float(test_acc_default),
        "test_acc_tuned": float(test_acc_tuned),
        "best_loss_epoch": best_loss_epoch,
        "best_acc_epoch": best_acc_epoch,
        "best_threshold": float(best_threshold),
    }


# ============================================================
# 5. GRAFICI, GREŠKE I PER-SCENE REZULTATI
# ============================================================
def save_plots(history, cm, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Tačnost (Accuracy)")
    axes[0].set_xlabel("Epoha"); axes[0].set_ylabel("Tačnost")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Gubitak (Loss)")
    axes[1].set_xlabel("Epoha"); axes[1].set_ylabel("Gubitak")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["bolestan", "zdrav"],
        yticklabels=["bolestan", "zdrav"], ax=axes[2]
    )
    axes[2].set_title(f"Matrica konfuzije-test (prag={threshold:.2f})")
    axes[2].set_xlabel("Predviđena klasa")
    axes[2].set_ylabel("Stvarna klasa")

    plt.tight_layout()
    plt.savefig("rezultati.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_test_examples(X_test, y_test, n=6):
    rng = np.random.default_rng(CONFIG["seed"])
    sick = np.where(y_test == 0)[0]
    healthy = np.where(y_test == 1)[0]
    sick = rng.choice(sick, min(n, len(sick)), replace=False)
    healthy = rng.choice(healthy, min(n, len(healthy)), replace=False)

    cols = max(len(sick), len(healthy), 1)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    if cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for ax in axes.ravel():
        ax.axis("off")

    for i, idx in enumerate(sick):
        axes[0, i].imshow(X_test[idx])
        axes[0, i].set_title("BOLESTAN", color="red")
    for i, idx in enumerate(healthy):
        axes[1, i].imshow(X_test[idx])
        axes[1, i].set_title("ZDRAV", color="green")

    plt.suptitle("Primeri iz nezavisnog test skupa")
    plt.tight_layout()
    plt.savefig("primeri_test.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_errors(X_test, yt, yp, probs, n=6):
    fn = np.where((yt == 0) & (yp == 1))[0][:n]  # bolestan -> zdrav
    fp = np.where((yt == 1) & (yp == 0))[0][:n]  # zdrav -> bolestan
    cols = max(len(fn), len(fp), 1)

    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.2))
    if cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for ax in axes.ravel():
        ax.axis("off")

    for i, idx in enumerate(fn):
        axes[0, i].imshow(X_test[idx])
        axes[0, i].set_title(f"FN bolestan→zdrav\nP(zdrav)={probs[idx,1]:.2f}", fontsize=9)
    for i, idx in enumerate(fp):
        axes[1, i].imshow(X_test[idx])
        axes[1, i].set_title(f"FP zdrav→bolestan\nP(bolestan)={probs[idx,0]:.2f}", fontsize=9)

    plt.suptitle("Pogrešne klasifikacije na test skupu")
    plt.tight_layout()
    plt.savefig("greske_test.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_scene_results(yt, yp, test_ids):
    rows = []
    for scene in sorted(set(test_ids.tolist())):
        idx = np.where(test_ids == scene)[0]
        ys, ps = yt[idx], yp[idx]
        p, r, f1, _ = precision_recall_fscore_support(
            ys, ps, labels=[0], average=None, zero_division=0
        )
        rows.append([
            short_scene_name(scene), len(idx), int((ys == 0).sum()), int((ys == 1).sum()),
            accuracy_score(ys, ps), p[0], r[0], f1[0]
        ])

    with open("per_scene_rezultati.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scena", "ukupno", "bolestan", "zdrav", "accuracy",
            "precision_bolestan", "recall_bolestan", "f1_bolestan"
        ])
        writer.writerows(rows)


# ================
# MAIN
# ================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", nargs="+", type=str)
    args = parser.parse_args()
    if args.dataset_json:
        CONFIG["dataset_json"] = args.dataset_json

    set_seed(CONFIG["seed"])

    print("\n" + "=" * 65)
    print("DETEKCIJA ESCA BOLESTI-CNN- BOLESTAN vs ZDRAV")
    print("SCENE-WISE TRAIN / VALIDATION / TEST")
    print("=" * 65)

    scene_patches = extract_patches_by_scene(CONFIG["dataset_json"], CONFIG["patch_size"])
    data = prepare_data(scene_patches)

    model = create_model(CONFIG["patch_size"])
    model.summary()
    save_architecture(model)

    result = train_and_evaluate(model, data)

    history = result["history"]
    probs = result["probs"]
    yt = result["yt"]
    yp = result["yp_tuned"]
    cm = result["cm_tuned"]
    best_threshold = result["best_threshold"]

    save_plots(history, cm, best_threshold)
    save_test_examples(data["X_test"], data["y_test"])
    save_errors(data["X_test"], yt, yp, probs)
    save_per_scene_results(yt, yp, data["test_ids"])

    print("\n" + "=" * 65)
    print("REZIME")
    print("=" * 65)
    for name, scene_key, y_key in [
        ("Training", "train_scenes", "y_train"),
        ("Validation", "val_scenes", "y_val"),
        ("Test", "test_scenes", "y_test"),
    ]:
        a, b = class_counts(data[y_key])
        print(f"{name:10s}: {len(data[scene_key])} scena | {len(data[y_key])} patch-eva | {a} bolestan + {b} zdrav")

    print(f"Najbolji val_loss je u epohi: {result['best_loss_epoch']}")
    print(f"Najveći val_accuracy je u epohi: {result['best_acc_epoch']}")
    print(f"Prag izabran na validation skupu: {best_threshold:.2f}")
    print(f"Validation accuracy — prag 0.50: {result['val_acc_default']:.2%}")
    print(f"Validation accuracy — tuned prag: {result['val_acc_tuned']:.2%}")
    print(f"TEST accuracy — prag 0.50: {result['test_acc_default']:.2%}")
    print(f"TEST accuracy — tuned prag: {result['test_acc_tuned']:.2%}")
    print("\nFajlovi:")
    print("  best_model.keras, model_final.keras")
    print("  model_summary.txt, cnn_arhitektura.png")
    print("  scene_split.txt, classification_report.txt")
    print("  per_scene_rezultati.csv, threshold_validation.csv")
    print("  rezultati.png, primeri_test.png, greske_test.png")
