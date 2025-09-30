"""
Count + metrics for Faster R-CNN (ResNet-50-FPN).
- Gera CSV com contagens por imagem
- (Opcional) métricas por imagem (TP/FP/FN, Precisão, Recall, F1, IoU) usando GT (COCO ou YOLO)
- Overlays com caixas: NEGATIVA=verde, POSITIVA=vermelho, INCERTA=amarelo

Observação sobre TN/Accuracy/Specificity:
- Para detecção por caixas (object detection), TN não é bem-definido sem um universo de "negativos" (ex.: pixels).
- Portanto, este script NÃO cria colunas tn_* / tn_all por padrão.
- Se você vier a ter uma fonte consistente de TN (ex.: métricas em nível de pixel/segmentação),
  basta adicionar tn_* / tn_all aqui que o report já aproveitará para calcular Accuracy/Specificity.
"""

import os, csv, json
from collections import defaultdict, Counter

import torch, torchvision
from PIL import Image, ImageDraw

# ========= INFERENCE CONFIG =========
CKPT = r"D:\UFSC\2025.2\TCC\resNet\checkpoint\fasterrcnn_r50fpn_e12_best.pth"
IMG_DIR = r"D:\UFSC\2025.2\TCC\Dataset\dataset-arthur-25.v2i.coco\test"
OUT_CSV = r"D:\UFSC\2025.2\TCC\resNet\contagens_test.csv"
SCORE_THR = 0.50

SAVE_DEBUG_OVERLAY = True
OUT_DEBUG_DIR = r"D:\UFSC\2025.2\TCC\resNet\vis"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========= END INFERENCE CONFIG =====

# ========= METRICS CONFIG =========
METRICS = {
    "ENABLED": True,
    "IOU_THR": 0.50,
    "GT_MODE": "coco",  # "coco" ou "yolo"
    "COCO_JSON": r"D:\UFSC\2025.2\TCC\Dataset\dataset-arthur-25.v2i.coco\test\_annotations.coco.json",
    "YOLO_LABELS_DIR": r"D:\UFSC\2025.2\TCC\Dataset\dataset-arthur-25.v2i.yolov9\test\labels",
    "YOLO_CLASS_NAMES": ["negative", "positive"],
}
# ========= END METRICS CONFIG =======

# ---------- helpers ----------
POS_NAMES = {"positive", "pos", "positivo", "positiva"}
NEG_NAMES = {"negative", "neg", "negativo", "negativa"}
UNC_NAMES = {"uncertain", "incerta", "duvidosa"}


def is_pos(name: str) -> bool: return name.lower() in POS_NAMES
def is_neg(name: str) -> bool: return name.lower() in NEG_NAMES
def is_unc(name: str) -> bool: return name.lower() in UNC_NAMES

def color_for(name: str):
    if is_pos(name): return (255, 0, 0)
    if is_neg(name): return (0, 255, 0)
    if is_unc(name): return (255, 255, 0)
    return (0, 255, 255)

def safe_div(n, d): return (n / d) if d else 0.0

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0

def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """
    Greedy matching por score (padronizado para detecção):
    retorna TP/FP/FN; IoU agregado é calculado à parte como TP/(TP+FP+FN).
    """
    gt_used = [False] * len(gt_boxes)
    tp = 0
    for i in sorted(range(len(pred_boxes)), key=lambda k: pred_scores[k], reverse=True):
        pb = pred_boxes[i]
        best_j, best_iou = -1, 0.0
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = box_iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            gt_used[best_j] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ---------- model ----------
def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE)
    num_classes = ckpt["num_classes"]
    idx_to_name = ckpt["idx_to_name"]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, idx_to_name


# ---------- GT ----------
def load_gt_for_metrics():
    if not METRICS["ENABLED"]:
        return None
    mode = METRICS["GT_MODE"].lower()
    if mode == "coco":
        try:
            from pycocotools.coco import COCO
        except Exception:
            raise SystemExit(
                "pycocotools requerido para métricas com COCO (pip install pycocotools)"
            )
        coco = COCO(METRICS["COCO_JSON"])
        cats = coco.loadCats(coco.getCatIds())
        cat_id_to_name = {c["id"]: c["name"] for c in cats}
        gt_index = defaultdict(list)
        for img_id in coco.imgs:
            im = coco.imgs[img_id]
            fname = os.path.basename(im["file_name"])
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            for a in anns:
                x, y, w, h = a["bbox"]
                gt_index[fname].append(
                    ([x, y, x + w, y + h], cat_id_to_name[a["category_id"]])
                )
        return {"mode": "coco", "gt_index": gt_index}
    elif mode == "yolo":
        labels_dir = METRICS["YOLO_LABELS_DIR"]
        class_names = METRICS["YOLO_CLASS_NAMES"]
        if not os.path.isdir(labels_dir):
            raise SystemExit(f"Pasta YOLO não encontrada: {labels_dir}")
        return {"mode": "yolo", "labels_dir": labels_dir, "class_names": class_names}
    else:
        raise SystemExit("GT_MODE inválido. Use 'coco' ou 'yolo'.")


def read_yolo_labels(txt_path, class_names, W, H):
    out = []
    if not os.path.exists(txt_path):
        return out
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            cid = int(p[0])
            xc, yc, w, h = map(float, p[1:])
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            out.append(([x1, y1, x2, y2], name))
    return out


# ---------- draw ----------
def draw_boxes(img: Image.Image, boxes, labels, scores, thr=0.5, idx_to_name=None):
    draw = ImageDraw.Draw(img)
    for box, lab, sc in zip(boxes, labels, scores):
        if sc < thr:
            continue
        if isinstance(lab, (int, float)) and idx_to_name is not None:
            name = idx_to_name.get(int(lab), str(lab))
        else:
            name = str(lab)
        x1, y1, x2, y2 = [float(v) for v in box]
        col = color_for(name)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        draw.text((x1 + 2, y1 + 2), f"{name}:{sc:.2f}", fill=col)
    return img


def main():
    model, idx_to_name = load_model()
    tf = torchvision.transforms.ToTensor()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if SAVE_DEBUG_OVERLAY:
        os.makedirs(OUT_DEBUG_DIR, exist_ok=True)

    gt_data = load_gt_for_metrics() if METRICS["ENABLED"] else None
    IOU_THR = METRICS["IOU_THR"]

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]
    files.sort()

    rows = []
    agg_metrics = {"_all": {"TP": 0, "FP": 0, "FN": 0}}  # manter simples; TN não é incluído aqui
    for fname in files:
        path = os.path.join(IMG_DIR, fname)
        img = Image.open(path).convert("RGB")
        W, H = img.size
        x = tf(img).to(DEVICE)

        with torch.no_grad():
            out = model([x])[0]

        keep = out["scores"] >= SCORE_THR
        boxes = out["boxes"][keep].tolist()
        labels = [idx_to_name[int(l)] for l in out["labels"][keep].tolist()]
        scores = out["scores"][keep].tolist()

        uniq_names = set(idx_to_name.values())
        counts = {name: 0 for name in uniq_names}
        for n in labels:
            counts[n] = counts.get(n, 0) + 1

        per_img_metrics = {}
        if gt_data:
            if gt_data["mode"] == "coco":
                gt_list = gt_data["gt_index"].get(fname, [])
            else:
                txt = os.path.join(
                    gt_data["labels_dir"], os.path.splitext(fname)[0] + ".txt"
                )
                gt_list = read_yolo_labels(txt, gt_data["class_names"], W, H)

            gt_by_c = defaultdict(list)
            pred_by_c = defaultdict(list)
            score_by_c = defaultdict(list)
            for (x1, y1, x2, y2), name in gt_list:
                gt_by_c[name].append([x1, y1, x2, y2])
            for b, n, s in zip(boxes, labels, scores):
                pred_by_c[n].append(b)
                score_by_c[n].append(s)

            for cname in set(list(gt_by_c.keys()) + list(pred_by_c.keys())):
                tp, fp, fn = greedy_match(
                    pred_by_c[cname], score_by_c[cname], gt_by_c[cname], IOU_THR
                )
                prec = safe_div(tp, tp + fp)
                rec = safe_div(tp, tp + fn)
                f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
                iou = safe_div(tp, tp + fp + fn)

                per_img_metrics[cname] = {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "P": prec,
                    "R": rec,
                    "F1": f1,
                    "IoU": iou,
                }

                agg = agg_metrics.setdefault(cname, {"TP": 0, "FP": 0, "FN": 0})
                agg["TP"] += tp
                agg["FP"] += fp
                agg["FN"] += fn
                agg_metrics["_all"]["TP"] += tp
                agg_metrics["_all"]["FP"] += fp
                agg_metrics["_all"]["FN"] += fn

        row = {"image": fname, **counts}
        if per_img_metrics:
            for cname, m in per_img_metrics.items():
                row[f"tp_{cname}"] = m["TP"]
                row[f"fp_{cname}"] = m["FP"]
                row[f"fn_{cname}"] = m["FN"]
                row[f"prec_{cname}"] = round(m["P"], 4)
                row[f"rec_{cname}"] = round(m["R"], 4)
                row[f"f1_{cname}"] = round(m["F1"], 4)
                row[f"iou_{cname}"] = round(m["IoU"], 4)

            tp_img = sum(m["TP"] for m in per_img_metrics.values())
            fp_img = sum(m["FP"] for m in per_img_metrics.values())
            fn_img = sum(m["FN"] for m in per_img_metrics.values())
            p_img = safe_div(tp_img, tp_img + fp_img)
            r_img = safe_div(tp_img, tp_img + fn_img)
            f1_img = safe_div(2 * p_img * r_img, p_img + r_img) if (p_img + r_img) else 0.0
            iou_img = safe_div(tp_img, tp_img + fp_img + fn_img)

            row.update(
                {
                    "tp_all": tp_img,
                    "fp_all": fp_img,
                    "fn_all": fn_img,
                    "prec_all": round(p_img, 4),
                    "rec_all": round(r_img, 4),
                    "f1_all": round(f1_img, 4),
                    "iou_all": round(iou_img, 4),
                }
            )

        rows.append(row)

        if SAVE_DEBUG_OVERLAY:
            vis = img.copy()
            vis = draw_boxes(vis, boxes, labels, scores, thr=SCORE_THR, idx_to_name=None)
            vis.save(os.path.join(OUT_DEBUG_DIR, f"det_{fname}"))

    fieldnames = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x != "image", x))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("CSV salvo em:", OUT_CSV)

    if METRICS["ENABLED"]:

        def pack(tp, fp, fn):
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
            iou = safe_div(tp, tp + fp + fn)
            return {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": r, "F1": f1, "IoU": iou}

        summary = {"iou_thr": METRICS["IOU_THR"], "per_class": {}, "micro": {}}
        for cname, m in agg_metrics.items():
            if cname == "_all":
                continue
            summary["per_class"][cname] = pack(m["TP"], m["FP"], m["FN"])
        m = agg_metrics["_all"]
        summary["micro"] = pack(m["TP"], m["FP"], m["FN"])
        sum_path = os.path.splitext(OUT_CSV)[0] + "_summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Resumo agregado salvo em:", sum_path)


if __name__ == "__main__":
    main()
