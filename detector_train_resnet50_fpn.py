"""
Treino de Faster R-CNN (ResNet-50-FPN) em dataset COCO (detecção).
Config pronto para GPU de 8 GB:
- batch_size = 2
- AMP (mixed precision) habilitado
- min_size/max_size ajustados (896/1536) para preservar detalhes sem estourar VRAM
- checkpoint "melhor" e "último"
- grad clipping
- flip horizontal com ajuste de caixas

Requisitos:
  pip install torch torchvision pycocotools pillow
Layout esperado:
COCO_ROOT/
  train/_annotations.coco.json + imagens
  valid/_annotations.coco.json + imagens
  (test é opcional no treino)
"""

import os, time, random
from typing import Dict, List, Tuple
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from PIL import Image
import torch

try:
    from pycocotools.coco import COCO
except Exception as e:
    raise SystemExit(
        "Missing pycocotools. Install with: pip install pycocotools"
    ) from e

# ========= CONFIG =========
COCO_ROOT = r"D:\UFSC\2025.2\TCC\Dataset\dataset-arthur-25.v2i.coco"
OUT_CKPT = r"D:\UFSC\2025.2\TCC\resNet\checkpoint\fasterrcnn_r50fpn_e12_last.pth"
OUT_BEST = OUT_CKPT.replace("_last.pth", "_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60
BATCH_SIZE = 4
NUM_WORKERS = 10
LR = 5e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 12
LR_GAMMA = 0.1
SCORE_THR_EVAL = 0.50
USE_AMP = True
GRAD_CLIP = 5.0
MIN_SIZE = 896
MAX_SIZE = 1536
SEED = 44
# ========= END CONFIG =====


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


def build_coco(split: str):
    img_dir = os.path.join(COCO_ROOT, split)
    ann = os.path.join(img_dir, "_annotations.coco.json")
    if not os.path.exists(ann):
        raise FileNotFoundError(f"Annotations not found for split '{split}': {ann}")
    coco = COCO(ann)
    img_ids = list(sorted(coco.imgs.keys()))
    cats = coco.loadCats(coco.getCatIds())
    cat_ids = sorted([c["id"] for c in cats])
    id_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    idx_to_name = {
        i + 1: c["name"] for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))
    }
    return coco, img_ids, img_dir, id_to_idx, idx_to_name


class CocoDet(torch.utils.data.Dataset):
    def __init__(self, split: str, train: bool):
        self.split = split
        self.train = train
        self.coco, self.ids, self.img_dir, self.id_to_idx, self.idx_to_name = (
            build_coco(split)
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        iminfo = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, iminfo["file_name"])
        img = Image.open(path).convert("RGB")
        W, H = img.width, img.height

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.id_to_idx[a["category_id"]])
            area.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        # ---- augment simples: flip horizontal (apenas no train) ----
        if self.train and random.random() < 0.5 and len(boxes) > 0:
            img = TF.hflip(img)
            b = target["boxes"]
            x1 = b[:, 0].clone()
            x2 = b[:, 2].clone()
            b[:, 0] = W - x2
            b[:, 2] = W - x1
            target["boxes"] = b

        img = TF.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int):
    # carrega Faster R-CNN com backbone ResNet-50-FPN pré-treinado
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
    )
    # troca a cabeça de classificação (bg + num_classes)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


@torch.no_grad()
def simple_eval_print(model, dl, idx_to_name, device, score_thr=0.5, max_batches=10):
    model.eval()
    stats = {name: 0 for name in idx_to_name.values()}
    n_images = 0
    for b, (images, targets) in enumerate(dl):
        images = [im.to(device) for im in images]
        outputs = model(images)
        for out in outputs:
            keep = out["scores"] >= score_thr
            labels = out["labels"][keep].tolist()
            for lab in labels:
                stats[idx_to_name[lab]] = stats.get(idx_to_name[lab], 0) + 1
            n_images += 1
        if b + 1 >= max_batches:
            break
    if n_images > 0:
        print(f"[val quick] detections@{score_thr}: {stats} on {n_images} images")
    model.train()


def main():
    print("Device:", DEVICE)

    train_ds = CocoDet("train", train=True)
    val_ds = CocoDet("valid", train=False)
    num_classes = len(train_ds.id_to_idx) + 1
    print("Classes:", train_ds.idx_to_name)

    pin = DEVICE == "cuda"
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        collate_fn=collate_fn,
        pin_memory=pin,
    )

    model = get_model(num_classes).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    lr_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    best_loss = float("inf")
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for images, targets in train_dl:
            images = [im.to(DEVICE, non_blocking=True) for im in images]
            targets = [
                {k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[OOM] Reduza MIN_SIZE/MAX_SIZE ou BATCH_SIZE.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            running += float(loss.item())

        lr_sched.step()
        dt = time.time() - t0
        avg_loss = running / max(1, len(train_dl))
        print(f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f} - {dt:.1f}s")

        simple_eval_print(model, val_dl, train_ds.idx_to_name, DEVICE, SCORE_THR_EVAL)

        # salvar "melhor até agora"
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "idx_to_name": train_ds.idx_to_name,
                },
                OUT_BEST,
            )
            print(f"[ckpt] melhor até agora salvo em {OUT_BEST} (loss={best_loss:.4f})")

        # salvar "último"
        torch.save(
            {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "idx_to_name": train_ds.idx_to_name,
            },
            OUT_CKPT,
        )

    print("Checkpoints salvos em:")
    print(" - last:", OUT_CKPT)
    print(" - best:", OUT_BEST)


if __name__ == "__main__":
    main()
