# IHC-Detect-ResNet

Detecção e contagem de células em lâminas IHC usando **Faster R-CNN (ResNet-50-FPN)**.
Pipeline completo: **treino → contagem → análise de métricas e relatórios**, com suporte a GPU (AMP), COCO mAP e métricas de contagem (Precisão/Recall/F1).

---

## Destaques

- Treino em COCO (train/valid/test) com ResNet-50-FPN pré-treinada.
- Contagem por classe (ex.: negativa/positiva/incerta) em imagens grandes.
- Métricas: mAP COCO (AP, AP50, AP75, AR) + TP/FP/FN, Precisão, Recall, F1 por classe e global.
- Overlays: caixas coloridas (negativa=verde, positiva=vermelho, incerta=amarelo).
- Relatórios automáticos: CSV enriquecido, Markdown e gráficos.

---

## 📦 Estrutura
```
IHC-Detect-ResNet/
  scripts/
    detector_train_resnet50_fpn.py
    detector_count_resnet50_fpn.py
    analyze_counts_report.py
  checkpoint/                      # saídas dos treinos (criado em runtime)
  reports/                         # relatórios/figuras (criado em runtime)
  
  data/                            # (fora do git) datasets em COCO
```

---

## 🖥️ Requisitos

- Python 3.10+
- NVIDIA + CUDA (recomendado). Instalação do PyTorch com CUDA (Windows/Linux):

```
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

- Demais libs:
```
pip install pillow pandas matplotlib pycocotools
```
---

## 📁 Dataset (COCO)

Layout esperado:
```
data/dataset-arthur-25.v2i.coco/
  train/_annotations.coco.json + imagens
  valid/_annotations.coco.json + imagens
  test/_annotations.coco.json  + imagens
```
---

## 🚀 Como usar

### 1) Treinar

Edite os caminhos no topo de scripts/detector_train_resnet50_fpn.py (COCO_ROOT/OUT_DIR ou OUT_CKPT/OUT_BEST) e rode:
```
python scripts/detector_train_resnet50_fpn.py
```

Saídas por run: `last.pth`, `best.pth`, `metrics_epoch.jsonl`, `final_metrics.json`.

>Dicas (GPU 8 GB): `BATCH_SIZE=2`, `USE_AMP=True`, `MIN_SIZE=896`, `MAX_SIZE=1536`. Para treinos longos, escale `LR_STEP_SIZE` (ex.: 36 épocas → step_size=12) ou use `CosineAnnealingLR`.

### 2) Contar em imagens grandes

Edite `CKPT`, `IMG_DIR`, `OUT_CSV`, `SCORE_THR` e, se quiser métricas com `GT`, o bloco `METRICS`. Rode:
```
python scripts/detector_count_resnet50_fpn.py
```

Saídas:

`contagens_*.csv` (contagens por imagem + tp/fp/fn/prec/rec/f1 por classe e global)

`*_summary.json` (métricas agregadas)

`vis/` com imagens marcadas (neg=verde, pos=vermelho, incerta=amarelo)

### 3) Analisar resultados e gerar relatório

Edite `INPUT_CSV` e `OUTPUT_DIR` em `scripts/analyze_counts_report.py` e rode:
```
python scripts/analyze_counts_report.py
```

Saídas em `reports/`:

`counts_enriched.csv` (TOTAL e taxas por imagem)

`metrics_summary.csv` (TP/FP/FN + Precisão/Recall/F1 por classe e micro)

`summary_report.md` (estatísticas gerais, TOP-N por classe)

`precision_report.md` (ranking de precisão por classe e por imagem)

Figuras: totais por classe, histograma da taxa positiva, F1 por classe e Precisão por classe.

---

## 🔧 Principais configs (onde mexer)

-Treino: `EPOCHS`, `BATCH_SIZE`, `LR`, `LR_STEP_SIZE/LR_GAMMA` (ou `Cosine`), `MIN_SIZE/MAX_SIZE` (ou lista para multi-scale), `SEED`, `NUM_WORKERS`.
-Contagem: `SCORE_THR` (sensibilidade vs. falsos positivos), `METRICS.IOU_THR` (acerto por IoU), `GT_MODE` (coco ou yolo).
-Overlays: cores já mapeadas (neg=verde, pos=vermelho, incerta=amarelo).

---

## 📊 O que significam as métricas

-mAP COCO: AP geral (0.50:0.95), AP50, AP75, AR.
-Precisão (P) = TP / (TP + FP).
-Recall (R) = TP / (TP + FN).
-F1 = 2PR / (P + R).
>Todas dependem do SCORE_THR (confiança) e do IOU_THR configurados.

---

## 🛠️ Solução de problemas

-cuda available: False: instale o PyTorch com CUDA cu121 (comando acima) e verifique `nvidia-smi`.
-OOM (memória): reduza `MIN_SIZE/MAX_SIZE` (ex.: 800/1333) ou `BATCH_SIZE=1`.
-GPU ociosa: aumente `NUM_WORKERS` (8–12 no i7-12700F) e use `persistent_workers=True`.

---


Baseado no problema do TCC da Thaynara (métodos clássicos) — aqui estendido com CNNs de detecção e pipeline de contagem/relatórios.
