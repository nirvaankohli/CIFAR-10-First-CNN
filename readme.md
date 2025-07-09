# CIFAR-10 ResNet-18 Advanced Training

A PyTorch implementation for progressively enhanced CNNs on the CIFAR-10 dataset.  
This repo tracks three versions:

- **V1**: simple CNN → ~82 % test accuracy  
- **V2**: deeper CNN + basic augmentations → ~87 % test accuracy  
- **V3**: ResNet-18 + advanced augmentations, SWA & AMP → 92%

---

## 🔍 About CIFAR-10

- **Dataset size**: 60 000 images (32×32, RGB), 10 classes  
  - 50 000 train / 10 000 test  
  - 6 000 images per class (balanced)  
- **Classes**:  
  
    ```
    airplane, automobile, bird, cat, deer,
    dog, frog, horse, ship, truck
    ```
- **Challenges**  
  - Low resolution makes fine-grained classes (cat vs. dog) hard  
  - Varied backgrounds and viewpoints  
  - Requires strong augmentation & regularization

---

## 🗂️ Repository Structure

```



````

---

## 🚀 Version History & Key Differences

| Version | Model                                  | Augmentations & Regularization                                                                                  | Val Accuracy |
|-------:|:---------------------------------------|:---------------------------------------------------------------------------------------------------------------|-------------:|
| **V1** | 3-layer CNN                            | • No augmentation<br>• Standard SGD                                                                            | ~ 82 %       |
| **V2** | Deeper CNN + BatchNorm + Dropout       | • RandomCrop + HorizontalFlip<br>• LR scheduling + weight decay                                                  | ~ 87 %       |
| **V3** | ResNet-18 (adapted for 32×32)          | • RandomCrop(32,pad=4)<br>• RandomHorizontalFlip<br>• RandAugment (2 ops, mag=9)<br>• RandomErasing (p=0.2)<br>• MixUp (α=0.2) & CutMix (α=1.0,p=0.5)<br>• Label smoothing (0.1)<br>• CosineAnnealingLR + SWA (start @ epoch 40)<br>• AMP (mixed-precision) | 92%|

---

## ⚙️ Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/nirvaankohli/CIFAR-10-First-CNN.git
   cd CIFAR-10-First-CNN
2. **Create & activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install torch torchvision tensorboard tqdm
   ```

---

## ▶️ Usage

```bash
python3 Cifar10V3.py \
  --data-dir ./data \
  --batch-size 128 \
  --epochs 50 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --cutmix-prob 0.5 \
  --swa-start 40
```


* **Metrics saved** in `V3_metrics.csv`
* **Best checkpoints**:

  * `modelV3.pth`

---

## 📊 Dataset & Training Insights

1. **Balanced classes** eliminate the need for re-weighting.
2. **Augmentation gains**

   * V1→V2: +5 pp with random crop & flip
   * V2→V3: further regularization (+RandAugment, MixUp/CutMix, Erasing)
3. **Schedulers & SWA**

   * CosineAnnealingLR yields smooth LR decay
   * SWA (start @ epoch 40) boosts final accuracy by \~1–2 pp
4. **AMP** halves memory and speeds up GPU training without accuracy loss

---

## 🤝 Contributing

1. Fork this repo
2. Create your branch: `git checkout -b feature/XYZ`
3. Commit: `git commit -m "Add XYZ"`
4. Push: `git push origin feature/XYZ`
5. Open a Pull Request

---

## 📝 License

This project is MIT-licensed. See [LICENSE](LICENSE) for details.

---

## ⚓ Shipwrecked 

<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 75%;">
  </a>
</div>

### See you in Boston!

---
