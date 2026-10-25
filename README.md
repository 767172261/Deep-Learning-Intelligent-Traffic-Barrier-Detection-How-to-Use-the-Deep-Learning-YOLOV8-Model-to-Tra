深度学习智慧化交通路障检测 如何使用深度学习YOLOV8模型训练道路障碍物落石石头检测数据集
@[toc]
以下文字及代码仅供参考学习使用

**道路障碍物落石数据集
两套，第一套数据集类别为Car', 'Motorcycle', 'People', 'shitou', 'Warning-Crash', 'landslide']
第二套数据集类别为【shitou】**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae66756b2bbd4c0c8d731bec90e176f9.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/88803ea6665c400c8f78845628cea495.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a010a87208341acbc6a04524ca06cee.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/854983e3a5cf4d37bfe9f8bdad886035.png)
1
非常感谢你的详细描述！你提供的 **落石检测数据集** 包含两套数据，适用于 **地质灾害监测、山区道路安全预警、智能巡检** 等场景。我们将目标统一为：**检测“落石（shitou）”这一关键类别**，并可根据需要选择是否保留其他类别（如车辆、行人、滑坡等）。

---

## ✅ 一、任务分析与目标设定

### 两套数据集情况：

| 数据集 | 类别 | 图像数 | 说明 |
|--------|------|--------|------|
| 第一套 | `['Car', 'Motorcycle', 'People', 'shitou', 'Warning-Crash', 'landslide']` | ~300+ 张 | 多类别，包含落石和其他干扰物 |
| 第二套 | `['shitou']` | ~100+ 张 | 专注落石，可用于增强训练 |

### ✅ 最终目标：
- **方案 A（推荐）**：只检测 `shitou`（二分类检测），提升模型专注度和召回率
- **方案 B**：保留 `shitou` 和 `landslide`，实现“落石/滑坡”双灾害检测
- **方案 C**：全类别检测（适合通用场景）

> 本教程以 **方案 A：只训练 `shitou` 类别** 为例，因数据量小，聚焦单一目标更易收敛。

---

## ✅ 二、环境搭建（从零开始）

### 1. 确认 CUDA 驱动（GPU 加速）

打开终端：
```bash
nvidia-smi
```
查看 CUDA 版本（如 11.8 或 12.1）。若无 GPU 可用 CPU，但训练较慢。

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装后打开 **Anaconda Prompt**（Windows）或终端（Mac/Linux）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建环境
conda create -n rockfall-det python=3.9

# 激活环境
conda activate rockfall-det
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics YOLOv8
pip install ultralytics

# 安装 OpenCV 和其他工具
pip install opencv-python matplotlib pillow tqdm
```

✅ 验证安装：
```python
import torch
print(torch.cuda.is_available())  # 应输出 True（有 GPU 时）
```

---

## ✅ 三、数据集合并与格式统一（关键步骤）

### 文件目录结构

```
datasets/
└── rockfall/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

---

### 1. 数据预处理：统一类别为 `shitou`

你需要将两套数据的标签文件（`.txt`）进行 **类别重映射**：

- 第一套中：只保留 `class_id == 3`（shitou）的标注行
- 第二套中：所有标注都是 `shitou`，但类别 ID 可能是 0，需统一为 0
- 最终所有标签文件中，`class_id = 0` 表示 `shitou`

#### 📌 脚本：`merge_labels.py`

```python
# merge_labels.py
import os
import shutil

# 原始路径（假设你已将两套数据复制到以下文件夹）
src1_images_train = "data1/images/train"
src1_labels_train = "data1/labels/train"
src1_images_val = "data1/images/val"
src1_labels_val = "data1/labels/val"

src2_images_train = "data2/images/train"
src2_labels_train = "data2/labels/train"
src2_images_val = "data2/images/val"
src2_labels_val = "data2/labels/val"

# 目标路径
dst_images_train = "datasets/rockfall/images/train"
dst_labels_train = "datasets/rockfall/labels/train"
dst_images_val = "datasets/rockfall/images/val"
dst_labels_val = "datasets/rockfall/labels/val"

os.makedirs(dst_images_train, exist_ok=True)
os.makedirs(dst_labels_train, exist_ok=True)
os.makedirs(dst_images_val, exist_ok=True)
os.makedirs(dst_labels_val, exist_ok=True)

def filter_and_copy(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, shitou_old_id=3):
    for img_file in os.listdir(src_img_dir):
        image_path = os.path.join(src_img_dir, img_file)
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(src_lbl_dir, label_file)

        # 复制图像
        shutil.copy(image_path, dst_img_dir)

        # 处理标签：只保留 shitou (原 class_id=3)
        new_lines = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == shitou_old_id:  # 只保留落石
                        new_lines.append(f"0 {' '.join(parts[1:])}\n")  # 新 class_id=0

            # 写入新标签文件
            with open(os.path.join(dst_lbl_dir, label_file), 'w') as f:
                f.writelines(new_lines)

# 合并训练集
filter_and_copy(src1_images_train, src1_labels_train, dst_images_train, dst_labels_train, shitou_old_id=3)
filter_and_copy(src2_images_train, src2_labels_train, dst_images_train, dst_labels_train, shitou_old_id=0)

# 合并验证集
filter_and_copy(src1_images_val, src1_labels_val, dst_images_val, dst_labels_val, shitou_old_id=3)
filter_and_copy(src2_images_val, src2_labels_val, dst_images_val, dst_labels_val, shitou_old_id=0)

print("✅ 数据集合并完成！所有类别统一为 'shitou' (class_id=0)")
```

运行：
```bash
python merge_labels.py
```

---

### 2. 创建 `data.yaml`

```yaml
# datasets/rockfall/data.yaml
train: ../datasets/rockfall/images/train
val: ../datasets/rockfall/images/val

nc: 1
names: ['shitou']
```

---

## ✅ 四、训练代码（YOLOv8 官方预训练模型）

### 📌 `train.py` —— 针对小样本、小目标优化

```python
# train.py
from ultralytics import YOLO
import torch

# 检查 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 推荐使用 yolov8m（小目标更强）
model = YOLO('yolov8m.pt')  # 可选: yolov8l.pt

# 开始训练
results = model.train(
    data='datasets/rockfall/data.yaml',
    epochs=300,                # 小样本需多轮训练
    imgsz=640,                 # 可选 640 或 1024（提升小目标检测）
    batch=16,                  # 根据显存调整
    name='rockfall_yolov8m',
    device=device,
    workers=4,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    patience=50,               # 早停
    project='runs/rockfall',
    amp=True,                  # 混合精度
    close_mosaic=20,
    # --- 数据增强 ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.6,                 # 增强泛化
    copy_paste=0.4,            # 增强落石样本
    degrees=15.0,
)
```

---

## ✅ 五、推理代码（检测新图像）

### 📌 `predict.py`

```python
# predict.py
from ultralytics import YOLO

model = YOLO('runs/rockfall/rockfall_yolov8m/weights/best.pt')

results = model.predict(
    source='road_scene.jpg',
    save=True,
    conf=0.25,
    iou=0.45,
    project='runs/predict',
    name='rockfall_test',
    imgsz=640
)

for result in results:
    boxes = result.boxes
    print(f"检测到 {len(boxes)} 块落石：")
    for box in boxes:
        conf = float(box.conf)
        print(f"  落石置信度: {conf:.3f}")
```

---

## ✅ 六、评估模型性能（在验证集上）

### 📌 `evaluate.py`

```python
# evaluate.py
from ultralytics import YOLO

model = YOLO('runs/rockfall/rockfall_yolov8m/weights/best.pt')

metrics = model.val(
    data='datasets/rockfall/data.yaml',
    split='val',
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/eval',
    name='rockfall_val'
)

print(f"mAP@0.5:  {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall:    {metrics.box.r:.4f}")
```

---

## ✅ 七、如果你要训练多类别（方案 C）

只需修改 `data.yaml`：

```yaml
nc: 6
names: ['Car', 'Motorcycle', 'People', 'shitou', 'Warning-Crash', 'landslide']
```

并在 `merge_labels.py` 中保留所有类别（无需过滤），然后训练即可。

---

以上文字及代码仅供参考学习使用。
