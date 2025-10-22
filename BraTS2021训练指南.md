# BraTS2021数据集训练指南

## 📋 目录
- [数据集简介](#数据集简介)
- [数据准备](#数据准备)
- [数据预处理](#数据预处理)
- [训练配置](#训练配置)
- [完整训练流程](#完整训练流程)

---

## 🧠 数据集简介

**BraTS2021** (Brain Tumor Segmentation Challenge 2021) 是一个脑肿瘤分割数据集，包含多模态MRI图像。

### 数据特点
- **模态**: 4种MRI序列（T1, T1ce, T2, FLAIR）
- **标注**: 3类肿瘤区域（坏死核心、水肿区域、增强肿瘤）
- **格式**: NIfTI格式（.nii.gz）
- **尺寸**: 240×240×155（3D体积）

### 分割类别
- **0**: 背景
- **1**: 坏死核心（Necrotic Core, NCR）
- **2**: 水肿区域（Peritumoral Edema, ED）
- **4**: 增强肿瘤（Enhancing Tumor, ET）

---

## 📦 数据准备

### 步骤1: 下载BraTS2021数据集

1. 访问官方网站注册并下载：
   - 官网: https://www.med.upenn.edu/cbica/brats2021/
   - 或通过Kaggle: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

2. 下载后的目录结构：
```
BraTS2021/
├── BraTS2021_Training_Data/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz  # 标注文件
│   ├── BraTS2021_00002/
│   └── ...
└── BraTS2021_Validation_Data/
    └── ...
```

---

## 🔧 数据预处理

由于BraTS2021是3D NIfTI格式，而当前UNet项目使用2D图像，需要进行预处理。

### 方案1: 提取2D切片（推荐）

创建预处理脚本 `preprocess_brats.py`:

```python
import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def normalize_image(img):
    """归一化图像到0-255范围"""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

def process_brats_to_2d(
    brats_dir='BraTS2021/BraTS2021_Training_Data',
    output_dir='data',
    modality='flair',  # 选择使用的MRI模态
    slice_range=(60, 120),  # 提取的切片范围
    min_tumor_pixels=100  # 最小肿瘤像素数（过滤空白切片）
):
    """
    将BraTS 3D数据转换为2D切片
    
    参数:
        brats_dir: BraTS数据集路径
        output_dir: 输出目录
        modality: MRI模态 ('t1', 't1ce', 't2', 'flair')
        slice_range: 提取的切片范围（轴向切片）
        min_tumor_pixels: 最小肿瘤像素数，用于过滤空白切片
    """
    
    # 创建输出目录
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有患者目录
    patient_dirs = sorted([d for d in Path(brats_dir).iterdir() if d.is_dir()])
    
    print(f'找到 {len(patient_dirs)} 个患者数据')
    print(f'使用MRI模态: {modality}')
    print(f'提取切片范围: {slice_range[0]}-{slice_range[1]}')
    
    total_slices = 0
    
    for patient_dir in tqdm(patient_dirs, desc='处理患者数据'):
        patient_id = patient_dir.name
        
        # 读取MRI图像
        img_file = patient_dir / f'{patient_id}_{modality}.nii.gz'
        seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
        
        if not img_file.exists() or not seg_file.exists():
            print(f'警告: {patient_id} 缺少文件，跳过')
            continue
        
        # 加载NIfTI文件
        img_nii = nib.load(str(img_file))
        seg_nii = nib.load(str(seg_file))
        
        img_data = img_nii.get_fdata()
        seg_data = seg_nii.get_fdata()
        
        # 提取指定范围的切片
        for slice_idx in range(slice_range[0], min(slice_range[1], img_data.shape[2])):
            # 提取2D切片
            img_slice = img_data[:, :, slice_idx]
            seg_slice = seg_data[:, :, slice_idx]
            
            # 过滤空白切片（没有肿瘤的切片）
            if np.sum(seg_slice > 0) < min_tumor_pixels:
                continue
            
            # 归一化图像
            img_normalized = normalize_image(img_slice)
            
            # 简化标注（将BraTS的1,2,4标签映射为1,2,3）
            seg_simplified = np.zeros_like(seg_slice, dtype=np.uint8)
            seg_simplified[seg_slice == 1] = 1  # 坏死核心
            seg_simplified[seg_slice == 2] = 2  # 水肿
            seg_simplified[seg_slice == 4] = 3  # 增强肿瘤
            
            # 保存为PNG图像
            slice_name = f'{patient_id}_slice{slice_idx:03d}'
            
            # 保存图像（转换为RGB）
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
            Image.fromarray(img_rgb).save(img_dir / f'{slice_name}.png')
            
            # 保存掩码
            Image.fromarray(seg_simplified).save(mask_dir / f'{slice_name}.png')
            
            total_slices += 1
    
    print(f'\n✅ 预处理完成！')
    print(f'总共生成 {total_slices} 个2D切片')
    print(f'图像保存在: {img_dir}')
    print(f'掩码保存在: {mask_dir}')

if __name__ == '__main__':
    # 配置参数
    BRATS_DIR = 'BraTS2021/BraTS2021_Training_Data'  # 修改为你的BraTS数据路径
    OUTPUT_DIR = 'data'
    MODALITY = 'flair'  # 可选: 't1', 't1ce', 't2', 'flair'
    SLICE_RANGE = (60, 120)  # 提取中间60个切片
    MIN_TUMOR_PIXELS = 100  # 最小肿瘤像素数
    
    process_brats_to_2d(
        brats_dir=BRATS_DIR,
        output_dir=OUTPUT_DIR,
        modality=MODALITY,
        slice_range=SLICE_RANGE,
        min_tumor_pixels=MIN_TUMOR_PIXELS
    )
```

### 运行预处理

```bash
# 安装依赖
pip install nibabel

# 运行预处理脚本
python preprocess_brats.py
```

---

### 方案2: 多模态融合（高级）

如果想使用多个MRI模态，可以将它们融合为RGB图像：

```python
def process_brats_multimodal(
    brats_dir='BraTS2021/BraTS2021_Training_Data',
    output_dir='data',
    modalities=['t1', 't2', 'flair'],  # 使用3个模态作为RGB通道
    slice_range=(60, 120),
    min_tumor_pixels=100
):
    """使用多模态MRI作为RGB通道"""
    
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted([d for d in Path(brats_dir).iterdir() if d.is_dir()])
    
    print(f'使用多模态: {modalities}')
    total_slices = 0
    
    for patient_dir in tqdm(patient_dirs, desc='处理患者数据'):
        patient_id = patient_dir.name
        
        # 读取多个模态
        modality_data = []
        for mod in modalities:
            img_file = patient_dir / f'{patient_id}_{mod}.nii.gz'
            if not img_file.exists():
                print(f'警告: {patient_id} 缺少 {mod} 模态')
                break
            img_nii = nib.load(str(img_file))
            modality_data.append(img_nii.get_fdata())
        
        if len(modality_data) != len(modalities):
            continue
        
        # 读取分割标注
        seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
        seg_data = nib.load(str(seg_file)).get_fdata()
        
        # 提取切片
        for slice_idx in range(slice_range[0], min(slice_range[1], modality_data[0].shape[2])):
            seg_slice = seg_data[:, :, slice_idx]
            
            if np.sum(seg_slice > 0) < min_tumor_pixels:
                continue
            
            # 融合多模态为RGB
            rgb_channels = []
            for mod_data in modality_data:
                channel = normalize_image(mod_data[:, :, slice_idx])
                rgb_channels.append(channel)
            
            img_rgb = np.stack(rgb_channels, axis=-1)
            
            # 简化标注
            seg_simplified = np.zeros_like(seg_slice, dtype=np.uint8)
            seg_simplified[seg_slice == 1] = 1
            seg_simplified[seg_slice == 2] = 2
            seg_simplified[seg_slice == 4] = 3
            
            # 保存
            slice_name = f'{patient_id}_slice{slice_idx:03d}'
            Image.fromarray(img_rgb).save(img_dir / f'{slice_name}.png')
            Image.fromarray(seg_simplified).save(mask_dir / f'{slice_name}.png')
            
            total_slices += 1
    
    print(f'\n✅ 多模态预处理完成！生成 {total_slices} 个切片')
```

---

## 🚀 训练配置

### 修改模型类别数

BraTS2021有4个类别（背景 + 3类肿瘤），需要设置 `--classes 4`

### 推荐训练参数

#### 配置1: 标准训练（推荐）

```bash
python train.py \
  --epochs 80 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --classes 4 \
  --scale 0.5 \
  --validation 15 \
  --amp
```

#### 配置2: 高精度训练

```bash
python train.py \
  --epochs 120 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --classes 4 \
  --scale 0.75 \
  --validation 10 \
  --amp
```

#### 配置3: 低显存训练

```bash
python train.py \
  --epochs 80 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --classes 4 \
  --scale 0.25 \
  --validation 15 \
  --amp \
  --bilinear \
  --accumulate-grad-batches 4
```

---

## 📝 完整训练流程

### 步骤1: 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
pip install nibabel  # BraTS数据处理需要

# 检查GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 步骤2: 数据预处理

```bash
# 创建预处理脚本（见上文）
python preprocess_brats.py
```

### 步骤3: 验证数据

```bash
# 检查生成的数据
python -c "
from pathlib import Path
img_dir = Path('data/imgs')
mask_dir = Path('data/masks')
print(f'图像数量: {len(list(img_dir.glob(\"*.png\")))}')
print(f'掩码数量: {len(list(mask_dir.glob(\"*.png\")))}')
"
```

### 步骤4: 开始训练

```bash
# 使用推荐配置训练
python train.py \
  --epochs 80 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --classes 4 \
  --scale 0.5 \
  --validation 15 \
  --amp
```

### 步骤5: 监控训练

训练开始后会显示WandB链接，点击查看：
- 训练损失曲线
- 验证Dice分数
- 预测结果可视化

### 步骤6: 使用最佳模型

```bash
# 训练完成后，最佳模型保存在
# checkpoints/best_model.pth

# 使用模型进行预测
python predict.py \
  --model checkpoints/best_model.pth \
  --input test_image.png \
  --output prediction.png
```

---

## 🎯 训练技巧

### 1. 数据增强

BraTS数据可以通过以下方式增强：
- 旋转（90°, 180°, 270°）
- 翻转（水平、垂直）
- 亮度调整
- 对比度调整

### 2. 类别不平衡处理

BraTS数据中背景占比很大，可以：
- 使用加权损失函数
- 过采样肿瘤切片
- 使用Focal Loss

### 3. 评估指标

BraTS官方使用的评估指标：
- Dice系数（DSC）
- Hausdorff距离（HD95）
- 敏感度（Sensitivity）
- 特异度（Specificity）

---

## ⚠️ 常见问题

### Q1: 预处理后图像数量太少

**原因**: `min_tumor_pixels` 阈值太高，过滤掉了太多切片

**解决**: 降低阈值
```python
MIN_TUMOR_PIXELS = 50  # 从100降低到50
```

### Q2: 训练时显存不足

**解决方案**:
```bash
# 使用低显存配置
python train.py --batch-size 2 --scale 0.25 --amp --bilinear
```

### Q3: 验证Dice分数很低

**可能原因**:
1. 类别数设置错误（应该是4）
2. 标注映射不正确
3. 学习率过大

**检查**:
```python
# 检查掩码值
from PIL import Image
import numpy as np
mask = np.array(Image.open('data/masks/xxx.png'))
print(f'掩码唯一值: {np.unique(mask)}')  # 应该是 [0, 1, 2, 3]
```

### Q4: 如何处理3D预测

当前模型是2D的，如果需要3D预测：
1. 对每个切片分别预测
2. 将预测结果堆叠回3D体积
3. 可选：使用3D后处理（如CRF）

---

## 📊 预期结果

### 训练时间（参考）
- GPU: RTX 3090
- 数据量: 5000个切片
- 配置: batch_size=8, epochs=80

**预计时间**: 2-3小时

### 性能指标（参考）
- **Dice系数**: 0.75-0.85（取决于类别）
- **训练损失**: 0.2-0.4
- **验证损失**: 0.3-0.5

---

## 🔗 相关资源

- **BraTS官网**: https://www.med.upenn.edu/cbica/brats2021/
- **论文**: https://arxiv.org/abs/2107.02314
- **UNet论文**: https://arxiv.org/abs/1505.04597
- **NiBabel文档**: https://nipy.org/nibabel/

---

## 📝 总结

使用BraTS2021数据集训练UNet的关键步骤：

1. ✅ 下载BraTS2021数据集
2. ✅ 运行预处理脚本将3D数据转换为2D切片
3. ✅ 设置正确的类别数（`--classes 4`）
4. ✅ 使用推荐的训练参数
5. ✅ 监控训练过程并调整超参数
6. ✅ 使用最佳模型进行预测

**祝训练顺利！** 🎉
