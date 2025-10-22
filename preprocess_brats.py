"""
BraTS2021数据集预处理脚本

将3D NIfTI格式的BraTS数据转换为2D PNG切片，用于UNet训练

功能:
1. 读取BraTS 3D MRI数据（.nii.gz格式）
2. 提取2D轴向切片
3. 归一化图像到0-255范围
4. 简化分割标注（1,2,4 -> 1,2,3）
5. 保存为PNG格式
6. 过滤空白切片（无肿瘤区域）

使用方法:
    python preprocess_brats.py

配置参数:
    BRATS_DIR: BraTS数据集路径
    OUTPUT_DIR: 输出目录
    MODALITY: MRI模态选择
    SLICE_RANGE: 提取的切片范围
    MIN_TUMOR_PIXELS: 最小肿瘤像素数
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 检查nibabel是否安装
try:
    import nibabel as nib
except ImportError:
    print("错误: 未安装nibabel库")
    print("请运行: pip install nibabel")
    sys.exit(1)


def normalize_image(img):
    """
    归一化图像到0-255范围
    
    Args:
        img: 输入图像数组
        
    Returns:
        归一化后的uint8图像
    """
    img = img.astype(np.float32)
    # 避免除零错误
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    
    img = (img - img_min) / (img_max - img_min)
    return (img * 255).astype(np.uint8)


def process_brats_to_2d(
    brats_dir='BraTS2021/BraTS2021_Training_Data',
    output_dir='data',
    modality='flair',
    slice_range=(60, 120),
    min_tumor_pixels=100
):
    """
    将BraTS 3D数据转换为2D切片
    
    Args:
        brats_dir: BraTS数据集路径
        output_dir: 输出目录
        modality: MRI模态 ('t1', 't1ce', 't2', 'flair')
        slice_range: 提取的切片范围（轴向切片）
        min_tumor_pixels: 最小肿瘤像素数，用于过滤空白切片
    """
    
    print("=" * 60)
    print("BraTS2021 数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  输入目录: {brats_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  MRI模态: {modality}")
    print(f"  切片范围: {slice_range[0]}-{slice_range[1]}")
    print(f"  最小肿瘤像素: {min_tumor_pixels}")
    
    # 检查输入目录
    brats_path = Path(brats_dir)
    if not brats_path.exists():
        print(f"\n❌ 错误: 找不到BraTS数据目录: {brats_dir}")
        print("请检查路径是否正确")
        return
    
    # 获取所有患者目录
    patient_dirs = sorted([d for d in brats_path.iterdir() if d.is_dir()])
    
    if len(patient_dirs) == 0:
        print(f"\n❌ 错误: {brats_dir} 中没有找到患者数据")
        return
    
    print(f"\n找到 {len(patient_dirs)} 个患者数据")
    print("\n开始处理...")
    
    total_slices = 0
    skipped_patients = 0
    
    for patient_dir in tqdm(patient_dirs, desc='处理患者数据'):
        patient_id = patient_dir.name
        
        # 构建文件路径
        img_file = patient_dir / f'{patient_id}_{modality}.nii.gz'
        seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
        
        # 检查文件是否存在
        if not img_file.exists():
            tqdm.write(f'⚠️  警告: {patient_id} 缺少 {modality} 文件，跳过')
            skipped_patients += 1
            continue
            
        if not seg_file.exists():
            tqdm.write(f'⚠️  警告: {patient_id} 缺少分割文件，跳过')
            skipped_patients += 1
            continue
        
        try:
            # 加载NIfTI文件
            img_nii = nib.load(str(img_file))
            seg_nii = nib.load(str(seg_file))
            
            img_data = img_nii.get_fdata()
            seg_data = seg_nii.get_fdata()
            
            # 检查数据形状
            if img_data.shape != seg_data.shape:
                tqdm.write(f'⚠️  警告: {patient_id} 图像和掩码形状不匹配，跳过')
                skipped_patients += 1
                continue
            
            # 提取指定范围的切片
            patient_slices = 0
            for slice_idx in range(slice_range[0], min(slice_range[1], img_data.shape[2])):
                # 提取2D切片
                img_slice = img_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]
                
                # 过滤空白切片（没有肿瘤的切片）
                tumor_pixels = np.sum(seg_slice > 0)
                if tumor_pixels < min_tumor_pixels:
                    continue
                
                # 归一化图像
                img_normalized = normalize_image(img_slice)
                
                # 简化标注（将BraTS的1,2,4标签映射为1,2,3）
                # BraTS标签: 0=背景, 1=坏死核心, 2=水肿, 4=增强肿瘤
                # 映射后: 0=背景, 1=坏死核心, 2=水肿, 3=增强肿瘤
                seg_simplified = np.zeros_like(seg_slice, dtype=np.uint8)
                seg_simplified[seg_slice == 1] = 1  # 坏死核心
                seg_simplified[seg_slice == 2] = 2  # 水肿
                seg_simplified[seg_slice == 4] = 3  # 增强肿瘤
                
                # 生成文件名
                slice_name = f'{patient_id}_slice{slice_idx:03d}'
                
                # 保存图像（转换为RGB）
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
                Image.fromarray(img_rgb).save(img_dir / f'{slice_name}.png')
                
                # 保存掩码
                Image.fromarray(seg_simplified).save(mask_dir / f'{slice_name}.png')
                
                patient_slices += 1
                total_slices += 1
            
            if patient_slices > 0:
                tqdm.write(f'✓ {patient_id}: 生成 {patient_slices} 个切片')
                
        except Exception as e:
            tqdm.write(f'❌ 错误: 处理 {patient_id} 时出错: {str(e)}')
            skipped_patients += 1
            continue
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"处理的患者数: {len(patient_dirs) - skipped_patients}/{len(patient_dirs)}")
    print(f"跳过的患者数: {skipped_patients}")
    print(f"生成的切片总数: {total_slices}")
    print(f"\n输出目录:")
    print(f"  图像: {img_dir.absolute()}")
    print(f"  掩码: {mask_dir.absolute()}")
    
    if total_slices == 0:
        print("\n⚠️  警告: 没有生成任何切片！")
        print("可能的原因:")
        print("  1. min_tumor_pixels 阈值太高")
        print("  2. slice_range 范围不合适")
        print("  3. 数据格式不正确")
    else:
        print(f"\n✅ 成功！可以开始训练了")
        print(f"\n训练命令示例:")
        print(f"python train.py --epochs 80 --batch-size 8 --learning-rate 1e-4 --classes 4 --amp")


def process_brats_multimodal(
    brats_dir='BraTS2021/BraTS2021_Training_Data',
    output_dir='data',
    modalities=['t1', 't2', 'flair'],
    slice_range=(60, 120),
    min_tumor_pixels=100
):
    """
    使用多模态MRI作为RGB通道
    
    Args:
        brats_dir: BraTS数据集路径
        output_dir: 输出目录
        modalities: MRI模态列表（必须是3个）
        slice_range: 提取的切片范围
        min_tumor_pixels: 最小肿瘤像素数
    """
    
    if len(modalities) != 3:
        print("错误: 多模态模式需要恰好3个模态（对应RGB三通道）")
        return
    
    print("=" * 60)
    print("BraTS2021 多模态数据预处理")
    print("=" * 60)
    
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  输入目录: {brats_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  MRI模态: {modalities}")
    print(f"  切片范围: {slice_range[0]}-{slice_range[1]}")
    print(f"  最小肿瘤像素: {min_tumor_pixels}")
    
    brats_path = Path(brats_dir)
    if not brats_path.exists():
        print(f"\n❌ 错误: 找不到BraTS数据目录: {brats_dir}")
        return
    
    patient_dirs = sorted([d for d in brats_path.iterdir() if d.is_dir()])
    
    if len(patient_dirs) == 0:
        print(f"\n❌ 错误: {brats_dir} 中没有找到患者数据")
        return
    
    print(f"\n找到 {len(patient_dirs)} 个患者数据")
    print("\n开始处理...")
    
    total_slices = 0
    skipped_patients = 0
    
    for patient_dir in tqdm(patient_dirs, desc='处理患者数据'):
        patient_id = patient_dir.name
        
        # 读取多个模态
        modality_data = []
        all_files_exist = True
        
        for mod in modalities:
            img_file = patient_dir / f'{patient_id}_{mod}.nii.gz'
            if not img_file.exists():
                tqdm.write(f'⚠️  警告: {patient_id} 缺少 {mod} 模态，跳过')
                all_files_exist = False
                break
            
            try:
                img_nii = nib.load(str(img_file))
                modality_data.append(img_nii.get_fdata())
            except Exception as e:
                tqdm.write(f'❌ 错误: 读取 {patient_id} 的 {mod} 模态失败: {str(e)}')
                all_files_exist = False
                break
        
        if not all_files_exist:
            skipped_patients += 1
            continue
        
        # 读取分割标注
        seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
        if not seg_file.exists():
            tqdm.write(f'⚠️  警告: {patient_id} 缺少分割文件，跳过')
            skipped_patients += 1
            continue
        
        try:
            seg_data = nib.load(str(seg_file)).get_fdata()
            
            # 提取切片
            patient_slices = 0
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
                
                patient_slices += 1
                total_slices += 1
            
            if patient_slices > 0:
                tqdm.write(f'✓ {patient_id}: 生成 {patient_slices} 个切片')
                
        except Exception as e:
            tqdm.write(f'❌ 错误: 处理 {patient_id} 时出错: {str(e)}')
            skipped_patients += 1
            continue
    
    print("\n" + "=" * 60)
    print("多模态预处理完成！")
    print("=" * 60)
    print(f"处理的患者数: {len(patient_dirs) - skipped_patients}/{len(patient_dirs)}")
    print(f"跳过的患者数: {skipped_patients}")
    print(f"生成的切片总数: {total_slices}")
    print(f"\n输出目录:")
    print(f"  图像: {img_dir.absolute()}")
    print(f"  掩码: {mask_dir.absolute()}")
    
    if total_slices > 0:
        print(f"\n✅ 成功！可以开始训练了")
        print(f"\n训练命令示例:")
        print(f"python train.py --epochs 80 --batch-size 8 --learning-rate 1e-4 --classes 4 --amp")


if __name__ == '__main__':
    # ========================================
    # 配置参数（根据实际情况修改）
    # ========================================
    
    # BraTS数据集路径（修改为你的实际路径）
    BRATS_DIR = 'BraTS2021/BraTS2021_Training_Data'
    
    # 输出目录
    OUTPUT_DIR = 'data'
    
    # 选择处理模式
    USE_MULTIMODAL = False  # True: 多模态融合, False: 单模态
    
    # 单模态配置
    MODALITY = 'flair'  # 可选: 't1', 't1ce', 't2', 'flair'
    
    # 多模态配置（仅当USE_MULTIMODAL=True时使用）
    MODALITIES = ['t1', 't2', 'flair']  # 必须是3个模态
    
    # 切片范围（轴向切片索引）
    # BraTS数据的z轴范围是0-154，中间部分通常包含更多肿瘤信息
    SLICE_RANGE = (60, 120)  # 提取第60-120个切片
    
    # 最小肿瘤像素数（用于过滤空白切片）
    # 降低此值可以保留更多切片，但可能包含较少肿瘤的切片
    MIN_TUMOR_PIXELS = 100
    
    # ========================================
    # 执行预处理
    # ========================================
    
    if USE_MULTIMODAL:
        print("使用多模态融合模式")
        process_brats_multimodal(
            brats_dir=BRATS_DIR,
            output_dir=OUTPUT_DIR,
            modalities=MODALITIES,
            slice_range=SLICE_RANGE,
            min_tumor_pixels=MIN_TUMOR_PIXELS
        )
    else:
        print("使用单模态模式")
        process_brats_to_2d(
            brats_dir=BRATS_DIR,
            output_dir=OUTPUT_DIR,
            modality=MODALITY,
            slice_range=SLICE_RANGE,
            min_tumor_pixels=MIN_TUMOR_PIXELS
        )
