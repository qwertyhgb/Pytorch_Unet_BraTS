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
7. 支持多进程并行处理

使用方法:
    python preprocess_brats.py
    python preprocess_brats.py --multimodal  # 多模态模式
    python preprocess_brats.py --workers 8   # 指定进程数

配置参数:
    BRATS_DIR: BraTS数据集路径
    OUTPUT_DIR: 输出目录
    MODALITY: MRI模态选择
    SLICE_RANGE: 提取的切片范围
    MIN_TUMOR_PIXELS: 最小肿瘤像素数
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 检查nibabel是否安装
try:
    import nibabel as nib
except ImportError:
    print("❌ 错误: 未安装nibabel库")
    print("请运行: pip install nibabel")
    sys.exit(1)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    归一化图像到0-255范围，使用更高效的向量化操作
    
    Args:
        img: 输入图像数组
        
    Returns:
        归一化后的uint8图像
    """
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    
    # 向量化操作，避免中间变量
    return ((img - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)


def simplify_segmentation(seg_slice: np.ndarray) -> np.ndarray:
    """
    简化BraTS分割标注
    
    BraTS标签: 0=背景, 1=坏死核心, 2=水肿, 4=增强肿瘤
    映射后: 0=背景, 1=坏死核心, 2=水肿, 3=增强肿瘤
    
    Args:
        seg_slice: 原始分割切片
        
    Returns:
        简化后的分割切片
    """
    seg_simplified = np.zeros_like(seg_slice, dtype=np.uint8)
    seg_simplified[seg_slice == 1] = 1
    seg_simplified[seg_slice == 2] = 2
    seg_simplified[seg_slice == 4] = 3
    return seg_simplified


def process_single_patient(
    patient_dir: Path,
    modality: str,
    slice_range: Tuple[int, int],
    min_tumor_pixels: int,
    img_dir: Path,
    mask_dir: Path
) -> Tuple[int, str]:
    """
    处理单个患者的数据（用于并行处理）
    
    Args:
        patient_dir: 患者数据目录
        modality: MRI模态
        slice_range: 切片范围
        min_tumor_pixels: 最小肿瘤像素数
        img_dir: 图像输出目录
        mask_dir: 掩码输出目录
        
    Returns:
        (生成的切片数, 状态信息)
    """
    patient_id = patient_dir.name
    img_file = patient_dir / f'{patient_id}_{modality}.nii.gz'
    seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
    
    # 检查文件存在性
    if not img_file.exists():
        return 0, f'⚠️  {patient_id}: 缺少 {modality} 文件'
    if not seg_file.exists():
        return 0, f'⚠️  {patient_id}: 缺少分割文件'
    
    try:
        # 加载数据
        img_data = nib.load(str(img_file)).get_fdata()
        seg_data = nib.load(str(seg_file)).get_fdata()
        
        if img_data.shape != seg_data.shape:
            return 0, f'⚠️  {patient_id}: 图像和掩码形状不匹配'
        
        # 处理切片
        patient_slices = 0
        end_slice = min(slice_range[1], img_data.shape[2])
        
        for slice_idx in range(slice_range[0], end_slice):
            seg_slice = seg_data[:, :, slice_idx]
            
            # 过滤空白切片
            if np.sum(seg_slice > 0) < min_tumor_pixels:
                continue
            
            # 处理图像
            img_slice = img_data[:, :, slice_idx]
            img_normalized = normalize_image(img_slice)
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
            
            # 简化标注
            seg_simplified = simplify_segmentation(seg_slice)
            
            # 保存
            slice_name = f'{patient_id}_slice{slice_idx:03d}'
            Image.fromarray(img_rgb).save(img_dir / f'{slice_name}.png')
            Image.fromarray(seg_simplified).save(mask_dir / f'{slice_name}.png')
            
            patient_slices += 1
        
        if patient_slices > 0:
            return patient_slices, f'✓ {patient_id}: {patient_slices} 个切片'
        return 0, f'  {patient_id}: 无有效切片'
        
    except Exception as e:
        return 0, f'❌ {patient_id}: {str(e)}'


def process_single_patient_multimodal(
    patient_dir: Path,
    modalities: List[str],
    slice_range: Tuple[int, int],
    min_tumor_pixels: int,
    img_dir: Path,
    mask_dir: Path
) -> Tuple[int, str]:
    """
    处理单个患者的多模态数据（用于并行处理）
    
    Args:
        patient_dir: 患者数据目录
        modalities: MRI模态列表
        slice_range: 切片范围
        min_tumor_pixels: 最小肿瘤像素数
        img_dir: 图像输出目录
        mask_dir: 掩码输出目录
        
    Returns:
        (生成的切片数, 状态信息)
    """
    patient_id = patient_dir.name
    
    # 加载多模态数据
    modality_data = []
    for mod in modalities:
        img_file = patient_dir / f'{patient_id}_{mod}.nii.gz'
        if not img_file.exists():
            return 0, f'⚠️  {patient_id}: 缺少 {mod} 模态'
        
        try:
            modality_data.append(nib.load(str(img_file)).get_fdata())
        except Exception as e:
            return 0, f'❌ {patient_id}: 读取 {mod} 失败'
    
    # 加载分割数据
    seg_file = patient_dir / f'{patient_id}_seg.nii.gz'
    if not seg_file.exists():
        return 0, f'⚠️  {patient_id}: 缺少分割文件'
    
    try:
        seg_data = nib.load(str(seg_file)).get_fdata()
        
        # 处理切片
        patient_slices = 0
        end_slice = min(slice_range[1], modality_data[0].shape[2])
        
        for slice_idx in range(slice_range[0], end_slice):
            seg_slice = seg_data[:, :, slice_idx]
            
            if np.sum(seg_slice > 0) < min_tumor_pixels:
                continue
            
            # 融合多模态为RGB
            rgb_channels = [normalize_image(mod_data[:, :, slice_idx]) 
                          for mod_data in modality_data]
            img_rgb = np.stack(rgb_channels, axis=-1)
            
            # 简化标注
            seg_simplified = simplify_segmentation(seg_slice)
            
            # 保存
            slice_name = f'{patient_id}_slice{slice_idx:03d}'
            Image.fromarray(img_rgb).save(img_dir / f'{slice_name}.png')
            Image.fromarray(seg_simplified).save(mask_dir / f'{slice_name}.png')
            
            patient_slices += 1
        
        if patient_slices > 0:
            return patient_slices, f'✓ {patient_id}: {patient_slices} 个切片'
        return 0, f'  {patient_id}: 无有效切片'
        
    except Exception as e:
        return 0, f'❌ {patient_id}: {str(e)}'


def process_brats_to_2d(
    brats_dir: str = 'BraTS2021/BraTS2021_Training_Data',
    output_dir: str = 'data',
    modality: str = 'flair',
    slice_range: Tuple[int, int] = (60, 120),
    min_tumor_pixels: int = 100,
    num_workers: Optional[int] = None
):
    """
    将BraTS 3D数据转换为2D切片（支持多进程并行）
    
    Args:
        brats_dir: BraTS数据集路径
        output_dir: 输出目录
        modality: MRI模态 ('t1', 't1ce', 't2', 'flair')
        slice_range: 提取的切片范围（轴向切片）
        min_tumor_pixels: 最小肿瘤像素数，用于过滤空白切片
        num_workers: 并行进程数，None表示自动检测
    """
    
    print("=" * 60)
    print("BraTS2021 单模态数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动检测进程数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"\n配置信息:")
    print(f"  输入目录: {brats_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  MRI模态: {modality}")
    print(f"  切片范围: {slice_range[0]}-{slice_range[1]}")
    print(f"  最小肿瘤像素: {min_tumor_pixels}")
    print(f"  并行进程数: {num_workers}")
    
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
    processed_patients = 0
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_patient,
                patient_dir, modality, slice_range, 
                min_tumor_pixels, img_dir, mask_dir
            ): patient_dir.name
            for patient_dir in patient_dirs
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(patient_dirs), desc='处理患者数据') as pbar:
            for future in as_completed(futures):
                slices, message = future.result()
                total_slices += slices
                if slices > 0:
                    processed_patients += 1
                    tqdm.write(message)
                elif '❌' in message or '⚠️' in message:
                    tqdm.write(message)
                pbar.update(1)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"处理的患者数: {processed_patients}/{len(patient_dirs)}")
    print(f"跳过的患者数: {len(patient_dirs) - processed_patients}")
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
    brats_dir: str = 'BraTS2021/BraTS2021_Training_Data',
    output_dir: str = 'data',
    modalities: List[str] = ['t1', 't2', 'flair'],
    slice_range: Tuple[int, int] = (60, 120),
    min_tumor_pixels: int = 100,
    num_workers: Optional[int] = None
):
    """
    使用多模态MRI作为RGB通道（支持多进程并行）
    
    Args:
        brats_dir: BraTS数据集路径
        output_dir: 输出目录
        modalities: MRI模态列表（必须是3个）
        slice_range: 提取的切片范围
        min_tumor_pixels: 最小肿瘤像素数
        num_workers: 并行进程数，None表示自动检测
    """
    
    if len(modalities) != 3:
        print("❌ 错误: 多模态模式需要恰好3个模态（对应RGB三通道）")
        return
    
    print("=" * 60)
    print("BraTS2021 多模态数据预处理")
    print("=" * 60)
    
    img_dir = Path(output_dir) / 'imgs'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动检测进程数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"\n配置信息:")
    print(f"  输入目录: {brats_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  MRI模态: {modalities}")
    print(f"  切片范围: {slice_range[0]}-{slice_range[1]}")
    print(f"  最小肿瘤像素: {min_tumor_pixels}")
    print(f"  并行进程数: {num_workers}")
    
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
    processed_patients = 0
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_patient_multimodal,
                patient_dir, modalities, slice_range,
                min_tumor_pixels, img_dir, mask_dir
            ): patient_dir.name
            for patient_dir in patient_dirs
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(patient_dirs), desc='处理患者数据') as pbar:
            for future in as_completed(futures):
                slices, message = future.result()
                total_slices += slices
                if slices > 0:
                    processed_patients += 1
                    tqdm.write(message)
                elif '❌' in message or '⚠️' in message:
                    tqdm.write(message)
                pbar.update(1)
    
    print("\n" + "=" * 60)
    print("多模态预处理完成！")
    print("=" * 60)
    print(f"处理的患者数: {processed_patients}/{len(patient_dirs)}")
    print(f"跳过的患者数: {len(patient_dirs) - processed_patients}")
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='BraTS2021数据集预处理脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--brats-dir', type=str, default=None,
                       help='BraTS数据集路径（自动检测如果未指定）')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='输出目录')
    parser.add_argument('--multimodal', action='store_true',
                       help='使用多模态融合模式')
    parser.add_argument('--modality', type=str, default='flair',
                       choices=['t1', 't1ce', 't2', 'flair'],
                       help='单模态MRI类型')
    parser.add_argument('--modalities', type=str, nargs=3, 
                       default=['t1', 't2', 'flair'],
                       help='多模态MRI类型（必须3个）')
    parser.add_argument('--slice-start', type=int, default=60,
                       help='切片起始索引')
    parser.add_argument('--slice-end', type=int, default=120,
                       help='切片结束索引')
    parser.add_argument('--min-tumor-pixels', type=int, default=100,
                       help='最小肿瘤像素数阈值')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数（默认自动检测）')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # ========================================
    # 自动检测或使用指定的数据路径
    # ========================================
    if args.brats_dir:
        BRATS_DIR = args.brats_dir
        if not os.path.exists(BRATS_DIR):
            print(f"❌ 错误: 指定的路径不存在: {BRATS_DIR}")
            sys.exit(1)
    else:
        # 自动检测可能的数据路径
        possible_paths = [
            'BraTS2021/BraTS2021_Training_Data',
            'BraTS2021',
            'BraTS2021_Training_Data'
        ]
        
        BRATS_DIR = None
        for path in possible_paths:
            if os.path.exists(path):
                BRATS_DIR = path
                print(f"✓ 找到BraTS数据路径: {path}")
                break
        
        if BRATS_DIR is None:
            print("❌ 错误: 未找到BraTS数据集")
            print("\n请执行以下步骤:")
            print("1. 下载数据: python download_data.py")
            print("2. 或手动指定路径: python preprocess_brats.py --brats-dir <路径>")
            sys.exit(1)
    
    # ========================================
    # 执行预处理
    # ========================================
    slice_range = (args.slice_start, args.slice_end)
    
    try:
        if args.multimodal:
            print("使用多模态融合模式")
            process_brats_multimodal(
                brats_dir=BRATS_DIR,
                output_dir=args.output_dir,
                modalities=args.modalities,
                slice_range=slice_range,
                min_tumor_pixels=args.min_tumor_pixels,
                num_workers=args.workers
            )
        else:
            print("使用单模态模式")
            process_brats_to_2d(
                brats_dir=BRATS_DIR,
                output_dir=args.output_dir,
                modality=args.modality,
                slice_range=slice_range,
                min_tumor_pixels=args.min_tumor_pixels,
                num_workers=args.workers
            )
    except KeyboardInterrupt:
        print("\n\n⏹️  处理被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
