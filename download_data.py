import kagglehub
import os

# 指定下载目录
download_dir = "BraTS2021"  # 你想要的目录名

# 创建目录（如果不存在）
os.makedirs(download_dir, exist_ok=True)

print(f"正在下载 BraTS2021 数据集到: {os.path.abspath(download_dir)}")

# 下载到指定目录
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1", path=download_dir)

print("Path to dataset files:", path)

# 验证下载结果
if os.path.exists(path):
    files = os.listdir(path)
    print(f"\n下载完成！找到 {len(files)} 个文件/文件夹:")
    for f in files[:10]:  # 只显示前10个
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... 还有 {len(files) - 10} 个文件")
else:
    print("❌ 下载失败，路径不存在")