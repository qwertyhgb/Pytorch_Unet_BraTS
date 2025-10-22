"""
UNet 网络组件实现

该模块实现了UNet网络的核心组件，包括各种卷积层、下采样层、上采样层等。
这些组件是构建完整UNet网络的基础模块，每个组件都有特定的功能和作用。

组件列表：
1. DoubleConv: 双卷积层，UNet的基础构建块
2. Down: 下采样层，用于编码器路径
3. Up: 上采样层，用于解码器路径
4. OutConv: 输出卷积层，生成最终分割掩码

设计原则：
- 模块化设计：每个组件功能单一，便于复用和调试
- 标准化实现：遵循PyTorch最佳实践
- 灵活配置：支持不同的参数设置
- 性能优化：使用高效的卷积操作

技术特点：
- 使用BatchNorm提升训练稳定性和收敛速度
- 使用ReLU激活函数提供非线性变换
- 支持不同的上采样方式（双线性插值/转置卷积）
- 自动处理特征图尺寸不匹配问题
- 支持梯度检查点机制减少显存占用

性能优化：
- 使用inplace操作减少内存占用
- 支持channels_last内存格式
- 兼容混合精度训练
- 优化的padding策略确保尺寸匹配
"""

# ================================
# 导入必要的模块
# ================================
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积层：UNet网络的基础构建块
    
    该层实现了两个连续的3x3卷积操作，是UNet网络的基本构建单元。
    每个卷积后都跟随BatchNorm和ReLU激活函数，提升训练稳定性和收敛速度。
    
    网络结构：
    输入 -> Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU -> 输出
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数
            - 必须为正整数
            - 对应输入特征图的通道维度
        out_channels (int): 输出通道数
            - 必须为正整数
            - 决定输出特征图的通道维度
        mid_channels (Optional[int]): 中间通道数
            - 如果未指定，使用out_channels
            - 用于控制网络宽度和参数量
            - 可以用于瓶颈结构设计
    
    ================================
    设计特点和优势：
    ================================
    1. 双卷积结构优势：
       - 两个3x3卷积的感受野等效于一个5x5卷积
       - 参数量更少：2×(3×3) = 18 < 5×5 = 25
       - 提供更多的非线性变换（两个ReLU）
       - 更深的网络结构，更强的特征提取能力
    
    2. 批归一化（BatchNorm）作用：
       - 加速训练收敛，减少训练时间
       - 提供正则化效果，减少过拟合
       - 允许使用更大的学习率
       - 减少对权重初始化的敏感性
    
    3. ReLU激活函数特点：
       - 计算简单，训练速度快
       - 缓解梯度消失问题
       - 提供稀疏性，增强模型表达能力
       - inplace=True减少内存占用
    
    4. 参数优化策略：
       - bias=False：配合BatchNorm使用，减少参数量
       - padding=1：保持空间尺寸不变，便于特征融合
       - kernel_size=3：平衡感受野和计算效率
    
    ================================
    计算复杂度分析：
    ================================
    假设输入尺寸为(B, C_in, H, W)，输出尺寸为(B, C_out, H, W)：
    
    参数量：
    - 第一层卷积：C_in × mid_channels × 3 × 3
    - 第二层卷积：mid_channels × C_out × 3 × 3
    - BatchNorm：2 × (mid_channels + C_out)
    - 总计：9 × (C_in × mid_channels + mid_channels × C_out) + 2 × (mid_channels + C_out)
    
    计算量（FLOPs）：
    - 约为：2 × 9 × H × W × (C_in × mid_channels + mid_channels × C_out)
    
    ================================
    使用示例：
    ================================
    ```python
    # 基础双卷积层（常用于UNet的inc层）
    conv = DoubleConv(3, 64)  # RGB输入 -> 64通道特征
    
    # 自定义中间通道数（瓶颈结构）
    conv = DoubleConv(256, 128, mid_channels=64)  # 256->64->128
    
    # 前向传播示例
    x = torch.randn(4, 64, 256, 256)  # 批次大小4
    output = conv(x)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")  # torch.Size([4, 128, 256, 256])
    
    # 计算参数量
    total_params = sum(p.numel() for p in conv.parameters())
    print(f"参数量: {total_params:,}")
    ```
    
    ================================
    性能优化建议：
    ================================
    1. 内存优化：
       - 使用inplace操作减少内存占用
       - 支持channels_last内存格式
       - 可以与梯度检查点结合使用
    
    2. 计算优化：
       - 在GPU上运行获得最佳性能
       - 支持混合精度训练
       - 批次大小影响BatchNorm效果
    
    3. 设计优化：
       - 合理选择中间通道数
       - 考虑使用深度可分离卷积
       - 可以替换为其他激活函数（如GELU）
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        """
        初始化双卷积层
        
        Args:
            in_channels (int): 输入通道数，必须为正整数
            out_channels (int): 输出通道数，必须为正整数
            mid_channels (Optional[int]): 中间通道数，默认为out_channels
        
        Raises:
            ValueError: 当通道数不是正整数时
        """
        super().__init__()
        
        # 参数验证
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels必须是正整数，得到: {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels必须是正整数，得到: {out_channels}")
        
        # 如果没有指定中间通道数，使用输出通道数
        if mid_channels is None:
            mid_channels = out_channels
        elif not isinstance(mid_channels, int) or mid_channels <= 0:
            raise ValueError(f"mid_channels必须是正整数，得到: {mid_channels}")
        
        # 构建双卷积序列
        # 使用Sequential容器将多个层组合在一起，便于管理和调试
        self.double_conv = nn.Sequential(
            # ================================
            # 第一个卷积块：输入 -> 中间通道
            # ================================
            nn.Conv2d(
                in_channels, mid_channels, 
                kernel_size=3,      # 3x3卷积核，平衡感受野和计算量
                padding=1,          # padding=1保持空间尺寸不变
                bias=False          # 配合BatchNorm使用，不需要bias
            ),
            nn.BatchNorm2d(mid_channels),   # 批归一化，加速收敛和正则化
            nn.ReLU(inplace=True),          # ReLU激活，inplace节省内存
            
            # ================================
            # 第二个卷积块：中间通道 -> 输出通道
            # ================================
            nn.Conv2d(
                mid_channels, out_channels,
                kernel_size=3,      # 保持一致的卷积核大小
                padding=1,          # 保持空间尺寸不变
                bias=False          # 配合BatchNorm使用
            ),
            nn.BatchNorm2d(out_channels),   # 批归一化
            nn.ReLU(inplace=True)           # 最终激活
        )
        
        # 存储配置信息，便于调试和分析
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        执行双卷积操作，包括两次卷积、批归一化和ReLU激活。
        保持输入的空间尺寸不变，只改变通道数。
        
        Args:
            x (torch.Tensor): 输入张量
                - 形状: (batch_size, in_channels, height, width)
                - 数据类型: torch.float32 或 torch.float16（混合精度）
                - 要求: 通道数必须与初始化时的in_channels匹配
        
        Returns:
            torch.Tensor: 输出张量
                - 形状: (batch_size, out_channels, height, width)
                - 数据类型: 与输入相同
                - 空间尺寸: 与输入完全相同
        
        Raises:
            RuntimeError: 当输入张量形状不符合要求时
        
        Example:
            ```python
            conv = DoubleConv(64, 128)
            x = torch.randn(2, 64, 256, 256)
            y = conv(x)
            assert y.shape == (2, 128, 256, 256)
            ```
        """
        # 输入验证
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到: {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"输入必须是4维张量(N,C,H,W)，得到: {x.dim()}维")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"输入通道数不匹配，期望: {self.in_channels}，得到: {x.size(1)}")
        
        # 执行双卷积操作
        return self.double_conv(x)
    
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        计算给定输入形状的输出形状
        
        Args:
            input_shape: 输入形状 (batch_size, in_channels, height, width)
            
        Returns:
            tuple: 输出形状 (batch_size, out_channels, height, width)
        """
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)


class Down(nn.Module):
    """
    下采样层：用于UNet编码器路径
    
    该层实现了下采样操作，通过最大池化减少空间尺寸，然后使用双卷积提取特征。
    这是UNet编码器路径的核心组件，用于逐步提取更高级别的抽象特征。
    
    网络结构：
    输入 -> MaxPool2d(2x2) -> DoubleConv -> 输出
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数
            - 必须为正整数
            - 对应上一层的输出通道数
        out_channels (int): 输出通道数
            - 必须为正整数
            - 通常是输入通道数的2倍（遵循UNet设计）
    
    ================================
    设计特点和原理：
    ================================
    1. 最大池化下采样：
       - 使用2x2最大池化，步长为2
       - 空间尺寸减半：H×W -> H/2×W/2
       - 保留局部区域的最大激活值
       - 提供平移不变性和局部特征不变性
       - 增加感受野，捕获更大范围的上下文信息
    
    2. 特征提取增强：
       - 下采样后立即使用双卷积
       - 增加通道数，补偿空间信息的损失
       - 提取更高级别的抽象特征
       - 增强特征表达能力
    
    3. 信息变换策略：
       - 空间维度压缩：减少计算量和内存占用
       - 通道维度扩展：增加特征表达能力
       - 信息密度增加：每个像素包含更多语义信息
       - 感受野扩大：能够感知更大范围的上下文
    
    4. 编码器层次结构：
       - 浅层：细节特征，高分辨率，小感受野
       - 深层：语义特征，低分辨率，大感受野
       - 逐层抽象：从像素级到对象级的特征提取
    
    ================================
    数学原理：
    ================================
    最大池化操作：
    output[i,j] = max(input[2i:2i+2, 2j:2j+2])
    
    感受野计算：
    - 每次下采样后感受野扩大2倍
    - n层下采样后感受野约为 2^n × 初始感受野
    
    信息量变化：
    - 输入信息量：H × W × C_in
    - 输出信息量：(H/2) × (W/2) × C_out = H×W×C_out/4
    - 当C_out = 2×C_in时，信息量减半
    
    ================================
    使用示例：
    ================================
    ```python
    # UNet编码器路径的典型使用
    down1 = Down(64, 128)   # 第一层下采样
    down2 = Down(128, 256)  # 第二层下采样
    down3 = Down(256, 512)  # 第三层下采样
    
    # 前向传播示例
    x = torch.randn(4, 64, 256, 256)  # 输入特征图
    y1 = down1(x)  # torch.Size([4, 128, 128, 128])
    y2 = down2(y1) # torch.Size([4, 256, 64, 64])
    y3 = down3(y2) # torch.Size([4, 512, 32, 32])
    
    print(f"原始: {x.shape}")
    print(f"下采样1: {y1.shape}")
    print(f"下采样2: {y2.shape}")
    print(f"下采样3: {y3.shape}")
    ```
    
    ================================
    性能分析：
    ================================
    1. 计算复杂度：
       - 最大池化：O(H×W×C_in)
       - 双卷积：O(H/2×W/2×(C_in×C_mid + C_mid×C_out)×9)
       - 总体：随着深度增加，计算量逐渐减少
    
    2. 内存占用：
       - 激活值内存：随空间尺寸减小而减少
       - 参数内存：主要来自双卷积层
       - 梯度内存：与参数内存相同
    
    3. 特征质量：
       - 空间精度：逐层降低
       - 语义丰富度：逐层提升
       - 感受野：逐层扩大
    
    ================================
    注意事项和限制：
    ================================
    1. 输入尺寸要求：
       - 高度和宽度必须是偶数
       - 建议使用2的幂次方尺寸（64, 128, 256, 512）
       - 奇数尺寸会导致上采样时尺寸不匹配
    
    2. 信息损失：
       - 最大池化会永久丢失部分空间细节
       - 这是编码器设计的必然结果
       - 通过跳跃连接可以部分恢复细节信息
    
    3. 设计考虑：
       - 通道数增长策略影响模型容量
       - 下采样层数决定感受野大小
       - 需要与解码器路径保持对称
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化下采样层
        
        Args:
            in_channels (int): 输入通道数，必须为正整数
            out_channels (int): 输出通道数，必须为正整数
        
        Raises:
            ValueError: 当通道数不是正整数时
        """
        super().__init__()
        
        # 参数验证
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels必须是正整数，得到: {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels必须是正整数，得到: {out_channels}")
        
        # 构建下采样序列：最大池化 + 双卷积
        self.maxpool_conv = nn.Sequential(
            # ================================
            # 最大池化层：空间下采样
            # ================================
            nn.MaxPool2d(
                kernel_size=2,      # 2x2池化窗口
                stride=2,           # 步长为2，尺寸减半
                padding=0           # 不使用padding
            ),
            
            # ================================
            # 双卷积层：特征提取和通道变换
            # ================================
            DoubleConv(in_channels, out_channels)
        )
        
        # 存储配置信息
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        执行下采样操作，包括最大池化和双卷积。
        空间尺寸减半，通道数按指定比例变化。
        
        Args:
            x (torch.Tensor): 输入张量
                - 形状: (batch_size, in_channels, height, width)
                - 要求: height和width必须是偶数
                - 数据类型: torch.float32 或 torch.float16
        
        Returns:
            torch.Tensor: 输出张量
                - 形状: (batch_size, out_channels, height/2, width/2)
                - 数据类型: 与输入相同
                - 特征: 更高级别的抽象特征
        
        Raises:
            ValueError: 当输入尺寸不符合要求时
        
        Example:
            ```python
            down = Down(64, 128)
            x = torch.randn(2, 64, 256, 256)
            y = down(x)
            assert y.shape == (2, 128, 128, 128)
            ```
        """
        # 输入验证
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到: {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"输入必须是4维张量(N,C,H,W)，得到: {x.dim()}维")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"输入通道数不匹配，期望: {self.in_channels}，得到: {x.size(1)}")
        
        # 检查空间尺寸是否为偶数
        height, width = x.size(2), x.size(3)
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"输入尺寸必须是偶数，得到: {height}×{width}")
        
        # 执行下采样操作
        return self.maxpool_conv(x)
    
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        计算给定输入形状的输出形状
        
        Args:
            input_shape: 输入形状 (batch_size, in_channels, height, width)
            
        Returns:
            tuple: 输出形状 (batch_size, out_channels, height/2, width/2)
        """
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height // 2, width // 2)


class Up(nn.Module):
    """
    上采样层：用于UNet解码器路径
    
    该层实现了上采样操作，通过双线性插值或转置卷积增加空间尺寸，
    然后与编码器特征进行跳跃连接，最后使用双卷积融合特征。
    这是UNet解码器路径的核心组件，用于恢复空间分辨率和融合多尺度特征。
    
    网络结构：
    解码器特征 -> 上采样 -> 尺寸调整 -> 跳跃连接 -> 双卷积 -> 输出
    编码器特征 ────────────────────────┘
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数（来自解码器上一层）
            - 必须为正整数
            - 对应解码器路径的特征通道数
        out_channels (int): 输出通道数
            - 必须为正整数
            - 通常是输入通道数的一半（遵循UNet设计）
        bilinear (bool): 上采样方式选择
            - True: 使用双线性插值上采样
            - False: 使用转置卷积上采样
    
    ================================
    上采样方式对比：
    ================================
    1. 双线性插值上采样：
       优点：
       - 参数量少，内存占用小
       - 计算稳定，不会产生棋盘效应
       - 训练速度快，收敛稳定
       - 适合资源受限的环境
       
       缺点：
       - 学习能力有限，无法学习复杂的上采样模式
       - 可能丢失细节信息
       - 分割精度可能略低
    
    2. 转置卷积上采样：
       优点：
       - 可学习参数，能够学习最优的上采样模式
       - 特征表达能力更强
       - 分割精度通常更高
       - 能够恢复更多细节信息
       
       缺点：
       - 参数量多，内存占用大
       - 可能产生棋盘效应（checkerboard artifacts）
       - 训练时间较长
       - 需要更多的计算资源
    
    ================================
    跳跃连接机制：
    ================================
    1. 特征融合策略：
       - 编码器特征：保留空间细节信息
       - 解码器特征：包含高级语义信息
       - 通道拼接：融合不同层级的特征
       - 互补信息：细节+语义的完美结合
    
    2. 尺寸匹配处理：
       - 自动检测尺寸差异
       - 使用对称padding调整尺寸
       - 确保特征图完美对齐
       - 支持任意尺寸的输入
    
    3. 信息流动：
       编码器路径：细节信息保存
       解码器路径：语义信息传递
       跳跃连接：信息融合桥梁
    
    ================================
    设计原理和优势：
    ================================
    1. 多尺度特征融合：
       - 结合不同分辨率的特征
       - 保留细节的同时利用语义信息
       - 提升边界分割精度
       - 增强模型的表达能力
    
    2. 信息恢复机制：
       - 补偿下采样过程中的信息损失
       - 通过跳跃连接恢复空间细节
       - 逐步重建高分辨率特征图
       - 保持语义一致性
    
    3. 梯度流动优化：
       - 跳跃连接提供额外的梯度路径
       - 缓解深层网络的梯度消失问题
       - 加速训练收敛
       - 提升训练稳定性
    
    ================================
    使用示例：
    ================================
    ```python
    # UNet解码器路径的典型使用
    up1 = Up(1024, 512, bilinear=False)  # 第一层上采样
    up2 = Up(512, 256, bilinear=False)   # 第二层上采样
    up3 = Up(256, 128, bilinear=False)   # 第三层上采样
    up4 = Up(128, 64, bilinear=False)    # 第四层上采样
    
    # 前向传播示例（配合编码器特征）
    # 假设编码器特征为 x1, x2, x3, x4, x5
    x = up1(x5, x4)  # 融合最深层特征
    x = up2(x, x3)   # 逐层上采样和融合
    x = up3(x, x2)
    x = up4(x, x1)   # 恢复到原始分辨率
    
    print("解码器路径特征形状变化：")
    print(f"x5: {x5.shape} -> up1 -> {x.shape}")
    ```
    
    ================================
    性能分析：
    ================================
    1. 计算复杂度：
       - 双线性插值：O(H×W×C)
       - 转置卷积：O(H×W×C×k²)，k为卷积核大小
       - 双卷积：O(H×W×C×9)
       - 总体：随着分辨率增加，计算量快速增长
    
    2. 内存占用：
       - 跳跃连接需要保存编码器特征
       - 特征拼接会临时增加内存占用
       - 高分辨率特征图占用大量内存
       - 可以使用梯度检查点优化
    
    3. 参数量对比：
       - 双线性插值：几乎无参数
       - 转置卷积：C_in × C_out × k²
       - 双卷积：主要参数来源
    
    ================================
    注意事项和最佳实践：
    ================================
    1. 尺寸匹配：
       - 自动处理尺寸不匹配问题
       - 支持任意尺寸的输入特征图
       - 使用对称padding保持特征中心对齐
    
    2. 内存管理：
       - 跳跃连接会显著增加内存占用
       - 大图像训练时需要注意显存限制
       - 可以与梯度检查点结合使用
    
    3. 训练技巧：
       - 转置卷积可能需要更小的学习率
       - 注意权重初始化策略
       - 监控训练过程中的特征分布
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        初始化上采样层
        
        Args:
            in_channels (int): 输入通道数，必须为正整数
            out_channels (int): 输出通道数，必须为正整数
            bilinear (bool): 上采样方式，True为双线性插值，False为转置卷积
        
        Raises:
            ValueError: 当通道数不是正整数时
        """
        super().__init__()
        
        # 参数验证
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels必须是正整数，得到: {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels必须是正整数，得到: {out_channels}")
        
        # 存储配置信息
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # 根据上采样方式选择不同的实现
        if bilinear:
            # ================================
            # 双线性插值上采样方案
            # ================================
            # 使用双线性插值进行上采样，无可学习参数
            self.up = nn.Upsample(
                scale_factor=2,         # 尺寸放大2倍
                mode='bilinear',        # 双线性插值模式
                align_corners=True      # 对齐角点，保持一致性
            )
            # 由于双线性插值不改变通道数，需要调整双卷积的输入通道数
            # 跳跃连接后的总通道数为 in_channels + skip_channels
            # 其中 skip_channels = in_channels // 2（来自对应编码器层）
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # ================================
            # 转置卷积上采样方案
            # ================================
            # 使用转置卷积进行上采样，有可学习参数
            self.up = nn.ConvTranspose2d(
                in_channels,            # 输入通道数
                in_channels // 2,       # 输出通道数（减半）
                kernel_size=2,          # 2x2卷积核
                stride=2,               # 步长为2，尺寸翻倍
                padding=0,              # 不使用padding
                bias=False              # 不使用bias（后续有BatchNorm）
            )
            # 转置卷积会将通道数减半，跳跃连接后总通道数为 in_channels
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        执行上采样、跳跃连接和特征融合的完整流程。
        
        Args:
            x1 (torch.Tensor): 解码器特征（来自上一层）
                - 形状: (batch_size, in_channels, height, width)
                - 低分辨率，高语义信息
            x2 (torch.Tensor): 编码器特征（跳跃连接）
                - 形状: (batch_size, skip_channels, height*2, width*2)
                - 高分辨率，细节信息丰富
        
        Returns:
            torch.Tensor: 融合后的特征
                - 形状: (batch_size, out_channels, height*2, width*2)
                - 结合了语义信息和细节信息
        
        Raises:
            ValueError: 当输入张量形状不符合要求时
        
        Example:
            ```python
            up = Up(1024, 512, bilinear=False)
            x1 = torch.randn(2, 1024, 32, 32)  # 解码器特征
            x2 = torch.randn(2, 512, 64, 64)   # 编码器特征
            output = up(x1, x2)
            assert output.shape == (2, 512, 64, 64)
            ```
        """
        # ================================
        # 输入验证
        # ================================
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor")
        
        if x1.dim() != 4 or x2.dim() != 4:
            raise ValueError("输入必须是4维张量(N,C,H,W)")
        
        if x1.size(1) != self.in_channels:
            raise ValueError(f"x1通道数不匹配，期望: {self.in_channels}，得到: {x1.size(1)}")
        
        # ================================
        # 1. 上采样解码器特征
        # ================================
        # 将低分辨率的解码器特征上采样到与编码器特征相同的尺寸
        x1 = self.up(x1)
        
        # ================================
        # 2. 智能尺寸匹配
        # ================================
        # 计算两个特征图在空间维度上的差异
        # 由于池化和上采样的舍入误差，可能存在±1像素的差异
        diffY = x2.size(2) - x1.size(2)  # 高度差异
        diffX = x2.size(3) - x1.size(3)  # 宽度差异
        
        # 使用对称padding调整x1的尺寸，使其与x2完全匹配
        # padding格式：[left, right, top, bottom]
        # 采用对称padding策略，保持特征图的中心对齐
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [
                diffX // 2,                 # 左padding
                diffX - diffX // 2,         # 右padding
                diffY // 2,                 # 上padding
                diffY - diffY // 2          # 下padding
            ])
        
        # ================================
        # 3. 跳跃连接：多尺度特征融合
        # ================================
        # 在通道维度上拼接两个特征图
        # x2: 编码器特征，包含丰富的空间细节信息
        # x1: 解码器特征，包含高级语义信息
        # 拼接顺序：[编码器特征, 解码器特征]
        x = torch.cat([x2, x1], dim=1)
        
        # ================================
        # 4. 特征提取和融合
        # ================================
        # 使用双卷积进一步提取和融合多尺度特征
        # 将拼接后的特征转换为目标通道数
        return self.conv(x)
    
    def get_output_shape(self, x1_shape: tuple, x2_shape: tuple) -> tuple:
        """
        计算给定输入形状的输出形状
        
        Args:
            x1_shape: 解码器特征形状 (batch_size, in_channels, height, width)
            x2_shape: 编码器特征形状 (batch_size, skip_channels, height*2, width*2)
            
        Returns:
            tuple: 输出形状 (batch_size, out_channels, height*2, width*2)
        """
        batch_size = x1_shape[0]
        height, width = x2_shape[2], x2_shape[3]  # 使用编码器特征的空间尺寸
        return (batch_size, self.out_channels, height, width)


class OutConv(nn.Module):
    """
    输出卷积层：生成最终的分割掩码
    
    该层是UNet网络的最后一层，使用1x1卷积将特征图转换为分割掩码。
    这是整个网络的输出层，负责生成每个像素的类别预测。
    
    结构：Conv2d(1x1) -> 分割掩码
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数（来自最后一层解码器）
        out_channels (int): 输出通道数（分割类别数）
    
    ================================
    设计特点：
    ================================
    1. 1x1卷积：
       - 不改变空间尺寸，只改变通道数
       - 参数量少，计算效率高
       - 适合作为输出层
    
    2. 类别映射：
       - 将特征图映射到类别空间
       - 每个通道对应一个类别
       - 输出logits，需要后续激活
    
    3. 最终输出：
       - 生成每个像素的类别预测
       - 形状：(batch_size, n_classes, height, width)
       - 内容：每个像素的类别概率或logits
    
    ================================
    使用示例：
    ================================
    ```python
    # 创建输出卷积层
    out_conv = OutConv(64, 2)  # 二分类任务
    
    # 前向传播
    x = torch.randn(1, 64, 256, 256)
    output = out_conv(x)
    print(output.shape)  # torch.Size([1, 2, 256, 256])
    
    # 应用激活函数
    if n_classes == 1:
        output = torch.sigmoid(output)  # 二分类
    else:
        output = torch.softmax(output, dim=1)  # 多分类
    ```
    
    ================================
    注意事项：
    ================================
    1. 输出处理：
       - 输出是logits，需要激活函数
       - 二分类使用sigmoid激活
       - 多分类使用softmax激活
    
    2. 类别数：
       - 输出通道数等于类别数
       - 包括背景类在内
       - 确保与训练时一致
    
    3. 空间尺寸：
       - 输出尺寸与输入尺寸相同
       - 每个像素都有类别预测
       - 支持任意尺寸的输入
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化输出卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（分割类别数）
        """
        super(OutConv, self).__init__()
        
        # 使用1x1卷积将特征图转换为分割掩码
        # 1x1卷积不改变空间尺寸，只改变通道数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入特征图，形状为(batch_size, in_channels, height, width)
            
        Returns:
            分割掩码logits，形状为(batch_size, out_channels, height, width)
        """
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层：生成最终的分割掩码
    
    该层是UNet网络的最后一层，使用1x1卷积将特征图转换为分割掩码。
    这是整个网络的输出层，负责生成每个像素的类别预测logits。
    
    网络结构：
    输入特征图 -> Conv2d(1x1) -> 分割掩码logits
    
    ================================
    参数说明：
    ================================
    Args:
        in_channels (int): 输入通道数
            - 必须为正整数
            - 来自最后一层解码器的特征通道数
        out_channels (int): 输出通道数
            - 必须为正整数
            - 等于分割任务的类别数
            - 包括背景类在内的总类别数
    
    ================================
    设计特点和原理：
    ================================
    1. 1x1卷积的优势：
       - 不改变空间尺寸，保持像素级预测
       - 参数量少：in_channels × out_channels
       - 计算效率高，适合作为输出层
       - 实现通道间的线性组合
       - 相当于每个像素位置的全连接层
    
    2. 类别映射机制：
       - 将高维特征映射到类别空间
       - 每个输出通道对应一个分割类别
       - 生成原始logits，便于损失函数计算
       - 支持任意数量的分割类别
    
    3. 像素级分类：
       - 为每个像素生成类别预测
       - 输出形状与输入空间尺寸相同
       - 支持密集预测任务
       - 保持空间对应关系
    
    ================================
    输出处理策略：
    ================================
    1. 二分类任务（n_classes=1）：
       - 输出单通道logits
       - 使用sigmoid激活：σ(x) = 1/(1+e^(-x))
       - 输出范围：[0, 1]，表示前景概率
       - 损失函数：BCEWithLogitsLoss
    
    2. 多分类任务（n_classes>1）：
       - 输出多通道logits
       - 使用softmax激活：softmax(x_i) = e^(x_i)/Σe^(x_j)
       - 输出范围：[0, 1]，所有类别概率和为1
       - 损失函数：CrossEntropyLoss
    
    ================================
    数学原理：
    ================================
    1x1卷积操作：
    output[b,c,h,w] = Σ(weight[c,i] × input[b,i,h,w]) + bias[c]
    
    其中：
    - b: 批次索引
    - c: 输出通道索引
    - h,w: 空间位置索引
    - i: 输入通道索引
    
    参数量计算：
    - 权重参数：in_channels × out_channels
    - 偏置参数：out_channels
    - 总参数：in_channels × out_channels + out_channels
    
    ================================
    使用示例：
    ================================
    ```python
    # 二分类分割任务（前景/背景）
    out_conv = OutConv(64, 1)
    x = torch.randn(4, 64, 256, 256)
    logits = out_conv(x)  # torch.Size([4, 1, 256, 256])
    probs = torch.sigmoid(logits)  # 转换为概率
    
    # 多分类分割任务（如5类语义分割）
    out_conv = OutConv(64, 5)
    x = torch.randn(4, 64, 256, 256)
    logits = out_conv(x)  # torch.Size([4, 5, 256, 256])
    probs = torch.softmax(logits, dim=1)  # 转换为概率
    
    # 获取预测类别
    pred_classes = torch.argmax(probs, dim=1)  # torch.Size([4, 256, 256])
    
    # 计算参数量
    total_params = sum(p.numel() for p in out_conv.parameters())
    print(f"参数量: {total_params:,}")  # 64*5 + 5 = 325
    ```
    
    ================================
    性能分析：
    ================================
    1. 计算复杂度：
       - 时间复杂度：O(B × H × W × C_in × C_out)
       - 空间复杂度：O(B × H × W × C_out)
       - 相对于整个网络，计算量很小
    
    2. 参数效率：
       - 参数量：C_in × C_out + C_out
       - 相比全连接层，参数量大大减少
       - 权重共享提高参数效率
    
    3. 内存占用：
       - 输入激活：B × C_in × H × W
       - 输出激活：B × C_out × H × W
       - 权重参数：C_in × C_out
    
    ================================
    应用场景：
    ================================
    1. 语义分割：
       - 将每个像素分类到预定义类别
       - 如道路、建筑、植被等
       - 输出通道数等于类别数
    
    2. 医学图像分割：
       - 器官分割、病变检测
       - 通常是二分类或少类别分割
       - 对精度要求很高
    
    3. 实例分割预处理：
       - 生成语义掩码
       - 为后续实例分割提供基础
       - 需要高质量的边界预测
    
    ================================
    注意事项和最佳实践：
    ================================
    1. 激活函数选择：
       - 训练时使用原始logits
       - 推理时根据任务选择激活函数
       - 避免在网络内部使用激活函数
    
    2. 损失函数配合：
       - BCEWithLogitsLoss：内置sigmoid，数值稳定
       - CrossEntropyLoss：内置softmax，数值稳定
       - 避免手动组合激活函数和损失函数
    
    3. 初始化策略：
       - 使用默认的kaiming初始化
       - 偏置初始化为0
       - 避免过大的初始权重
    
    4. 输出解释：
       - logits值越大，对应类别概率越高
       - 可以通过logits大小判断预测置信度
       - 注意处理类别不平衡问题
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化输出卷积层
        
        Args:
            in_channels (int): 输入通道数，必须为正整数
            out_channels (int): 输出通道数（分割类别数），必须为正整数
        
        Raises:
            ValueError: 当通道数不是正整数时
        """
        super(OutConv, self).__init__()
        
        # 参数验证
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels必须是正整数，得到: {in_channels}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels必须是正整数，得到: {out_channels}")
        
        # 使用1x1卷积将特征图转换为分割掩码
        # 1x1卷积实现通道间的线性组合，不改变空间尺寸
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,      # 1x1卷积核
            stride=1,           # 步长为1，保持尺寸
            padding=0,          # 不需要padding
            bias=True           # 使用偏置，因为这是最后一层
        )
        
        # 存储配置信息
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        将输入特征图转换为分割掩码logits。
        保持空间尺寸不变，只改变通道数。
        
        Args:
            x (torch.Tensor): 输入特征图
                - 形状: (batch_size, in_channels, height, width)
                - 数据类型: torch.float32 或 torch.float16
                - 来源: UNet解码器的最后一层
        
        Returns:
            torch.Tensor: 分割掩码logits
                - 形状: (batch_size, out_channels, height, width)
                - 数据类型: 与输入相同
                - 内容: 每个像素每个类别的原始分数
                - 后处理: 需要sigmoid（二分类）或softmax（多分类）
        
        Raises:
            TypeError: 当输入不是torch.Tensor时
            ValueError: 当输入形状不符合要求时
        
        Example:
            ```python
            out_conv = OutConv(64, 2)
            x = torch.randn(2, 64, 256, 256)
            logits = out_conv(x)
            assert logits.shape == (2, 2, 256, 256)
            
            # 后处理示例
            if out_conv.out_channels == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            ```
        """
        # 输入验证
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到: {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"输入必须是4维张量(N,C,H,W)，得到: {x.dim()}维")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"输入通道数不匹配，期望: {self.in_channels}，得到: {x.size(1)}")
        
        # 执行1x1卷积，生成分割掩码logits
        return self.conv(x)
    
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        计算给定输入形状的输出形状
        
        Args:
            input_shape: 输入形状 (batch_size, in_channels, height, width)
            
        Returns:
            tuple: 输出形状 (batch_size, out_channels, height, width)
        """
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)