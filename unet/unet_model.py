"""
UNet 完整网络架构实现

该模块实现了经典的UNet编码器-解码器分割网络架构。
UNet是一种专门用于图像分割的卷积神经网络，在医学图像分割领域表现优异。

网络特点：
1. 编码器-解码器结构：编码器提取特征，解码器恢复空间分辨率
2. 跳跃连接：将编码器特征直接传递到解码器，保留细节信息
3. 对称设计：编码器和解码器层数相同，结构对称
4. 多尺度特征：通过不同层级的特征融合提升分割精度

技术优势：
- 跳跃连接保留细节信息，提升边界分割精度
- 编码器-解码器结构适合像素级分类任务
- 对称设计便于理解和调试
- 可扩展性强，支持不同输入尺寸和类别数

适用场景：
- 医学图像分割（器官、病变区域等）
- 卫星图像分割（建筑、道路等）
- 生物图像分析（细胞分割等）
- 自动驾驶场景分割
- 工业检测和质量控制

网络结构：
输入 -> 编码器 -> 瓶颈层 -> 解码器 -> 输出
  |      |         |         |        |
  |      |         |         |        v
  |      |         |         |    分类头
  |      |         |         |        |
  |      |         |         |        v
  |      |         |         |    分割掩码
  |      |         |         |
  |      |         |         v
  |      |         |    上采样+特征融合
  |      |         |
  |      |         v
  |      |     瓶颈层(最深层)
  |      |
  |      v
  |   下采样+特征提取
  |
  v
输入图像

作者：Ronneberger et al. (2015)
论文：U-Net: Convolutional Networks for Biomedical Image Segmentation

性能优化：
- 支持梯度检查点机制减少显存占用
- 支持channels_last内存格式优化GPU性能
- 支持混合精度训练加速推理
- 灵活的上采样方式选择（转置卷积/双线性插值）
"""

# ================================
# 导入必要的模块
# ================================
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    """
    UNet 编码器-解码器分割网络
    
    该类实现了经典的UNet架构，专门用于图像分割任务。
    采用编码器-解码器结构，通过跳跃连接保留细节信息。
    
    ================================
    网络架构说明：
    ================================
    编码器部分（下采样路径）：
    - inc: 输入卷积层 (n_channels -> 64)
    - down1: 下采样层1 (64 -> 128)，特征图尺寸减半
    - down2: 下采样层2 (128 -> 256)，特征图尺寸减半
    - down3: 下采样层3 (256 -> 512)，特征图尺寸减半
    - down4: 下采样层4 (512 -> 1024)，特征图尺寸减半
    
    解码器部分（上采样路径）：
    - up1: 上采样层1 (1024 -> 512)，特征图尺寸翻倍
    - up2: 上采样层2 (512 -> 256)，特征图尺寸翻倍
    - up3: 上采样层3 (256 -> 128)，特征图尺寸翻倍
    - up4: 上采样层4 (128 -> 64)，特征图尺寸翻倍
    - outc: 输出卷积层 (64 -> n_classes)
    
    跳跃连接机制：
    - 编码器每层的特征图直接传递到对应的解码器层
    - 通过特征拼接的方式融合多尺度信息
    - 保留细节信息，提升边界分割精度
    
    ================================
    参数说明：
    ================================
    Args:
        n_channels (int): 输入图像通道数
            - 1: 灰度图像（医学影像常用）
            - 3: RGB彩色图像（自然图像）
            - 4: RGBA图像（包含透明度通道）
            - 其他: 多光谱图像或特殊应用
        
        n_classes (int): 分割类别数
            - 1: 二分类任务（使用sigmoid激活）
            - 2+: 多分类任务（使用softmax激活）
            - 包括背景类在内的总类别数
        
        bilinear (bool, optional): 上采样方式选择. 默认值: False
            - True: 使用双线性插值上采样
                * 优点: 参数量少，内存占用小，计算稳定
                * 缺点: 可能丢失细节信息，分割精度略低
            - False: 使用转置卷积上采样
                * 优点: 可学习参数，分割精度更高
                * 缺点: 参数量多，内存占用大
    
    ================================
    网络特点和优势：
    ================================
    1. 对称U型结构：
       - 编码器和解码器层数相同，结构对称
       - 每层特征图尺寸严格对应，便于跳跃连接
       - 网络深度适中，避免梯度消失问题
    
    2. 跳跃连接机制：
       - 将编码器的高分辨率特征直接传递到解码器
       - 保留空间细节信息，提升边界分割精度
       - 缓解深层网络的梯度消失问题
       - 实现多尺度特征融合
    
    3. 多尺度特征提取：
       - 不同层级提取不同感受野的特征
       - 浅层特征保留细节，深层特征提取语义
       - 全局和局部信息有效结合
    
    4. 灵活的配置选项：
       - 支持任意输入图像尺寸（建议2的幂次方）
       - 支持不同通道数的输入图像
       - 支持不同数量的分割类别
       - 支持不同的上采样策略
    
    ================================
    使用示例：
    ================================
    ```python
    # 医学图像二分类分割（病灶检测）
    model = UNet(n_channels=1, n_classes=1, bilinear=False)
    
    # 自然图像多分类分割（语义分割）
    model = UNet(n_channels=3, n_classes=21, bilinear=True)
    
    # 前向传播示例
    input_tensor = torch.randn(4, 3, 256, 256)  # 批次大小4
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    ```
    
    ================================
    性能优化建议：
    ================================
    1. 输入尺寸优化：
       - 使用2的幂次方尺寸（64, 128, 256, 512, 1024）
       - 避免奇数尺寸，可能导致上采样时尺寸不匹配
       - 根据显存大小选择合适的输入尺寸
    
    2. 内存优化策略：
       - 大图像训练时启用梯度检查点
       - 使用混合精度训练减少显存占用
       - 适当减小批次大小
       - 使用channels_last内存格式
    
    3. 训练优化技巧：
       - 使用数据增强提升模型泛化能力
       - 采用学习率调度器优化收敛过程
       - 使用权重初始化策略
       - 考虑使用预训练编码器
    
    4. 推理优化：
       - 使用torch.jit.script编译模型
       - 启用CUDA图优化
       - 考虑模型量化和剪枝
    
    ================================
    注意事项和限制：
    ================================
    1. 硬件要求：
       - 建议使用GPU进行训练和推理
       - 大模型需要足够的显存支持
       - CPU推理速度较慢，适合小模型
    
    2. 数据要求：
       - 输入图像和掩码必须尺寸匹配
       - 掩码标签值应在合理范围内
       - 建议进行数据预处理和归一化
    
    3. 模型限制：
       - 固定的网络深度，不适合极大或极小图像
       - 跳跃连接要求编码器和解码器特征图尺寸匹配
       - 内存占用随输入尺寸平方增长
    """
    
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        """
        初始化UNet网络
        
        该构造函数创建UNet网络的所有组件，包括编码器、解码器和输出层。
        网络采用对称的U型结构，通过跳跃连接实现多尺度特征融合。
        
        Args:
            n_channels (int): 输入图像通道数
                - 必须为正整数
                - 常见值: 1(灰度), 3(RGB), 4(RGBA)
            n_classes (int): 分割类别数
                - 必须为正整数
                - 1表示二分类，>1表示多分类
                - 包括背景类在内的总类别数
            bilinear (bool, optional): 上采样方式. 默认值: False
                - True: 双线性插值，参数少但精度略低
                - False: 转置卷积，参数多但精度更高
        
        Raises:
            ValueError: 当n_channels或n_classes不是正整数时
            
        Note:
            - 双线性插值模式下，特征通道数会减半以保持参数量平衡
            - 网络深度固定为5层，适合大多数分割任务
            - 所有卷积层都使用ReLU激活和批归一化
        """
        super(UNet, self).__init__()
        
        # ================================
        # 参数验证
        # ================================
        if not isinstance(n_channels, int) or n_channels <= 0:
            raise ValueError(f"n_channels必须是正整数，得到: {n_channels}")
        if not isinstance(n_classes, int) or n_classes <= 0:
            raise ValueError(f"n_classes必须是正整数，得到: {n_classes}")
        
        # ================================
        # 网络配置参数
        # ================================
        self.n_channels = n_channels      # 输入通道数
        self.n_classes = n_classes        # 输出类别数
        self.bilinear = bilinear          # 上采样方式
        self.checkpointing = False        # 梯度检查点标志
        
        # 双线性插值模式下的通道数调整因子
        # 用于减少参数量，保持模型大小平衡
        factor = 2 if bilinear else 1
        
        # ================================
        # 编码器部分（下采样路径）
        # ================================
        # 输入卷积层：将输入图像转换为64通道特征图
        # 使用两个3x3卷积层，保持空间尺寸不变
        # 这是网络的第一层，负责初始特征提取
        self.inc = DoubleConv(n_channels, 64)
        
        # 下采样层1：64 -> 128通道，空间尺寸减半
        # 包含2x2最大池化和双卷积操作
        # 开始提取更高级的特征表示
        self.down1 = Down(64, 128)
        
        # 下采样层2：128 -> 256通道，空间尺寸减半
        # 继续增加特征通道数，减小空间分辨率
        self.down2 = Down(128, 256)
        
        # 下采样层3：256 -> 512通道，空间尺寸减半
        # 提取更抽象的语义特征
        self.down3 = Down(256, 512)
        
        # 下采样层4：512 -> 1024通道，空间尺寸减半
        # 这是瓶颈层，特征图尺寸最小，感受野最大
        # 在双线性模式下通道数减半以控制参数量
        self.down4 = Down(512, 1024 // factor)
        
        # ================================
        # 解码器部分（上采样路径）
        # ================================
        # 上采样层1：1024 -> 512通道，空间尺寸翻倍
        # 使用跳跃连接融合down3的特征（512通道）
        # 输入通道数为1024（瓶颈层）+ 512（跳跃连接）= 1536
        self.up1 = Up(1024, 512 // factor, bilinear)
        
        # 上采样层2：512 -> 256通道，空间尺寸翻倍
        # 使用跳跃连接融合down2的特征（256通道）
        self.up2 = Up(512, 256 // factor, bilinear)
        
        # 上采样层3：256 -> 128通道，空间尺寸翻倍
        # 使用跳跃连接融合down1的特征（128通道）
        self.up3 = Up(256, 128 // factor, bilinear)
        
        # 上采样层4：128 -> 64通道，空间尺寸翻倍
        # 使用跳跃连接融合inc的特征（64通道）
        # 恢复到原始输入尺寸
        self.up4 = Up(128, 64, bilinear)
        
        # ================================
        # 输出层
        # ================================
        # 输出卷积层：64 -> n_classes通道
        # 使用1x1卷积生成最终的分割掩码
        # 不使用激活函数，输出原始logits
        self.outc = OutConv(64, n_classes)
        
        # ================================
        # 网络初始化
        # ================================
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        
        使用He初始化方法初始化卷积层权重，使用常数初始化批归一化层。
        这有助于网络训练的稳定性和收敛速度。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化（适合ReLU激活函数）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_model_info(self) -> dict:
        """
        获取模型信息统计
        
        Returns:
            dict: 包含模型参数量、内存占用等信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算模型大小（MB）
        param_size = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': param_size,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'checkpointing': self.checkpointing
        }



    def use_checkpointing(self, enabled: bool = True):
        """
        启用或禁用梯度检查点机制以节省显存
        
        梯度检查点是一种内存优化技术，通过重新计算中间激活值来减少显存占用。
        这是一种时间换空间的策略，适用于大模型或显存不足的情况。
        
        ================================
        工作原理：
        ================================
        1. 前向传播时不保存中间激活值到内存
        2. 反向传播时重新计算需要的激活值
        3. 显存占用减少约50-70%，但计算时间增加约20-30%
        4. 对于深层网络效果更明显
        
        ================================
        性能影响：
        ================================
        优点：
        - 显著减少显存占用，允许训练更大的模型
        - 支持更大的批次大小
        - 支持更高分辨率的输入图像
        - 不影响模型精度和收敛性
        
        缺点：
        - 增加训练时间（通常20-30%）
        - 增加GPU计算负载
        - 某些操作可能不支持检查点
        
        ================================
        适用场景：
        ================================
        - 显存不足导致OOM错误时
        - 需要训练大分辨率图像时
        - 希望使用更大批次大小时
        - 模型参数量很大时
        
        ================================
        使用建议：
        ================================
        - 优先尝试其他优化方法（减小批次、降低分辨率）
        - 在确实需要时再启用检查点
        - 可以与混合精度训练结合使用
        - 推理时建议禁用以提升速度
        
        Args:
            enabled (bool, optional): 是否启用梯度检查点. 默认值: True
                - True: 启用检查点，减少显存占用
                - False: 禁用检查点，提升训练速度
        
        Example:
            ```python
            model = UNet(n_channels=3, n_classes=2)
            
            # 启用梯度检查点
            model.use_checkpointing(True)
            
            # 禁用梯度检查点
            model.use_checkpointing(False)
            
            # 检查当前状态
            print(f"检查点状态: {model.checkpointing}")
            ```
        
        Note:
            - 该设置会影响所有后续的前向传播
            - 可以在训练过程中动态切换
            - 推理时建议禁用以获得最佳性能
        """
        self.checkpointing = enabled
        status = "启用" if enabled else "禁用"
        memory_info = "显存占用将减少约50%" if enabled else "训练速度将提升约20%"
        logging.info(f'已{status}梯度检查点机制，{memory_info}')
    
    def estimate_memory_usage(self, input_shape: Tuple[int, int, int, int]) -> dict:
        """
        估算模型的显存使用量
        
        Args:
            input_shape: 输入张量形状 (batch_size, channels, height, width)
            
        Returns:
            dict: 包含各种显存使用估算的字典
        """
        batch_size, channels, height, width = input_shape
        
        # 估算激活值显存占用（假设float32，4字节）
        # 编码器路径的特征图尺寸
        activations_memory = 0
        h, w = height, width
        
        # 编码器各层激活值
        for out_channels in [64, 128, 256, 512, 1024]:
            activations_memory += batch_size * out_channels * h * w * 4
            h, w = h // 2, w // 2
        
        # 解码器各层激活值（大致相同）
        activations_memory *= 2
        
        # 模型参数显存占用
        param_memory = sum(p.numel() for p in self.parameters()) * 4
        
        # 梯度显存占用（与参数相同）
        grad_memory = param_memory
        
        # 优化器状态显存占用（Adam需要2倍参数量）
        optimizer_memory = param_memory * 2
        
        # 总显存占用
        total_memory = activations_memory + param_memory + grad_memory + optimizer_memory
        
        # 检查点模式下激活值显存减少
        if self.checkpointing:
            activations_memory *= 0.3  # 约减少70%
            total_memory = activations_memory + param_memory + grad_memory + optimizer_memory
        
        return {
            'activations_mb': activations_memory / (1024 * 1024),
            'parameters_mb': param_memory / (1024 * 1024),
            'gradients_mb': grad_memory / (1024 * 1024),
            'optimizer_mb': optimizer_memory / (1024 * 1024),
            'total_mb': total_memory / (1024 * 1024),
            'checkpointing': self.checkpointing
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        UNet前向传播函数
        
        该函数实现了UNet网络的完整前向传播过程，包括编码器、解码器和跳跃连接。
        支持梯度检查点机制，可以在显存不足时启用以节省内存。
        
        ================================
        前向传播流程详解：
        ================================
        1. 编码器路径（特征提取和下采样）：
           输入图像 (H×W) -> inc -> 64通道特征图 (H×W)
           64通道 (H×W) -> down1 -> 128通道 (H/2×W/2)
           128通道 (H/2×W/2) -> down2 -> 256通道 (H/4×W/4)
           256通道 (H/4×W/4) -> down3 -> 512通道 (H/8×W/8)
           512通道 (H/8×W/8) -> down4 -> 1024通道 (H/16×W/16)
        
        2. 解码器路径（特征融合和上采样）：
           1024通道 (H/16×W/16) + 512通道跳跃连接 -> up1 -> 512通道 (H/8×W/8)
           512通道 (H/8×W/8) + 256通道跳跃连接 -> up2 -> 256通道 (H/4×W/4)
           256通道 (H/4×W/4) + 128通道跳跃连接 -> up3 -> 128通道 (H/2×W/2)
           128通道 (H/2×W/2) + 64通道跳跃连接 -> up4 -> 64通道 (H×W)
        
        3. 输出层：
           64通道 (H×W) -> outc -> n_classes通道 (H×W)
        
        ================================
        参数说明：
        ================================
        Args:
            x (torch.Tensor): 输入图像张量
                - 形状: (batch_size, n_channels, height, width)
                - 数据类型: torch.float32 或 torch.float16（混合精度）
                - 值范围: 通常为[0, 1]（归一化后）或[-1, 1]
                - 要求: height和width建议为2的幂次方
        
        Returns:
            torch.Tensor: 分割预测结果（原始logits）
                - 形状: (batch_size, n_classes, height, width)
                - 数据类型: 与输入相同
                - 内容: 每个像素每个类别的原始分数（未经激活函数）
                - 后处理: 需要sigmoid（二分类）或softmax（多分类）
        
        Raises:
            RuntimeError: 当输入尺寸不符合要求时
            torch.cuda.OutOfMemoryError: 当显存不足时
        
        ================================
        跳跃连接机制详解：
        ================================
        跳跃连接是UNet的核心创新，实现多尺度特征融合：
        
        连接方式：
        - 编码器特征图通过拼接（concatenation）传递到解码器
        - 保留空间细节信息，补偿下采样造成的信息丢失
        - 每个解码器层都接收对应编码器层的特征
        
        特征融合过程：
        x1 (64通道, H×W) ────────────┐
                                    │
        x2 (128通道, H/2×W/2) ──────┼─┐
                                    │ │
        x3 (256通道, H/4×W/4) ──────┼─┼─┐
                                    │ │ │
        x4 (512通道, H/8×W/8) ──────┼─┼─┼─┐
                                    │ │ │ │
        x5 (1024通道, H/16×W/16) ───┼─┼─┼─┼─> up1 -> up2 -> up3 -> up4 -> 输出
                                    │ │ │ │     ↑     ↑     ↑     ↑
                                    │ │ │ └─────┘     │     │     │
                                    │ │ └─────────────┘     │     │
                                    │ └───────────────────────┘     │
                                    └─────────────────────────────────┘
        
        优势：
        - 保留细节信息，提升边界分割精度
        - 缓解深层网络的梯度消失问题
        - 实现多尺度特征融合
        - 提供丰富的上下文信息
        
        ================================
        梯度检查点机制：
        ================================
        当启用梯度检查点时：
        - 前向传播不保存中间激活值
        - 反向传播时重新计算激活值
        - 显存占用减少50-70%
        - 计算时间增加20-30%
        - 适用于大模型或高分辨率图像
        
        ================================
        使用示例：
        ================================
        ```python
        # 基础使用
        model = UNet(n_channels=3, n_classes=2)
        input_tensor = torch.randn(4, 3, 256, 256)
        output = model(input_tensor)
        print(f"输入: {input_tensor.shape}")
        print(f"输出: {output.shape}")
        
        # 启用梯度检查点（显存优化）
        model.use_checkpointing(True)
        output = model(input_tensor)
        
        # 后处理示例
        if model.n_classes == 1:
            # 二分类：使用sigmoid
            probs = torch.sigmoid(output)
            masks = (probs > 0.5).float()
        else:
            # 多分类：使用softmax
            probs = torch.softmax(output, dim=1)
            masks = torch.argmax(probs, dim=1)
        ```
        
        ================================
        性能优化建议：
        ================================
        1. 输入尺寸优化：
           - 使用2的幂次方尺寸（64, 128, 256, 512, 1024）
           - 避免奇数尺寸，可能导致上采样问题
           - 根据显存大小选择合适尺寸
        
        2. 内存优化：
           - 显存不足时启用梯度检查点
           - 使用混合精度训练（AMP）
           - 适当减小批次大小
           - 使用channels_last内存格式
        
        3. 计算优化：
           - 在GPU上运行以获得最佳性能
           - 使用torch.jit.script编译模型
           - 考虑使用TensorRT等推理优化工具
        
        ================================
        注意事项：
        ================================
        1. 输入要求：
           - 确保输入张量在正确的设备上
           - 检查输入通道数与模型匹配
           - 验证输入数据类型和值范围
        
        2. 输出处理：
           - 输出是原始logits，需要激活函数处理
           - 二分类使用sigmoid，多分类使用softmax
           - 注意处理批次维度和类别维度
        
        3. 错误处理：
           - 显存不足时考虑启用梯度检查点
           - 输入尺寸过大时考虑分块处理
           - 注意检查CUDA设备可用性
        """
        # ================================
        # 输入验证
        # ================================
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor，得到: {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"输入必须是4维张量 (N,C,H,W)，得到: {x.dim()}维")
        
        if x.size(1) != self.n_channels:
            raise ValueError(f"输入通道数不匹配，期望: {self.n_channels}，得到: {x.size(1)}")
        
        # 检查输入尺寸是否合理（建议最小64x64）
        height, width = x.size(2), x.size(3)
        if height < 64 or width < 64:
            logging.warning(f"输入尺寸较小 ({height}×{width})，可能影响分割效果，建议至少64×64")
        
        # ================================
        # 前向传播执行
        # ================================
        if self.checkpointing:
            # ================================
            # 使用梯度检查点的前向传播
            # ================================
            # 编码器路径：逐层下采样，提取多尺度特征
            # 使用checkpoint包装每个模块，节省显存
            x1 = torch.utils.checkpoint.checkpoint(self.inc, x, use_reentrant=False)
            x2 = torch.utils.checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
            x3 = torch.utils.checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
            x4 = torch.utils.checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
            x5 = torch.utils.checkpoint.checkpoint(self.down4, x4, use_reentrant=False)
            
            # 解码器路径：逐层上采样，融合编码器特征
            # 每个up层都接收两个输入：上一层输出和对应编码器层特征
            x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up2, x, x3, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up3, x, x2, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.up4, x, x1, use_reentrant=False)
            
            # 输出层：生成最终的分割掩码
            logits = torch.utils.checkpoint.checkpoint(self.outc, x, use_reentrant=False)
            
        else:
            # ================================
            # 正常的前向传播（无梯度检查点）
            # ================================
            # 编码器路径：逐层下采样，提取多尺度特征
            # 保存每层特征图用于跳跃连接
            x1 = self.inc(x)        # 输入卷积：n_channels -> 64通道
            x2 = self.down1(x1)     # 下采样1：64 -> 128通道，尺寸减半
            x3 = self.down2(x2)     # 下采样2：128 -> 256通道，尺寸减半
            x4 = self.down3(x3)     # 下采样3：256 -> 512通道，尺寸减半
            x5 = self.down4(x4)     # 下采样4：512 -> 1024通道，尺寸减半（瓶颈层）
            
            # 解码器路径：逐层上采样，融合编码器特征
            # 通过跳跃连接将编码器特征融合到解码器
            x = self.up1(x5, x4)    # 上采样1：融合x5和x4，输出512通道
            x = self.up2(x, x3)     # 上采样2：融合当前x和x3，输出256通道
            x = self.up3(x, x2)     # 上采样3：融合当前x和x2，输出128通道
            x = self.up4(x, x1)     # 上采样4：融合当前x和x1，输出64通道
            
            # 输出层：生成最终的分割掩码
            # 使用1x1卷积将64通道特征图转换为n_classes通道
            logits = self.outc(x)
        
        # ================================
        # 输出验证
        # ================================
        # 确保输出形状正确
        expected_shape = (x.size(0), self.n_classes, x.size(2), x.size(3))
        if logits.shape != expected_shape:
            raise RuntimeError(f"输出形状不正确，期望: {expected_shape}，得到: {logits.shape}")
        
        return logits