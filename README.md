# EmoDiff-Mamba: Dynamic Emotional Speech Synthesis

## 项目概述
基于扩散模型和Mamba架构的情感可控文本转语音系统，结合推理时强化学习实现动态情感语音合成。

## 情感类别
支持4种情感类别：
- **happy**: 快乐、兴奋、愉悦
- **sad**: 悲伤、忧郁、沮丧  
- **angry**: 愤怒、生气、恼火
- **anxious**: 焦虑、紧张、担忧

## 数据准备

### 1. 自动下载和整合公开数据集
```bash
# 下载和整合多个数据集
python prepare_dataset.py --datasets RAVDESS CREMA-D --step all

# 只下载数据集
python prepare_dataset.py --datasets RAVDESS CREMA-D --step download

# 只预处理数据集
python prepare_dataset.py --datasets RAVDESS CREMA-D --step preprocess

# 只整合数据集
python prepare_dataset.py --datasets RAVDESS CREMA-D --step integrate
```

### 2. 手动标注优化
```bash
# 创建标注界面
python prepare_dataset.py --step annotate

# 运行标注工具
cd data/integrated
streamlit run annotation_interface.py
```

### 3. 支持的数据集
- **RAVDESS**: 情感语音数据集（8种情感）
- **CREMA-D**: 多情感语音数据集（6种情感）
- **IEMOCAP**: 情感对话数据集（需要手动下载）
- **EmoDB**: 德语情感语音数据集（7种情感）

### 4. 数据整合策略
- 自动情感映射到4个目标类别
- 音频标准化（22050Hz采样率）
- 质量检查和过滤
- 统一的数据格式和元数据

## 核心创新点
1. **选择性状态空间情感感知机制** - 使用Mamba架构捕获情感的时间演化
2. **动态策略网络** - 在推理时自适应调整引导尺度
3. **逐步奖励机制** - 解决全局奖励的时间归因模糊问题

## 项目结构
```
Exper/
├── data/                   # 数据存储
│   ├── raw/               # 原始语音数据
│   ├── processed/         # 预处理后的数据
│   ├── features/          # 提取的特征
│   └── results/           # 实验结果
├── src/                   # 源代码
│   ├── models/            # 模型定义
│   │   ├── mamba/        # Mamba架构实现
│   │   ├── diffusion/    # 扩散模型
│   │   └── policy/       # 动态策略网络
│   ├── data_processing/   # 数据处理
│   ├── training/         # 训练脚本
│   ├── inference/        # 推理脚本
│   └── utils/            # 工具函数
├── configs/              # 配置文件
├── notebooks/            # 实验记录
├── logs/                # 训练日志
└── requirements.txt     # 依赖包
```

## 技术栈
- **深度学习**: PyTorch, PyTorch Lightning
- **语音处理**: librosa, torchaudio, espnet
- **扩散模型**: diffusers, accelerate
- **强化学习**: stable-baselines3, ray[rllib]
- **实验跟踪**: wandb, tensorboard

## 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据: 将语音数据放入 `data/raw/`
3. 运行预处理: `python src/data_processing/preprocess.py`
4. 训练模型: `python src/training/train_emodiff_mamba.py`
5. 运行推理: `python src/inference/inference.py`
