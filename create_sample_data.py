#!/usr/bin/env python3
"""
创建示例数据集用于测试EmoDiff-Mamba系统
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
import logging

def create_sample_dataset():
    """创建示例数据集"""
    
    # 创建目录
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("创建示例数据集...")
    
    # 情感标签
    emotions = ["happy", "sad", "angry", "anxious"]
    
    # 示例文本
    sample_texts = {
        "happy": [
            "I'm so excited about this new project!",
            "What a wonderful day it is today!",
            "I can't wait to see you again!"
        ],
        "sad": [
            "I feel so lonely and empty inside.",
            "Nothing seems to go right anymore.",
            "I miss the good old days."
        ],
        "angry": [
            "I can't believe you did that!",
            "This is absolutely unacceptable!",
            "I'm furious about this situation!"
        ],
        "anxious": [
            "I'm worried about what might happen.",
            "I feel nervous about the presentation.",
            "What if something goes wrong?"
        ]
    }
    
    data = []
    
    # 为每种情感创建示例数据
    for emotion in emotions:
        texts = sample_texts[emotion]
        
        for i, text in enumerate(texts):
            # 生成示例音频特征（mel频谱图）
            # 在实际应用中，这些会是真实的音频文件
            duration = 2.0  # 2秒
            sr = 22050
            n_mels = 80
            
            # 生成随机mel频谱图（模拟音频特征）
            mel_spec = np.random.randn(n_mels, int(duration * sr / 512))
            
            # 根据情感调整频谱图特征
            if emotion == "happy":
                mel_spec = mel_spec * 1.2 + 0.1  # 更亮的频谱
            elif emotion == "sad":
                mel_spec = mel_spec * 0.8 - 0.1  # 更暗的频谱
            elif emotion == "angry":
                mel_spec = mel_spec * 1.5 + 0.2  # 更强烈的频谱
            elif emotion == "anxious":
                mel_spec = mel_spec * 1.1 + np.random.randn(*mel_spec.shape) * 0.3  # 不稳定的频谱
            
            # 保存示例数据
            sample_id = f"{emotion}_{i:02d}"
            
            data.append({
                "sample_id": sample_id,
                "text": text,
                "emotion": emotion,
                "emotion_id": emotions.index(emotion),
                "duration": duration,
                "sample_rate": sr,
                "mel_shape": mel_spec.shape,
                "file_path": f"data/processed/{sample_id}.npy"
            })
            
            # 保存mel频谱图
            np.save(f"data/processed/{sample_id}.npy", mel_spec)
    
    # 创建元数据DataFrame
    df = pd.DataFrame(data)
    
    # 保存元数据
    df.to_csv("data/processed/metadata.csv", index=False)
    
    logger.info(f"示例数据集创建完成: {len(df)} 个样本")
    logger.info(f"情感分布:")
    for emotion in emotions:
        count = len(df[df["emotion"] == emotion])
        logger.info(f"  {emotion}: {count} 个样本")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sample_dataset()
