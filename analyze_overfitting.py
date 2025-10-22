"""
分析训练过程中的过拟合情况
检查训练损失和验证损失的变化趋势
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_overfitting():
    """分析过拟合情况"""
    
    # 检查最佳模型
    best_model_path = "outputs/emodiff_mamba/simple_best_model.pth"
    final_model_path = "outputs/emodiff_mamba/simple_checkpoint_epoch_50.pth"
    
    print("分析训练过程中的过拟合情况...")
    
    # 加载最佳模型
    if Path(best_model_path).exists():
        best_checkpoint = torch.load(best_model_path, map_location='cpu')
        print(f"最佳模型加载成功")
        print(f"   最佳验证损失: {best_checkpoint['val_metrics']['val_loss']:.4f}")
        print(f"   最佳情感准确率: {best_checkpoint['val_metrics']['emotion_accuracy']:.4f}")
        print(f"   最佳epoch: {best_checkpoint['epoch'] + 1}")
        
        # 分析训练历史
        train_losses = best_checkpoint['train_losses']
        val_losses = best_checkpoint['val_losses']
        
        print(f"\n训练历史分析:")
        print(f"   训练轮数: {len(train_losses)}")
        print(f"   初始训练损失: {train_losses[0]:.4f}")
        print(f"   最终训练损失: {train_losses[-1]:.4f}")
        print(f"   初始验证损失: {val_losses[0]:.4f}")
        print(f"   最终验证损失: {val_losses[-1]:.4f}")
        
        # 计算过拟合指标
        train_loss_reduction = train_losses[0] - train_losses[-1]
        val_loss_reduction = val_losses[0] - val_losses[-1]
        overfitting_gap = val_losses[-1] - train_losses[-1]
        
        print(f"\n过拟合分析:")
        print(f"   训练损失下降: {train_loss_reduction:.4f}")
        print(f"   验证损失下降: {val_loss_reduction:.4f}")
        print(f"   训练-验证差距: {overfitting_gap:.4f}")
        
        # 判断过拟合程度
        if overfitting_gap > 0.1:
            print("   可能存在过拟合 (差距 > 0.1)")
        elif overfitting_gap > 0.05:
            print("   轻微过拟合 (差距 > 0.05)")
        else:
            print("   过拟合程度较低")
            
        # 检查情感准确率
        emotion_acc = best_checkpoint['val_metrics']['emotion_accuracy']
        if emotion_acc >= 0.99:
            print(f"   情感准确率过高 ({emotion_acc:.4f})，可能存在过拟合")
        elif emotion_acc >= 0.95:
            print(f"   情感准确率较高 ({emotion_acc:.4f})，但可接受")
        else:
            print(f"   情感准确率正常 ({emotion_acc:.4f})")
            
        # 绘制损失曲线
        plot_loss_curves(train_losses, val_losses)
        
        return {
            'overfitting_gap': overfitting_gap,
            'emotion_accuracy': emotion_acc,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    else:
        print("找不到最佳模型文件")
        return None

def plot_loss_curves(train_losses, val_losses):
    """绘制损失曲线"""
    try:
        plt.figure(figsize=(12, 5))
        
        # 训练和验证损失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失差距
        plt.subplot(1, 2, 2)
        gap = np.array(val_losses) - np.array(train_losses)
        plt.plot(epochs, gap, 'g-', label='验证-训练差距', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('损失差距')
        plt.title('过拟合程度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/emodiff_mamba/training_analysis.png', dpi=150, bbox_inches='tight')
        print("损失曲线已保存到: outputs/emodiff_mamba/training_analysis.png")
        
    except Exception as e:
        print(f"绘图失败: {e}")

def suggest_improvements(analysis_result):
    """建议改进措施"""
    if analysis_result is None:
        return
        
    print(f"\n改进建议:")
    
    if analysis_result['overfitting_gap'] > 0.1:
        print("   1. 增加正则化: 提高dropout率或添加L2正则化")
        print("   2. 减少模型复杂度: 降低隐藏层维度或层数")
        print("   3. 增加数据增强: 添加噪声、时间拉伸等")
        print("   4. 早停策略: 在验证损失不再下降时停止训练")
        
    if analysis_result['emotion_accuracy'] >= 0.99:
        print("   5. 情感分类器过拟合: 考虑简化分类器结构")
        print("   6. 数据平衡性: 检查各类别样本是否均衡")
        print("   7. 交叉验证: 使用k-fold验证更准确评估")
        
    print("   8. 增加数据集: 使用更多样化的情感语音数据")
    print("   9. 测试集评估: 在独立测试集上验证性能")

def main():
    """主函数"""
    print("=" * 60)
    print("EmoDiff-Mamba 过拟合分析")
    print("=" * 60)
    
    # 分析过拟合
    analysis_result = analyze_overfitting()
    
    # 提供改进建议
    suggest_improvements(analysis_result)
    
    print("\n" + "=" * 60)
    print("分析完成！")

if __name__ == "__main__":
    main()
