#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合训练和调优主脚本
整合K-fold交叉验证、超参数优化、模型集成等非数据方式的改进
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import logging
import argparse
from typing import Dict, List, Tuple

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入我们的模块
from data.ravdess_dataloader import RAVDESSDataset, RAVDESSDataModule
from train_improved_emodiff import DiffusionModel, EmotionClassifier, ImprovedEmoDiffTrainer
from kfold_cross_validation import KFoldCrossValidator
from hyperparameter_optimization import HyperparameterOptimizer
from model_ensemble import ModelEnsemble

class ComprehensiveTrainer:
    """综合训练器"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 output_dir: str = "outputs/comprehensive_training",
                 device: str = "cpu"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 存储所有结果
        self.results = {
            'kfold_cv': None,
            'hyperparameter_optimization': None,
            'model_ensemble': None,
            'final_evaluation': None
        }
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / f"comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_kfold_cross_validation(self, k_folds: int = 5, num_epochs: int = 20):
        """运行K-fold交叉验证"""
        self.logger.info("=" * 60)
        self.logger.info("Starting K-Fold Cross Validation")
        self.logger.info("=" * 60)
        
        cv = KFoldCrossValidator(
            data_dir=self.data_dir,
            output_dir=str(self.output_dir / "kfold_cv"),
            k_folds=k_folds,
            batch_size=8,
            num_epochs=num_epochs,
            device=self.device
        )
        
        cv.run_cross_validation()
        self.results['kfold_cv'] = cv.overall_results
        
        self.logger.info("K-Fold Cross Validation completed!")
        return cv.overall_results
        
    def run_hyperparameter_optimization(self, n_trials: int = 20):
        """运行超参数优化"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Hyperparameter Optimization")
        self.logger.info("=" * 60)
        
        optimizer = HyperparameterOptimizer(
            data_dir=self.data_dir,
            output_dir=str(self.output_dir / "hyperparameter_tuning"),
            n_trials=n_trials,
            device=self.device
        )
        
        optimizer.optimize()
        self.results['hyperparameter_optimization'] = {
            'best_trial_number': optimizer.best_trial.number,
            'best_value': optimizer.best_trial.value,
            'best_params': optimizer.best_trial.params
        }
        
        self.logger.info("Hyperparameter Optimization completed!")
        return optimizer.best_trial.params
        
    def run_model_ensemble(self, num_models: int = 3, num_epochs: int = 15):
        """运行模型集成"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Model Ensemble")
        self.logger.info("=" * 60)
        
        ensemble = ModelEnsemble(
            data_dir=self.data_dir,
            output_dir=str(self.output_dir / "model_ensemble"),
            device=self.device
        )
        
        # 创建多样化模型
        ensemble.create_diverse_models(num_models=num_models)
        
        # 训练各个模型
        ensemble.train_individual_models(num_epochs=num_epochs)
        
        # 评估集成模型
        test_loader = DataLoader(ensemble.test_dataset, batch_size=8, shuffle=False)
        ensemble_results = ensemble.evaluate_ensemble(test_loader)
        
        # 绘制结果
        ensemble.plot_ensemble_results(ensemble_results)
        
        self.results['model_ensemble'] = ensemble_results
        
        self.logger.info("Model Ensemble completed!")
        return ensemble_results
        
    def run_final_evaluation(self, best_params: Dict = None):
        """运行最终评估"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Final Evaluation")
        self.logger.info("=" * 60)
        
        # 使用最佳参数创建模型
        if best_params:
            diffusion_model = DiffusionModel(
                mel_channels=80,
                hidden_dim=best_params.get('hidden_dim_diffusion', 256),
                num_denoising_steps=1000
            )
            
            emotion_classifier = EmotionClassifier(
                input_dim=80,
                hidden_dim=best_params.get('hidden_dim_classifier', 128),
                num_emotions=4,
                dropout=best_params.get('dropout_rate', 0.2)
            )
        else:
            # 使用默认参数
            diffusion_model = DiffusionModel(
                mel_channels=80,
                hidden_dim=256,
                num_denoising_steps=1000
            )
            
            emotion_classifier = EmotionClassifier(
                input_dim=80,
                hidden_dim=128,
                num_emotions=4,
                dropout=0.2
            )
        
        # 创建数据模块
        data_module = RAVDESSDataModule(
            data_dir=self.data_dir,
            batch_size=8,
            num_workers=2,
            target_sr=22050,
            max_length=4,
            mel_channels=80
        )
        
        train_dataset, val_dataset, test_dataset = data_module.get_datasets()
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # 创建训练器
        trainer = ImprovedEmoDiffTrainer(
            diffusion_model=diffusion_model,
            emotion_classifier=emotion_classifier,
            device=self.device,
            output_dir=str(self.output_dir / "final_model")
        )
        
        # 训练最终模型
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=25
        )
        
        # 在测试集上评估
        test_results = trainer.evaluate(test_loader)
        
        self.results['final_evaluation'] = test_results
        
        self.logger.info("Final Evaluation completed!")
        return test_results
        
    def generate_comprehensive_report(self):
        """生成综合报告"""
        self.logger.info("=" * 60)
        self.logger.info("Generating Comprehensive Report")
        self.logger.info("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # 保存报告
        report_file = self.output_dir / "comprehensive_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 生成可视化报告
        self._plot_comprehensive_results()
        
        self.logger.info(f"Comprehensive report saved to {report_file}")
        
    def _generate_summary(self) -> Dict:
        """生成结果摘要"""
        summary = {}
        
        # K-fold交叉验证摘要
        if self.results['kfold_cv']:
            cv_results = self.results['kfold_cv']
            summary['kfold_cv'] = {
                'mean_val_loss': cv_results['mean_val_loss'],
                'std_val_loss': cv_results['std_val_loss'],
                'mean_val_accuracy': cv_results['mean_val_accuracy'],
                'std_val_accuracy': cv_results['std_val_accuracy']
            }
            
        # 超参数优化摘要
        if self.results['hyperparameter_optimization']:
            hp_results = self.results['hyperparameter_optimization']
            summary['hyperparameter_optimization'] = {
                'best_value': hp_results['best_value'],
                'best_params': hp_results['best_params']
            }
            
        # 模型集成摘要
        if self.results['model_ensemble']:
            ensemble_results = self.results['model_ensemble']
            best_method = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['accuracy'])
            summary['model_ensemble'] = {
                'best_method': best_method,
                'best_accuracy': ensemble_results[best_method]['accuracy']
            }
            
        # 最终评估摘要
        if self.results['final_evaluation']:
            final_results = self.results['final_evaluation']
            summary['final_evaluation'] = {
                'test_accuracy': final_results.get('accuracy', 0),
                'test_loss': final_results.get('loss', 0)
            }
            
        return summary
        
    def _plot_comprehensive_results(self):
        """绘制综合结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comprehensive Training Results', fontsize=16)
        
        # K-fold交叉验证结果
        ax1 = axes[0, 0]
        if self.results['kfold_cv']:
            cv_results = self.results['kfold_cv']
            methods = ['Mean Val Loss', 'Mean Val Accuracy']
            values = [cv_results['mean_val_loss'], cv_results['mean_val_accuracy']]
            errors = [cv_results['std_val_loss'], cv_results['std_val_accuracy']]
            
            bars = ax1.bar(methods, values, yerr=errors, capsize=5, 
                          color=['skyblue', 'lightgreen'], alpha=0.7)
            ax1.set_title('K-Fold Cross Validation')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'K-Fold CV\nNot Run', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('K-Fold Cross Validation')
            
        # 超参数优化结果
        ax2 = axes[0, 1]
        if self.results['hyperparameter_optimization']:
            hp_results = self.results['hyperparameter_optimization']
            ax2.text(0.5, 0.5, f"Best Value: {hp_results['best_value']:.4f}\n"
                               f"Best Trial: {hp_results['best_trial_number']}", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title('Hyperparameter Optimization')
        else:
            ax2.text(0.5, 0.5, 'Hyperparameter\nOptimization\nNot Run', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Hyperparameter Optimization')
            
        # 模型集成结果
        ax3 = axes[1, 0]
        if self.results['model_ensemble']:
            ensemble_results = self.results['model_ensemble']
            methods = list(ensemble_results.keys())
            accuracies = [ensemble_results[method]['accuracy'] for method in methods]
            
            bars = ax3.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax3.set_title('Model Ensemble')
            ax3.set_ylabel('Accuracy')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Model Ensemble\nNot Run', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Model Ensemble')
            
        # 最终评估结果
        ax4 = axes[1, 1]
        if self.results['final_evaluation']:
            final_results = self.results['final_evaluation']
            metrics = ['Test Accuracy', 'Test Loss']
            values = [final_results.get('accuracy', 0), final_results.get('loss', 0)]
            
            bars = ax4.bar(metrics, values, color=['lightgreen', 'lightcoral'])
            ax4.set_title('Final Evaluation')
            ax4.set_ylabel('Value')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Final Evaluation\nNot Run', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Final Evaluation')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comprehensive results plot saved to {self.output_dir / 'comprehensive_results.png'}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Comprehensive Training and Optimization')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/comprehensive_training', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of folds for cross validation')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--num_models', type=int, default=3, help='Number of models for ensemble')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--skip_kfold', action='store_true', help='Skip K-fold cross validation')
    parser.add_argument('--skip_hyperopt', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--skip_ensemble', action='store_true', help='Skip model ensemble')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comprehensive Training and Optimization for EmoDiff-Mamba")
    print("=" * 60)
    
    # 创建综合训练器
    trainer = ComprehensiveTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    best_params = None
    
    # 运行K-fold交叉验证
    if not args.skip_kfold:
        cv_results = trainer.run_kfold_cross_validation(
            k_folds=args.k_folds,
            num_epochs=args.num_epochs
        )
    
    # 运行超参数优化
    if not args.skip_hyperopt:
        best_params = trainer.run_hyperparameter_optimization(n_trials=args.n_trials)
    
    # 运行模型集成
    if not args.skip_ensemble:
        ensemble_results = trainer.run_model_ensemble(
            num_models=args.num_models,
            num_epochs=args.num_epochs
        )
    
    # 运行最终评估
    final_results = trainer.run_final_evaluation(best_params=best_params)
    
    # 生成综合报告
    trainer.generate_comprehensive_report()
    
    print("\n" + "=" * 60)
    print("Comprehensive Training and Optimization Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
