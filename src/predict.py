import torch
import numpy as np
import os
import pickle
import argparse
from model import SimplifiedProteinModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="蛋白质序列分类预测器")
    parser.add_argument("--model_path", type=str, required=False,
                        help="模型权重文件路径")
    parser.add_argument("--features_dir", type=str, default="./features",
                        help="特征文件目录路径")
    parser.add_argument("--max_length", type=int, default=128,
                        help="序列最大长度（需与训练时保持一致）")
    parser.add_argument("--model_type", type=str, default="simple",
                        choices=["simple"],
                        help="模型类型: simple(默认)")
    parser.add_argument("--input_dim", type=int, default=480,
                        help="输入特征维度（需与训练时保持一致）")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda/cpu)，默认自动检测")
    parser.add_argument("--pad_to_max", type=bool, default=True,
                        help="是否填充到固定长度（与训练时保持一致）")
    return parser.parse_args()


class ProteinPredictor:
    def __init__(self, model_path, model_type="simple", input_dim=480,
                 num_classes=2, max_length=128, pad_to_max=True, device=None):
        """
        初始化预测器

        Args:
            model_path: 模型权重文件路径
            model_type: 模型类型 ("simple")
            input_dim: 输入特征维度
            num_classes: 分类数量
            max_length: 最大序列长度
            pad_to_max: 是否填充到固定长度
            device: 设备 (cuda/cpu)，为None时自动检测
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载模型检查点
        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get('args', {})

        # 从检查点获取模型参数（如果存在）
        if 'model_type' in args:
            model_type = args['model_type']
        if 'input_dim' in args:
            input_dim = args['input_dim']
        if 'num_classes' in args:
            num_classes = args['num_classes']
        if 'max_length' in args:
            max_length = args['max_length']

        # 根据模型类型初始化模型
        if model_type == "simple":
            self.model = SimplifiedProteinModel(
                input_dim=input_dim,
                cnn_channels=args.get('cnn_channels', 64),
                lstm_hidden=args.get('lstm_hidden', 128),
                num_classes=num_classes,
                dropout=args.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 保存模型信息
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        self.model_type = model_type

        print(f"模型加载完成！")
        print(f"  模型类型: {model_type}")
        print(f"  输入维度: {input_dim}")
        print(f"  类别数: {num_classes}")
        print(f"  最大序列长度: {self.max_length}")
        print(f"  填充到最大长度: {self.pad_to_max}")
        print(f"  设备: {self.device}")

        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    def load_features_from_dataset_format(self, features_dir, split="test"):
        """
        按照数据集格式加载特征

        Args:
            features_dir: 特征文件目录
            split: 数据集划分 (train/val/test)

        Returns:
            特征列表和标签列表
        """
        embeddings_path = os.path.join(features_dir, f"{split}_embeddings.pkl")
        metadata_path = os.path.join(features_dir, f"{split}_metadata.pkl")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"特征文件不存在: {embeddings_path}")

        # 加载特征和元数据
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            labels = metadata["labels"]

        print(f"从 {split} 数据集加载了 {len(embeddings)} 个样本")

        return embeddings, labels

    def preprocess_features_dataset_style(self, emb):
        """
        按照数据集类的预处理方式处理特征

        Args:
            emb: 原始特征，形状为 (seq_len, feature_dim)

        Returns:
            处理后的特征张量，形状为 (1, max_length, input_dim) 和 原始序列长度
        """
        # 确保是numpy数组
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()

        seq_len = emb.shape[0]

        # 处理不同长度序列（与_dataset.py中的逻辑保持一致）
        if seq_len < self.max_length and self.pad_to_max:
            # 填充到max_length
            padding = np.zeros((self.max_length - seq_len, emb.shape[1]))
            emb_padded = np.vstack([emb, padding])
        elif seq_len > self.max_length:
            # 截断到max_length
            emb_padded = emb[:self.max_length, :]
        else:
            emb_padded = emb

        # 转换为tensor并添加batch维度
        features_tensor = torch.FloatTensor(emb_padded).unsqueeze(0).to(self.device)
        return features_tensor, seq_len

    def predict_single(self, feature_array, return_attention=True):
        """
        预测单个样本

        Args:
            feature_array: 特征数组，形状为 (seq_len, input_dim)
            return_attention: 是否返回注意力权重

        Returns:
            包含预测结果的字典
        """
        # 按照数据集格式预处理特征
        features_tensor, seq_len = self.preprocess_features_dataset_style(feature_array)

        # 检查特征维度
        if features_tensor.shape[2] != self.input_dim:
            raise ValueError(f"特征维度不匹配: 模型期望 {self.input_dim}, 实际 {features_tensor.shape[2]}")

        # 预测
        with torch.no_grad():
            output, attention_weights = self.model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        result = {
            'prediction': predicted.item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'original_length': seq_len
        }

        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights.cpu().numpy()[0]

        return result

    def predict_from_dataset_files(self, features_dir, split="test", indices=None, return_attention=True):
        """
        从数据集格式的文件加载并预测

        Args:
            features_dir: 特征文件目录
            split: 数据集划分 (train/val/test)
            indices: 要预测的索引列表，为None时预测所有样本
            return_attention: 是否返回注意力权重

        Returns:
            预测结果列表
        """
        # 加载特征和标签
        embeddings, labels = self.load_features_from_dataset_format(features_dir, split)

        # 如果指定了索引，只预测指定索引
        if indices is not None:
            embeddings = [embeddings[i] for i in indices]
            labels = [labels[i] for i in indices]

        results = []
        batch_features = []
        original_lengths = []

        # 预处理所有特征
        for i, emb in enumerate(embeddings):
            features_tensor, seq_len = self.preprocess_features_dataset_style(emb)

            # 检查特征维度
            if features_tensor.shape[2] != self.input_dim:
                raise ValueError(f"样本 {i} 特征维度不匹配: 模型期望 {self.input_dim}, 实际 {features_tensor.shape[2]}")

            # 移除batch维度用于批处理
            batch_features.append(features_tensor.squeeze(0))
            original_lengths.append(seq_len)

        # 堆叠为batch
        if batch_features:
            batch_tensor = torch.stack(batch_features).to(self.device)

            # 批量预测
            with torch.no_grad():
                outputs, attention_weights = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

            # 组织结果
            for i in range(len(embeddings)):
                result = {
                    'sample_idx': indices[i] if indices is not None else i,
                    'prediction': predictions[i].item(),
                    'true_label': labels[i] if labels is not None else None,
                    'probabilities': probabilities[i].cpu().numpy(),
                    'original_length': original_lengths[i],
                    'is_correct': None
                }

                # 如果提供了真实标签，计算是否正确
                if labels is not None:
                    result['is_correct'] = (predictions[i].item() == labels[i])

                if return_attention and attention_weights is not None:
                    result['attention_weights'] = attention_weights[i].cpu().numpy()

                results.append(result)

        return results

    def evaluate_on_dataset(self, features_dir, split="test"):
        """
        在数据集上评估模型性能

        Args:
            features_dir: 特征文件目录
            split: 数据集划分 (train/val/test)

        Returns:
            评估结果字典
        """
        results = self.predict_from_dataset_files(features_dir, split, return_attention=False)

        if not results:
            return {"error": "没有预测结果"}

        # 计算评估指标
        predictions = [r['prediction'] for r in results]
        true_labels = [r['true_label'] for r in results if r['true_label'] is not None]

        if not true_labels:
            return {
                "total_samples": len(results),
                "predictions": predictions,
                "note": "没有真实标签，无法计算准确率"
            }

        # 计算准确率
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / len(results)

        # 计算各类别的统计
        from collections import defaultdict
        class_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for r in results:
            if r['true_label'] is not None:
                label = r['true_label']
                class_stats[label]["total"] += 1
                if r['is_correct']:
                    class_stats[label]["correct"] += 1

        # 计算每个类别的准确率
        class_accuracies = {}
        for label, stats in class_stats.items():
            if stats["total"] > 0:
                class_accuracies[label] = stats["correct"] / stats["total"]

        return {
            "total_samples": len(results),
            "accuracy": accuracy,
            "class_accuracies": dict(class_accuracies),
            "class_distribution": dict(class_stats),
            "predictions": predictions,
            "true_labels": true_labels
        }

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'pad_to_max': self.pad_to_max,
            'device': str(self.device)
        }


def main():
    """主函数：命令行使用示例"""
    args = parse_args()

    # 初始化预测器
    predictor = ProteinPredictor(
        model_path=args.model_path,
        model_type="simple",
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        max_length=128,
        pad_to_max=True,
        device=args.device
    )

    print("\n模型信息:")
    for key, value in predictor.get_model_info().items():
        print(f"  {key}: {value}")

    # 示例：从数据集文件加载并预测
    print(f"\n正在从目录加载特征: {args.features_dir}")

    try:
        # 尝试从test数据集加载
        print(f"尝试加载测试集...")

        # 方法1：从数据集格式文件加载并评估
        print("\n方法1: 从数据集格式文件评估模型性能")
        eval_results = predictor.evaluate_on_dataset(args.features_dir, split="test")

        if "accuracy" in eval_results:
            print(f"评估结果:")
            print(f"  总样本数: {eval_results['total_samples']}")
            print(f"  整体准确率: {eval_results['accuracy']:.4f}")

            if "class_accuracies" in eval_results:
                print(f"  各类别准确率:")
                for label, acc in eval_results['class_accuracies'].items():
                    print(f"    类别 {label}: {acc:.4f}")

            if "class_distribution" in eval_results:
                print(f"  类别分布:")
                for label, stats in eval_results['class_distribution'].items():
                    print(f"    类别 {label}: {stats['correct']}/{stats['total']} 正确")
        else:
            print(f"评估结果: {eval_results}")

        # 方法2：预测前几个样本
        print(f"\n方法2: 预测前5个样本的详细信息")
        sample_results = predictor.predict_from_dataset_files(
            args.features_dir,
            split="test",
            indices=list(range(min(5, eval_results.get('total_samples', 0))))
        )

        for i, result in enumerate(sample_results):
            print(f"\n样本 {i} (索引 {result['sample_idx']}):")
            print(f"  真实标签: {result['true_label']}")
            print(f"  预测类别: {result['prediction']}")
            print(f"  类别概率: {[f'{p:.4f}' for p in result['probabilities']]}")
            print(f"  是否正确: {result['is_correct']}")
            print(f"  原始序列长度: {result['original_length']}")

    except FileNotFoundError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()