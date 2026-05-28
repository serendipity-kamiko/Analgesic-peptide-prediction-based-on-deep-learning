import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
from dataset import ProteinSequenceDataset
from model import SimplifiedProteinModel
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="训练简化的蛋白质分类模型")
    parser.add_argument("--features_dir", type=str, default="./features",
                        help="特征文件目录路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--model_type", type=str, default="simple",
                        choices=["simple"],
                        help="模型类型: simple(默认)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=100,
                        help="序列最大长度")
    parser.add_argument("--input_dim", type=int, default=480,
                        help="输入特征维度")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="分类数量")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减(L2正则化)")
    parser.add_argument("--grad_clip", type=float, default=0.5,
                        help="梯度裁剪阈值")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="学习率warmup轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(args):
    """根据参数获取模型"""
    if args.model_type == "simple":
        model = SimplifiedProteinModel(
            input_dim=args.input_dim,
            cnn_channels=64,
            lstm_hidden=128,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
    return model


def get_class_weights(train_dataset, device):
    """计算类别权重"""
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    # 平衡类别权重
    weights = [total_samples / (num_classes * count) for count in class_counts]
    print(f"类别分布: {dict(enumerate(class_counts))}")
    print(f"类别权重: {weights}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_epoch(model, dataloader, criterion, optimizer, device,
                epoch, total_epochs, grad_clip=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1:3d}/{total_epochs}",
        leave=False,
        ncols=80
    )

    for batch_idx, (features, labels, seq_lens) in enumerate(pbar):
        features = features.to(device)
        labels = labels.squeeze().to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs, _ = model(features)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # 记录
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # 更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'acc': f'{batch_acc:.3f}'
        })

    pbar.close()

    # 计算epoch指标
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1
    }


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, seq_lens in dataloader:
            features = features.to(device)
            labels = labels.squeeze().to(device)

            outputs, _ = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    return metrics


def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # F1分数曲线
    axes[0, 2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Training and Validation F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1])

    # 精确率曲线
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Train Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Val Precision', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Training and Validation Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 召回率曲线
    axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='Train Recall', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Training and Validation Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    # 学习率曲线
    if 'lr' in history and history['lr']:
        axes[1, 2].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    print(f"模型类型: {args.model_type}")
    print(f"参数配置: lr={args.lr}, bs={args.batch_size}, dropout={args.dropout}, wd={args.weight_decay}")

    # 加载数据集
    print("加载数据集...")
    train_dataset = ProteinSequenceDataset(
        args.features_dir, split="train", max_length=args.max_length
    )
    val_dataset = ProteinSequenceDataset(
        args.features_dir, split="val", max_length=args.max_length
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == "cuda"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device == "cuda"
    )

    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # 初始化模型
    print("初始化模型...")
    model = get_model(args).to(device)

    # 打印模型信息
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    # 计算类别权重
    class_weights = get_class_weights(train_dataset, device)
    print(f"使用的类别权重: {class_weights.cpu().numpy()}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器 (使用AdamW，更好的权重衰减)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控F1最大化
        factor=0.5,
        patience=5,  # 5个epoch无改善则降低学习率
        verbose=True,
        min_lr=1e-6
    )

    # 记录训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'lr': []
    }

    # 早停设置
    best_val_f1 = 0
    early_stop_counter = 0
    early_stop_patience = 10

    # 训练循环
    print("\n开始训练...")
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, args.grad_clip
        )

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # 记录指标
        for key in ['loss', 'accuracy', 'f1', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])

        # 学习率调度
        scheduler.step(val_metrics['f1'])

        # 打印epoch结果
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1:3d}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
        print(f"  Train: Loss {train_metrics['loss']:.4f} | Acc {train_metrics['accuracy']:.4f} | "
              f"F1 {train_metrics['f1']:.4f} | Prec {train_metrics['precision']:.4f} | Rec {train_metrics['recall']:.4f}")
        print(f"  Val:   Loss {val_metrics['loss']:.4f} | Acc {val_metrics['accuracy']:.4f} | "
              f"F1 {val_metrics['f1']:.4f} | Prec {val_metrics['precision']:.4f} | Rec {val_metrics['recall']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            early_stop_counter = 0

            best_model_path = os.path.join(args.output_dir, f"best_model_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_acc': val_metrics['accuracy'],
                'args': vars(args),
                'history': history
            }, best_model_path)
            print(f"  ✓ 保存最佳模型到 {best_model_path} (F1: {val_metrics['f1']:.4f})")
        else:
            early_stop_counter += 1

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'args': vars(args)
            }, checkpoint_path)

        # 早停检查
        if early_stop_counter >= early_stop_patience:
            print(f"\n⚠ 早停触发，连续 {early_stop_patience} 个epoch验证F1未提升")
            break

    # 绘制训练历史
    plot_training_history(history, args.output_dir)

    # 加载最佳模型并在测试集上评估
    print("\n在测试集上评估最佳模型...")
    test_dataset = ProteinSequenceDataset(
        args.features_dir, split="test", max_length=args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 找到最佳模型文件
    best_checkpoints = [f for f in os.listdir(args.output_dir) if f.startswith('best_model_epoch')]
    if best_checkpoints:
        best_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('epoch', '')))
        best_model_path = os.path.join(args.output_dir, best_checkpoints[-1])
    else:
        best_model_path = os.path.join(args.output_dir, "best_model.pth")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试评估
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 50)
    print("测试集性能:")
    print("=" * 50)
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  {test_metrics['confusion_matrix'][0]}")
    print(f"  {test_metrics['confusion_matrix'][1]}")

    # 保存最终结果
    result_path = os.path.join(args.output_dir, "test_results.txt")
    with open(result_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("测试集性能:\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"Loss:      {test_metrics['loss']:.4f}\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"混淆矩阵:\n")
        f.write(f"[{test_metrics['confusion_matrix'][0][0]} {test_metrics['confusion_matrix'][0][1]}]\n")
        f.write(f"[{test_metrics['confusion_matrix'][1][0]} {test_metrics['confusion_matrix'][1][1]}]\n")
        f.write(f"最佳epoch: {checkpoint.get('epoch', 'unknown') + 1}\n")
        f.write(f"最佳验证F1: {checkpoint.get('val_f1', 0):.4f}\n")

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'args': vars(args),
        'history': history
    }, final_model_path)

    print(f"\n训练完成! 结果已保存到 {args.output_dir}")
    print(f"最佳模型: {best_model_path}")
    print(f"最终模型: {final_model_path}")
    print(f"测试结果: {result_path}")


if __name__ == "__main__":
    main()