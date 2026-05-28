import os
import re
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import pickle
from Bio import SeqIO
import argparse


def parse_fasta_with_labels(fasta_path):
    """
    解析 header 格式为 ">id|label=0" 或 ">id|label=1" 的 FASTA 文件
    返回: ids, sequences, labels (0/1)
    """
    ids = []
    sequences = []
    labels = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description  # 例如 "123|label=1"
        # 提取 label
        match = re.search(r'label=([01])', header)
        if match:
            label = int(match.group(1))
        else:
            raise ValueError(f"Cannot parse label from header: {header}")

        ids.append(record.id)  # 原始 id 部分（竖线前的内容）
        sequences.append(str(record.seq))
        labels.append(label)

    return ids, sequences, labels


def extract_esm2_embeddings_per_position(sequences, model, tokenizer, device, batch_size=4):
    """
    提取每个氨基酸位置的 embedding（不池化）
    返回: list of numpy arrays，每个数组形状 (seq_len, hidden_size)
    注意：不同序列长度不同，所以用 list 保存
    """
    model.eval()
    all_embeddings = []  # 每个元素是一个 (L_i, D) 的 numpy 数组

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting per-residue embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: (batch, padded_len, hidden_size)
            hidden = outputs.last_hidden_state.cpu().numpy()
            attention_mask = inputs["attention_mask"].cpu().numpy()  # (batch, padded_len)

        # 去除 padding 和特殊 token [CLS] / [EOS]?
        # ESM-2 tokenizer 会给序列加上 <cls> 和 <eos>，我们需要去掉它们，只保留氨基酸对应的 token
        # 通常 tokenizer 的输出中，第一个 token 是 <cls> (索引0)，最后一个可能是 <eos> (索引 -1)
        # 但为了方便，建议去除首尾。注意：如果序列长度为 L，tokenizer 后的长度 = L+2
        for j, seq_len_original in enumerate(len(seq) for seq in batch_seqs):
            # tokenizer 后的有效长度（包括 <cls> 和 <eos>） = 实际 token 数，可以通过 attention_mask 求和得到
            valid_len = int(attention_mask[j].sum())
            # 去掉 <cls> (位置0) 和 <eos> (位置 valid_len-1)
            # 注意：如果序列很短（例如1个氨基酸），要处理边界情况
            if valid_len >= 3:
                emb = hidden[j, 1:valid_len - 1, :]  # 只保留氨基酸部分
            else:
                emb = hidden[j, 1:valid_len, :]  # 如果长度太短，去掉 cls 但保留全部
            all_embeddings.append(emb)

    return all_embeddings


def extract_esm2_global_features(sequences, model, tokenizer, device, batch_size=8):
    """
    提取全局池化特征（平均池化所有氨基酸位置）
    返回: numpy array, shape (n_seq, hidden_size)
    """
    model.eval()
    all_features = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting global features"):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"]

            # 去掉 <cls> 和 <eos> 位置的 mask（如果它们存在）
            # 简单做法：将 mask 中对应 <cls> (索引0) 和 <eos> (最后一个有效位置) 置为0
            # 这里简化：所有非 padding 位置都参与池化，通常 <cls> 和 <eos> 也会包含，影响不大
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            all_features.append(pooled.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Extract ESM-2 features for pain peptide prediction")
        parser.add_argument("--input_dir", type=str, required=True,
                            help="Directory containing train.fasta, val.fasta, test.fasta")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save features")
        parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D",
                            help="ESM-2 model (default: esm2_t12_35M_UR50D, dim=480)")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--feature_type", type=str, default="per_residue", choices=["per_residue", "global"],
                            help="per_residue: each amino acid gets a vector (for CNN+LSTM); global: one vector per peptide")
        args = parser.parse_args()
    else:
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ESM-2 model: {args.model_name}")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    model = EsmModel.from_pretrained(args.model_name)
    device = torch.device(args.device)
    model.to(device)
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}, Device: {device}")

    for split in ["train", "val", "test"]:
        fasta_path = os.path.join(args.input_dir, f"{split}.fasta")
        if not os.path.exists(fasta_path):
            print(f"Warning: {fasta_path} not found, skipping {split}")
            continue

        print(f"\nProcessing {split}.fasta ...")
        ids, sequences, labels = parse_fasta_with_labels(fasta_path)
        print(f"  Loaded {len(sequences)} sequences")

        if args.feature_type == "per_residue":
            # 提取每个残基的 embedding（序列特征）
            embeddings_list = extract_esm2_embeddings_per_position(
                sequences, model, tokenizer, device, args.batch_size
            )
            # 保存为 list of arrays (pickle)
            out_path = os.path.join(args.output_dir, f"{split}_embeddings.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(embeddings_list, f)
            print(f"Saved per-residue embeddings to {out_path}")
            # 同时保存每个序列的长度，方便后续构建 batch
            seq_lens = [arr.shape[0] for arr in embeddings_list]
            with open(os.path.join(args.output_dir, f"{split}_seq_lens.pkl"), "wb") as f:
                pickle.dump(seq_lens, f)
        else:
            # 全局特征
            features = extract_esm2_global_features(
                sequences, model, tokenizer, device, args.batch_size
            )
            out_path = os.path.join(args.output_dir, f"{split}_features.npy")
            np.save(out_path, features)
            print(f"Saved global features to {out_path}, shape: {features.shape}")

        # 保存标签和 IDs（无论哪种模式都保存）
        meta_path = os.path.join(args.output_dir, f"{split}_metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({"ids": ids, "labels": labels}, f)
        print(f"Saved metadata to {meta_path}")
        print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

    print("\nAll done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with train/val/test .fasta files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature_type", type=str, default="per_residue", choices=["per_residue", "global"])
    args = parser.parse_args()
    main(args)