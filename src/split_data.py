from pathlib import Path
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split

# ========== 路径配置（请按实际位置修改） ==========
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_FILE = RAW_DIR / "positive100.fasta"
NEGATIVE_FILE = RAW_DIR / "negative100.fasta"
# =================================================

def save_fasta(df, filepath):
    with open(filepath, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{i}|label={row['label']}\n{row['sequence']}\n")

def main():
    # 读取正负样本（逻辑不变，仅替换文件路径）
    with open(POSITIVE_FILE, "r", encoding="utf-16") as f:
        positive_seqs = [(str(seq.seq), 1) for seq in SeqIO.parse(f, "fasta")]
    with open(NEGATIVE_FILE, "r", encoding="utf-16") as f:
        negative_seqs = [(str(seq.seq), 0) for seq in SeqIO.parse(f, "fasta")]

    all_seqs = positive_seqs + negative_seqs
    df = pd.DataFrame(all_seqs, columns=["sequence", "label"])

    print(df["label"].value_counts())

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    save_fasta(train_df, PROCESSED_DIR / "train.fasta")
    save_fasta(val_df, PROCESSED_DIR / "val.fasta")
    save_fasta(test_df, PROCESSED_DIR / "test.fasta")

if __name__ == "__main__":
    main()