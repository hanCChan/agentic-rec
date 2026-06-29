# 数据获取说明 (国内无 Google Drive)

Rec-R1 README 里三个 GDrive 文件在国内通常不可用。本项目已用 HuggingFace 镜像替代 ESCI 语料。

## ESCI (已就绪)

| 用途 | 路径 | 来源 |
|------|------|------|
| RL 训练/验证/测试 | `Rec-R1/data/esci/inst/sparse/subset/*.parquet` | 仓库自带 |
| BM25 商品语料 | `Rec-R1/database/esci/jsonl_docs/esci_metadata.jsonl` | HF 重建 |
| Lucene 索引 | `Rec-R1/database/esci/pyserini_index/` | 已构建 |

重建语料 (如需重跑):

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec/Rec-R1
mkdir -p data/esci/raw
curl -L -C - "https://hf-mirror.com/datasets/spacemanidol/ESCI-product-dataset-corpus-us/resolve/main/collection.jsonl.gz" \
  -o data/esci/raw/collection.jsonl.gz
python src/Lucene/esci/0_build_corpus_from_hf.py
bash src/Lucene/esci/2_build_database.sh
```

## Amazon C4 (阶段扩展, 可选)

HF 搜索 `amazon c4` 或 Rec-R1 论文配套; 也可用类似方式从 HF 拉商品 metadata 再建索引。

## Amazon Review / All Beauty (生成式推荐出口, 可选)

仓库已有部分 `data/amazon_review/`; 完整语料可搜 HF `amazon review beauty`。

## 原则

- **训练 parquet**：ESCI 子集随仓库提供；Amazon C4 / Review 等大体积数据需本地下载（已从 Git 排除，见 `.gitignore`）
- **检索语料/索引**：用 HF (`hf-mirror.com`) 替代 GDrive，本地构建
- **基座模型**：`/data1/hcc/.hf_home/Qwen2.5-3B-Instruct`（已下载校验，不纳入 Git）
