"""
用 HuggingFace 上的 ESCI US 商品语料 (spacemanidol/ESCI-product-dataset-corpus-us 的
collection.jsonl.gz) 重建 Rec-R1 所需语料, 替代原仓库需从 Google Drive 下载的
sampled_item_metadata_esci.jsonl (国内被墙)。

输入: data/esci/raw/collection.jsonl.gz
输出: database/esci/jsonl_docs/esci_metadata.jsonl
每行: {"id": <ASIN>, "title": <title>, "contents": <可检索文本>}

下载语料(若尚未下载):
  curl -L "https://hf-mirror.com/datasets/spacemanidol/ESCI-product-dataset-corpus-us/resolve/main/collection.jsonl.gz" \
       -o data/esci/raw/collection.jsonl.gz
之后用 src/Lucene/esci/2_build_database.sh 建 Lucene 索引。
"""
import os
import gzip
import json
import glob
import pandas as pd

SRC = "data/esci/raw/collection.jsonl.gz"
OUT = "database/esci/jsonl_docs/esci_metadata.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def build_contents(row):
    contents = (row.get("contents") or "").strip()
    if contents:
        return contents
    parts = [
        row.get("title") or "",
        row.get("bullet_points") or "",
        row.get("brand") or "",
        row.get("color") or "",
        row.get("text") or "",
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


# 收集训练/验证/测试涉及的目标 ASIN, 用于覆盖率检查
target_ids = set()
for p in glob.glob("data/esci/inst/sparse/subset/*.parquet"):
    df = pd.read_parquet(p)
    for arr in df["item_id"]:
        if hasattr(arr, "__iter__") and not isinstance(arr, str):
            target_ids.update(arr)
        else:
            target_ids.add(arr)
print(f"target item_ids in subset: {len(target_ids)}")

corpus_ids = set()
n = 0
with gzip.open(SRC, "rt", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        docid = row.get("docid") or row.get("id")
        if not docid:
            continue
        doc = {
            "id": docid,
            "title": (row.get("title") or "").strip(),
            "contents": build_contents(row),
        }
        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
        corpus_ids.add(docid)
        n += 1

covered = len(target_ids & corpus_ids)
print(f"wrote {n} docs -> {OUT}")
print(f"target coverage: {covered}/{len(target_ids)} "
      f"({100.0*covered/max(1,len(target_ids)):.2f}%)")
missing = list(target_ids - corpus_ids)[:10]
if missing:
    print("examples of missing target ids:", missing)
