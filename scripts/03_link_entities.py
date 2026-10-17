"""Link ML-100K and ML-1M items to Freebase / YAGO via KB4Rec (decision 0003)."""

import json
from pathlib import Path

from tkgat.data.clean import clean
from tkgat.data.loaders import load
from tkgat.kg.linking import attach_kb_ids, load_kb4rec, match_ml100k_to_ml1m

out = Path("data/processed/links")
out.mkdir(parents=True, exist_ok=True)

inter100, items100 = load("ml-100k", "data/raw/ml-100k")
inter100, items100, _ = clean(inter100, items100)
inter1m, items1m = load("ml-1m", "data/raw/ml-1m")
fb, yago = load_kb4rec("data/raw/kb4rec")

links100 = attach_kb_ids(match_ml100k_to_ml1m(items100, items1m), "item_ml1m", fb, yago)
links1m = attach_kb_ids(items1m[["item", "title", "year"]], "item", fb, yago)
links100.to_csv(out / "ml100k_links.csv", index=False)
links1m.to_csv(out / "ml1m_links.csv", index=False)


def coverage(links, inter, key):
    w = inter[key if key in inter else "item"].value_counts()
    has = links.set_index(key)["freebase_id"].notna()
    return {
        "items": int(len(links)),
        "freebase_items": int(has.sum()),
        "freebase_item_share": round(float(has.mean()), 4),
        "freebase_rating_share": round(float((w * has.reindex(w.index).fillna(False)).sum() / w.sum()), 4),
        "yago_item_share": round(float(links["yago_id"].notna().mean()), 4),
    }


report = {
    "ml100k": {
        "match_rules": links100["rule"].value_counts(dropna=False).rename(index=str).to_dict(),
        **coverage(links100, inter100.rename(columns={"item": "item_100k"}), "item_100k"),
    },
    "ml1m": coverage(links1m, inter1m, "item"),
}
(out / "linking_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
