from typing import Dict, Any, List, Tuple
from collections import Counter

def normalize_owner(x: Any):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s.startswith("@"):
        s = s[1:]
    return s or None

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def score_sample(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    # priority exact
    p_pred = (pred.get("priority") or "").lower()
    p_gold = (gold.get("priority") or "").lower()
    out["priority"] = (p_pred == p_gold) if p_gold else True

    # owner exact (case-insensitive, '@' stripped)
    out["owner"] = (normalize_owner(pred.get("owner")) == normalize_owner(gold.get("owner")))

    # tags jaccard >= 0.6
    t_pred = [str(x).lower() for x in (pred.get("tags") or [])]
    t_gold = [str(x).lower() for x in (gold.get("tags") or [])]
    out["tags"] = (jaccard(t_pred, t_gold) >= 0.6) if t_gold else True

    # deadline exact string match if gold has one (ISO date string)
    d_pred = str(pred.get("deadline")) if pred.get("deadline") is not None else None
    d_gold = str(gold.get("deadline")) if gold.get("deadline") is not None else None
    out["deadline"] = (d_pred == d_gold) if d_gold else True

    # overall = macro average of fields considered
    out["overall"] = all(out.values())
    return out

def aggregate(scores: List[Dict[str, bool]]) -> Dict[str, float]:
    keys = ["priority","owner","tags","deadline","overall"]
    agg = {k: 0 for k in keys}
    for s in scores:
        for k in keys:
            agg[k] += 1 if s.get(k) else 0
    n = len(scores) or 1
    return {k: round(agg[k] / n, 3) for k in keys}
