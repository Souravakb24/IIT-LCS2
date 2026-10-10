"""Numbers quoted in the report that are not already in results/analysis/*.csv.

Reads only the saved records in results/, so every quoted number can be regenerated:
  part_a  Part A accuracy (Wilson CI), ECE and AUROC (bootstrap CI), chance accuracy, parse statistics
  5_1     evidence echo of the injected sentence (+ control) and answer = k by provenance / content type
  5_3     answer flip rates with Wilson CIs (pair and control)
usage: python scripts/report_numbers.py [section ...]     e.g. python scripts/report_numbers.py part_a
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "experiments"
MODELS = {"llama": "Llama-3.2-3B", "qwen": "Qwen2.5-3B"}
RNG = np.random.default_rng(42)


def load(part: str, name: str) -> pd.DataFrame:
    return pd.read_json(OUT / part / f"{name}.jsonl", lines=True).sort_values("row_index").reset_index(drop=True)


def wilson(k: int, n: int, z: float = 1.96) -> str:
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def boot_ci(stat, *arrays, n: int = 10_000) -> str:
    m = len(arrays[0])
    vals = [stat(*(a[s] for a in arrays)) for s in (RNG.integers(0, m, m) for _ in range(n))]
    return f"[{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]"


def auroc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    return mannwhitneyu(score_pos, score_neg).statistic / (len(score_pos) * len(score_neg))


def part_a():
    print("== Part A (Core, n=200)")
    for key, name in MODELS.items():
        d = load("4_part_a_baseline", f"{key}_core")
        conf, corr = d["confidence_token"].to_numpy(), d["correct"].to_numpy().astype(float)
        chance = (1 / d["n_options"]).mean()
        print(f"  {name}: accuracy {wilson(int(corr.sum()), len(d))} | chance {chance:.3f}")
        print(f"    mean conf {conf.mean():.3f} | ECE {ece(conf, corr):.3f} {boot_ci(ece, conf, corr)}")
        print(f"    conf correct {conf[corr == 1].mean():.3f} vs wrong {conf[corr == 0].mean():.3f} | "
              f"AUROC(conf -> correct) {auroc(conf[corr == 1], conf[corr == 0]):.3f} "
              f"{boot_ci(lambda c, y: auroc(c[y == 1], c[y == 0]), conf, corr)}")
        print(f"    conf >= 0.9: {int((conf >= .9).sum())} items, accuracy {corr[conf >= .9].mean():.3f} | "
              f"wrong with conf >= 0.8: {int(((corr == 0) & (conf >= .8)).sum())}")
        print(f"    parse {d['parse_status'].value_counts().to_dict()} | truncated {int(d['truncated'].sum())} | "
              f"mean generated tokens {d['generated_tokens'].mean():.1f}")


STOP = set("""a an the and or of to in on for with by from at as is are was were be been being this that these those
it its into than then such which who whom whose what when where why how all any both each few more most other some
no nor not only own same so too very can will just should now may might must shall would could also there their they
them he she his her him patient patients""".split())


def content_words(text: str) -> set:
    import re
    return {w for w in re.findall(r"[a-z][a-z\-]+", str(text).lower()) if len(w) >= 4 and w not in STOP}


def part_5_1():
    """Evidence echo: share of the injected sentence's content words that reappear in the evidence.
    Control = the clean evidence of the same item (written without seeing the injected sentence)."""
    print("== 5.1 evidence echo of the injected sentence (Stress, n=60)")
    for key, name in MODELS.items():
        clean = load("4_part_a_baseline", f"{key}_core").query("row_index < 60").set_index("row_index")
        mis = load("5_1_hallucination_uncertainty", f"{key}_stress").set_index("row_index")
        inj = mis["injected_text"].map(content_words)
        share = lambda ev: np.array([len(content_words(e) & w) / len(w) for e, w in zip(ev, inj)])
        s_mis, s_cln = share(mis["evidence"]), share(clean.loc[mis.index, "evidence"])
        echo_mis, echo_cln = s_mis >= 0.5, s_cln >= 0.5
        chose = mis["chose_injected"].to_numpy()
        print(f"  {name}: mean overlap clean {s_cln.mean():.3f} -> misleading {s_mis.mean():.3f} | "
              f"echo (>=50% of injected words) clean {wilson(int(echo_cln.sum()), 60)} -> misleading {wilson(int(echo_mis.sum()), 60)}")
        print(f"    echo when answer = injected option: {int(echo_mis[chose].sum())}/{int(chose.sum())} | "
              f"when not: {int(echo_mis[~chose].sum())}/{int((~chose).sum())}")
        for col in ("injection_provenance", "injection_content"):
            g = mis.assign(flip=mis["answer"] != clean.loc[mis.index, "answer"]).groupby(col)["chose_injected"]
            print(f"    chose injected by {col}: " + ", ".join(f"{k} {int(v.sum())}/{len(v)}" for k, v in g))


def part_5_3():
    """Wilson CIs for the answer flip rate between counterfactual pairs and against the clean control."""
    print("== 5.3 flip rates with Wilson CIs (Stress, n=60)")
    pairs = {"race": ("race_black", "race_white"), "insurance": ("insurance_uninsured", "insurance_private"),
             "residence": ("residence_rural", "residence_urban")}
    for key, name in MODELS.items():
        clean = load("4_part_a_baseline", f"{key}_core").query("row_index < 60")["answer"].to_numpy()
        for attr, (va, vb) in pairs.items():
            a, b = (load("5_3_bias_fairness", f"{key}_{v}")["answer"].to_numpy() for v in (va, vb))
            print(f"  {name} {attr:<9}: A vs B {wilson(int((a != b).sum()), 60)} | "
                  f"A vs clean {wilson(int((a != clean).sum()), 60)} | B vs clean {wilson(int((b != clean).sum()), 60)}")


SECTIONS = {"part_a": part_a, "5_1": part_5_1, "5_3": part_5_3}

if __name__ == "__main__":
    for s in sys.argv[1:] or SECTIONS:
        SECTIONS[s]()
