"""Run the whole benchmark with one command: every notebook in notebooks/, in dependency order.

    python main_run.py                    # reuse saved results; generate only what is missing
    python main_run.py --analysis-only    # recompute every table and figure from results/ (no GPU, no token)
    python main_run.py --fresh            # delete the saved generations and regenerate everything (GPU + token)
    python main_run.py --only 5_3 6       # run some notebooks only (prefix of the notebook name)
    python main_run.py --list             # show the plan and exit

On Kaggle (GPU T4, Internet on, HF_TOKEN secret), in one cell:
    !python IIT-LCS2/main_run.py

How a notebook is run
  * Each notebook has experiment cells (generation, GPU) followed by analysis cells (CPU).
  * Default: if a notebook's saved generations in results/experiments/ are complete, only its analysis cells
    run, so the saved results are never touched. If anything is missing, the whole notebook runs; the
    experiment cells skip every item that is already saved (resumable), then the analysis runs.
  * Executed copies are written to executed/ (git-ignored); the notebooks in notebooks/ are not modified.
  * The run stops at the first failing notebook and names the failing cell.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "executed"
EXP_DIR = ROOT / "results" / "experiments"

MODELS = ("llama", "qwen")
BIAS_VARIANTS = ("race_black", "race_white", "insurance_uninsured", "insurance_private",
                 "residence_rural", "residence_urban")

# Dependency order. Each entry: notebook -> {experiment folder: {file name pattern: expected rows}}.
# Notebooks 6 and 7 only analyse saved results, so they have no experiment files.
PLAN = {
    "4_part_a_baseline": {"4_part_a_baseline": {"{m}_core.jsonl": 200}},
    "5_1_hallucination_uncertainty": {"5_1_hallucination_uncertainty": {"{m}_stress.jsonl": 60}},
    "5_2_interpretability": {"5_2_interpretability": {"{m}_interp.jsonl": 40}},
    "5_3_bias_fairness": {"5_3_bias_fairness": {f"{{m}}_{v}.jsonl": 60 for v in BIAS_VARIANTS}},
    "5_4_privacy_security": {"5_4_1_prompt_injection": {"{m}_injection.jsonl": 60},
                             "5_4_2_canary_leakage": {"{m}_canary.jsonl": 60, "{m}_canary_control.jsonl": 60}},
    "8_part_d_prompt_hardening": {"8_part_d_prompt_hardening": {f"{{m}}_hardened_{c}.jsonl": 60
                                                                for c in ("misleading", "clean", "injection")}},
    "6_cross_dimension": {},
    "7_overview_all_experiments": {},
}


def missing_generations(name: str) -> list[str]:
    """Saved generation files of a notebook that are absent or incomplete (empty list = complete)."""
    missing = []
    for folder, files in PLAN[name].items():
        for pattern, rows in files.items():
            for m in MODELS:
                path = EXP_DIR / folder / pattern.format(m=m)
                n = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0
                if n < rows:
                    missing.append(f"{folder}/{path.name} ({n}/{rows})")
    return missing


def analysis_cells(nb: dict) -> list[dict]:
    """The title cell plus every cell from the analysis section on (the whole notebook if it has no sections)."""
    cells = nb["cells"]
    start = next((i for i, c in enumerate(cells) if c["cell_type"] == "markdown"
                  and (c.get("id") or "").endswith("-ana")), None)
    return cells if start is None else cells[:1] + cells[start:]


def hf_token_available() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    if Path("/kaggle/input").exists():          # Kaggle: the notebooks read the HF_TOKEN secret themselves
        return True
    env = ROOT / ".env"
    if env.exists() and any(l.strip().startswith("HF_TOKEN=") and l.split("=", 1)[1].strip()
                            for l in env.read_text().splitlines()):
        return True
    return (Path.home() / ".cache" / "huggingface" / "token").exists()   # `huggingface-cli login`


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_notebook(name: str, analysis_only: bool, timeout: int | None) -> None:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(NB_DIR / f"{name}.ipynb", as_version=4)
    if analysis_only:
        nb.cells = analysis_cells(nb)
    for c in nb.cells:                           # start from clean outputs
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{name}.ipynb"
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}})
    try:
        client.execute()
    except CellExecutionError as err:
        nbformat.write(nb, out_path)             # keep the partial run for inspection
        failed = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code"
                       and any(o.get("output_type") == "error" for o in c.get("outputs", []))), None)
        first_line = next((l for l in nb.cells[failed].source.splitlines() if l.strip()), "") if failed is not None else ""
        raise SystemExit(f"\nFAILED in {name}, cell {failed}: {first_line}\n"
                         f"{str(err).strip().splitlines()[-1]}\nPartial run saved to {out_path.relative_to(ROOT)}")
    nbformat.write(nb, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run every notebook of the benchmark in dependency order.")
    ap.add_argument("--analysis-only", action="store_true",
                    help="only recompute tables and figures from results/ (no GPU, no token)")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the saved generations of the selected notebooks and regenerate them")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="run only notebooks whose name starts with these prefixes, e.g. --only 4 5_3 6")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--timeout", type=int, default=None, help="per-cell timeout in seconds (default: none)")
    args = ap.parse_args()
    if args.fresh and args.analysis_only:
        ap.error("--fresh and --analysis-only cannot be combined")

    if not (ROOT / "config.yaml").exists():
        raise SystemExit("config.yaml not found next to main_run.py - run this from the repository.")

    names = list(PLAN)
    if args.only:
        names = [n for n in names if any(n.startswith(p) for p in args.only)]
        if not names:
            raise SystemExit(f"no notebook matches {args.only}; choose from: {', '.join(PLAN)}")

    # Decide what each notebook needs
    plan = []
    for name in names:
        if args.analysis_only or not PLAN[name]:
            plan.append((name, "analysis", []))
        elif args.fresh:
            plan.append((name, "generate + analysis", ["all (fresh)"]))
        else:
            missing = missing_generations(name)
            plan.append((name, "generate + analysis" if missing else "analysis (results complete)", missing))

    print(f"Repository: {ROOT}")
    for i, (name, mode, missing) in enumerate(plan, 1):
        extra = f"  missing: {', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}" if missing else ""
        print(f"  [{i}/{len(plan)}] {name:<32} {mode}{extra}")
    if args.list:
        return

    # Checks before the long run
    if any(mode.startswith("generate") for _, mode, _ in plan):
        problems = []
        if not gpu_available():
            problems.append("no CUDA GPU found (on Kaggle: Settings -> Accelerator -> GPU T4)")
        if not hf_token_available():
            problems.append("no Hugging Face token (Kaggle: Add-ons -> Secrets -> HF_TOKEN; locally: .env)")
        if problems:
            raise SystemExit("Generation is needed but:\n  - " + "\n  - ".join(problems) +
                             "\nTo recompute tables and figures from the saved results instead, use --analysis-only.")

    if args.fresh:
        for name, _, _ in plan:
            for folder in PLAN[name]:
                for f in sorted((EXP_DIR / folder).glob("*.jsonl")) + [EXP_DIR / folder / "run_meta.json"]:
                    if f.exists():
                        f.unlink()
                        print(f"  deleted {f.relative_to(ROOT)}")

    t_all = time.time()
    for i, (name, mode, _) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {name} - {mode} ...", flush=True)
        t0 = time.time()
        run_notebook(name, analysis_only=mode.startswith("analysis"), timeout=args.timeout)
        print(f"        ok ({(time.time() - t0) / 60:.1f} min) -> executed/{name}.ipynb", flush=True)
    print(f"Done: {len(plan)} notebooks in {(time.time() - t_all) / 60:.1f} min. "
          f"Results in results/, executed notebooks in executed/.")


if __name__ == "__main__":
    sys.exit(main())
