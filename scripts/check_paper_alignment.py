#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "rows": 27000,
    "datasets": 25,
    "seeds": [42, 44, 49],
    "folds": 5,
    "primary_mean_pr_auc": 0.6869525445453175,
    "pste_fol_vs_smote_delta": 0.02035953863345493,
    "pste_fol_vs_smote_wins": 68,
    "pste_fol_vs_pste_smote_delta": 0.002124022867623115,
    "pste_fol_vs_pste_smote_wins": 51,
    "pste_fol_vs_pste_smote_holm_p": 0.032277478951423,
}

CLASSIFIERS = ["rf", "extratrees", "bagged_cart"]


def load_reference(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def primary_method(clf: str) -> str:
    return f"fc_acc_ppob_{clf}_fast_outward_ladder_smote"


def smote_method(clf: str) -> str:
    return f"{clf}_smote"


def pste_smote_method(clf: str) -> str:
    return f"fc_acc_ppob_{clf}_smote"


def block_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clf in CLASSIFIERS:
        sub = df[df["classifier"].eq(clf)]
        for ds, g in sub.groupby("dataset"):
            rows.append(
                {
                    "dataset": ds,
                    "classifier": clf,
                    "pste_fol": g[g.method.eq(primary_method(clf))]["pr_auc"].mean(),
                    "smote": g[g.method.eq(smote_method(clf))]["pr_auc"].mean(),
                    "pste_smote": g[g.method.eq(pste_smote_method(clf))]["pr_auc"].mean(),
                }
            )
    return pd.DataFrame(rows)


def assert_close(name: str, actual: float, expected: float, tol: float = 5e-10):
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{name}: actual={actual!r}, expected={expected!r}, tol={tol}")


def method_key(df: pd.DataFrame) -> pd.Series:
    if "paper_method_alias" in df.columns:
        return df["paper_method_alias"].fillna(df["method"]).astype(str)
    return df["method"].astype(str)


def count_signature(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_method"] = method_key(tmp)
    return tmp.groupby(["dataset", "seed", "fold", "_method"], dropna=False).size().rename("n").reset_index()


def compare_generated_to_reference(generated_path: str | Path, reference_df: pd.DataFrame, *, compare_values: bool = False, value_tol: float = 1e-6) -> None:
    generated = pd.read_csv(generated_path)
    if len(generated) != len(reference_df):
        raise AssertionError(f"generated row count mismatch: {len(generated)} != reference {len(reference_df)}")
    ref_sig = count_signature(reference_df)
    gen_sig = count_signature(generated)
    merged = ref_sig.merge(gen_sig, on=["dataset", "seed", "fold", "_method"], how="outer", suffixes=("_ref", "_gen"), indicator=True)
    bad = merged[(merged["_merge"] != "both") | (merged["n_ref"] != merged["n_gen"])]
    if len(bad):
        raise AssertionError("generated method/block count signature does not match reference; first differences:\n" + bad.head(20).to_string(index=False))
    if compare_values:
        ref = reference_df.copy(); ref["_method"] = method_key(ref)
        gen = generated.copy(); gen["_method"] = method_key(gen)
        ref_mean = ref.groupby("_method")["pr_auc"].mean()
        gen_mean = gen.groupby("_method")["pr_auc"].mean()
        diff = (gen_mean - ref_mean).abs().sort_values(ascending=False)
        worst = float(diff.iloc[0]) if len(diff) else 0.0
        if worst > value_tol:
            raise AssertionError(f"generated PR-AUC method means differ from reference by up to {worst:.6g} > {value_tol}; first differences:\n{diff.head(20).to_string()}")
    print(f"generated-vs-reference shape OK: {generated_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check packaged reference results against the ESWA manuscript claims.")
    p.add_argument("--reference", default=str(ROOT / "reference" / "paper_results"))
    p.add_argument("--material", default=str(ROOT / "reference" / "paper_material"))
    p.add_argument("--generated", default=None, help="Optional freshly generated paper-exact CSV to compare against the packaged reference row/method/block signature.")
    p.add_argument("--compare-generated-values", action="store_true", help="Also compare generated PR-AUC method means against the reference; normally leave off because library versions can change exact values.")
    p.add_argument("--generated-value-tol", type=float, default=1e-6)
    p.add_argument("--tol", type=float, default=5e-10)
    args = p.parse_args(argv)

    df = load_reference(Path(args.reference))
    print(f"reference rows: {len(df)}")
    if len(df) != EXPECTED["rows"]:
        raise AssertionError(f"row count mismatch: {len(df)} != {EXPECTED['rows']}")
    if df["dataset"].nunique() != EXPECTED["datasets"]:
        raise AssertionError(f"dataset count mismatch: {df['dataset'].nunique()} != {EXPECTED['datasets']}")
    if sorted(map(int, df["seed"].unique())) != EXPECTED["seeds"]:
        raise AssertionError(f"seed set mismatch: {sorted(df['seed'].unique())}")
    if int(df["fold"].max()) != EXPECTED["folds"] or int(df["fold"].min()) != 1:
        raise AssertionError("fold labels are inconsistent with 1..5 CV")

    prim = pd.concat([df[df.method.eq(primary_method(clf))] for clf in CLASSIFIERS])
    primary_mean = float(prim["pr_auc"].mean())
    assert_close("primary PSTE-FOL mean PR-AUC", primary_mean, EXPECTED["primary_mean_pr_auc"], args.tol)

    blocks = block_table(df)
    if len(blocks) != 75 or blocks.isna().any().any():
        raise AssertionError(f"expected 75 complete dataset-classifier blocks; got {len(blocks)} rows and na={blocks.isna().sum().to_dict()}")

    delta_smote = float((blocks["pste_fol"] - blocks["smote"]).mean())
    wins_smote = int((blocks["pste_fol"] > blocks["smote"]).sum())
    assert_close("PSTE-FOL vs sampler-only SMOTE mean delta", delta_smote, EXPECTED["pste_fol_vs_smote_delta"], args.tol)
    if wins_smote != EXPECTED["pste_fol_vs_smote_wins"]:
        raise AssertionError(f"PSTE-FOL vs SMOTE wins mismatch: {wins_smote} != {EXPECTED['pste_fol_vs_smote_wins']}")

    delta_pste_smote = float((blocks["pste_fol"] - blocks["pste_smote"]).mean())
    wins_pste_smote = int((blocks["pste_fol"] > blocks["pste_smote"]).sum())
    assert_close("PSTE-FOL vs PSTE-SMOTE mean delta", delta_pste_smote, EXPECTED["pste_fol_vs_pste_smote_delta"], args.tol)
    if wins_pste_smote != EXPECTED["pste_fol_vs_pste_smote_wins"]:
        raise AssertionError(f"PSTE-FOL vs PSTE-SMOTE wins mismatch: {wins_pste_smote} != {EXPECTED['pste_fol_vs_pste_smote_wins']}")

    # The manuscript reports Holm-adjusted p-values from the full pairwise table.
    pairwise = pd.read_csv(Path(args.material) / "OVERALL_PFOL_PAIRWISE_PR_AUC.csv")
    row = pairwise[pairwise["code"].eq("P-SM")].iloc[0]
    assert_close("PSTE-FOL vs PSTE-SMOTE Holm p", float(row["holm_p"]), EXPECTED["pste_fol_vs_pste_smote_holm_p"], args.tol)
    row_os = pairwise[pairwise["code"].eq("OS-SM")].iloc[0]
    assert_close("pairwise table OS-SM delta", float(row_os["mean_delta"]), EXPECTED["pste_fol_vs_smote_delta"], args.tol)

    ranks = pd.read_csv(Path(args.material) / "OVERALL_FRIEDMAN_PR_AUC_RANKS.csv")
    pfol = ranks[ranks["code"].eq("P-FOL")].iloc[0]
    assert_close("Friedman table P-FOL mean PR-AUC", float(pfol["mean_pr_auc"]), EXPECTED["primary_mean_pr_auc"], args.tol)

    # Informative unadjusted one-sided Wilcoxon values over the 75 blocks.
    _, p_smote = wilcoxon(blocks["pste_fol"], blocks["smote"], zero_method="wilcox", alternative="greater")
    _, p_pste_smote = wilcoxon(blocks["pste_fol"], blocks["pste_smote"], zero_method="wilcox", alternative="greater")

    if args.generated:
        compare_generated_to_reference(args.generated, df, compare_values=args.compare_generated_values, value_tol=args.generated_value_tol)

    print("alignment OK")
    print(f"PSTE-FOL mean PR-AUC: {primary_mean:.10f}")
    print(f"vs sampler-only SMOTE: delta={delta_smote:.10f}, wins={wins_smote}/75, one-sided Wilcoxon p={p_smote:.3g}")
    print(f"vs PSTE-SMOTE: delta={delta_pste_smote:.10f}, wins={wins_pste_smote}/75, Holm p={float(row['holm_p']):.6g}, one-sided Wilcoxon p={p_pste_smote:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
