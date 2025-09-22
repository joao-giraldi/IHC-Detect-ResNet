"""
Lê o CSV de contagens gerado pelo detector e produz:
1) Um CSV "enriquecido" com TOTAL por imagem e % por classe (taxas).
2) Um relatório Markdown com estatísticas gerais, TOP-N e **métricas agregadas** (P/R/F1).
3) Gráficos: totais por classe, histograma da taxa positiva, **F1 por classe**.

Requisitos: pandas, matplotlib
    pip install pandas matplotlib
"""

import os, math, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========
INPUT_CSV = r"D:\UFSC\2025.2\TCC\resNet\contagens_test.csv"
OUTPUT_DIR = r"D:\UFSC\2025.2\TCC\resNet\reports"
TOP_N = 15
POS_NAMES = {"positive", "pos", "positivo", "positiva"}
# ========= END CONFIG =====

os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_PREFIXES = ("tp_", "fp_", "fn_", "prec_", "rec_", "f1_")
GLOBAL_METRICS = {"tp_all", "fp_all", "fn_all", "prec_all", "rec_all", "f1_all"}


def _find_image_col(cols):
    for c in cols:
        if str(c).strip().lower() == "image":
            return c
    return cols[0]


def _is_metric_col(col: str) -> bool:
    c = str(col).strip().lower()
    return c in GLOBAL_METRICS or any(c.startswith(p) for p in METRIC_PREFIXES)


def _candidate_class_cols(df, img_col):
    out = []
    for c in df.columns:
        if c == img_col:
            continue
        cl = str(c).strip()
        if cl.lower().startswith("unnamed"):
            continue
        if cl.upper() == "TOTAL":
            continue
        if _is_metric_col(cl):
            continue
        out.append(c)
    return out


def _discover_metric_classes(cols):
    """
    Descobre nomes de classes presentes nas métricas a partir de colunas como:
    tp_positive, fp_negative, f1_uncertain...
    """
    names = set()
    for c in cols:
        c = str(c).strip()
        for pref in METRIC_PREFIXES:
            if c.startswith(pref):
                names.add(c[len(pref) :])
                break
    return sorted(names)


def load_counts(path):
    df = pd.read_csv(path)
    img_col = _find_image_col(df.columns)

    class_cols = _candidate_class_cols(df, img_col)

    for c in class_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    metric_class_names = _discover_metric_classes(df.columns)
    return df, img_col, class_cols, metric_class_names


def enrich(df, img_col, class_cols):
    out = df.copy()
    out["TOTAL"] = out[class_cols].sum(axis=1)
    for c in class_cols:
        out[f"rate_{c}"] = np.where(out["TOTAL"] > 0, out[c] / out["TOTAL"], 0.0)
    return out


def summarize(df_en, img_col, class_cols):
    totals = {c: int(df_en[c].sum()) for c in class_cols}
    grand_total = int(df_en["TOTAL"].sum())
    prevalence_micro = {
        c: (totals[c] / grand_total if grand_total > 0 else 0.0) for c in class_cols
    }
    macro_means = {c: float(df_en[c].mean()) for c in class_cols}
    macro_rate_means = {c: float(df_en[f"rate_{c}"].mean()) for c in class_cols}
    stats_total = {
        "mean": float(df_en["TOTAL"].mean()),
        "median": float(df_en["TOTAL"].median()),
        "std": float(df_en["TOTAL"].std(ddof=1) if len(df_en) > 1 else 0.0),
        "min": int(df_en["TOTAL"].min()),
        "max": int(df_en["TOTAL"].max()),
        "n_images": int(len(df_en)),
        "n_zero_total": int((df_en["TOTAL"] == 0).sum()),
    }
    return {
        "totals": totals,
        "grand_total": grand_total,
        "prevalence_micro": prevalence_micro,
        "macro_means": macro_means,
        "macro_rate_means": macro_rate_means,
        "stats_total": stats_total,
    }


def aggregate_metrics(df, metric_class_names):
    """
    Agrega métricas por classe a partir de tp_*, fp_*, fn_* nas linhas.
    Se houver colunas *_all, usa também para micro (senão soma tudo).
    Retorna dois DataFrames:
      - per_class_df: index=classe, cols=[TP,FP,FN,P,R,F1]
      - micro_df: uma linha com TP/FP/FN/P/R/F1
    """
    if not metric_class_names:
        return None, None

    # soma tp/fp/fn por classe (se a coluna não existir, trata como zero)
    records = []
    for cname in metric_class_names:
        tp = pd.to_numeric(df.get(f"tp_{cname}", 0), errors="coerce").fillna(0).sum()
        fp = pd.to_numeric(df.get(f"fp_{cname}", 0), errors="coerce").fillna(0).sum()
        fn = pd.to_numeric(df.get(f"fn_{cname}", 0), errors="coerce").fillna(0).sum()
        p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        records.append(
            {
                "classe": cname,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "Precisao": p,
                "Recall": r,
                "F1": f1,
            }
        )
    per_class_df = pd.DataFrame(records).sort_values("F1", ascending=False)

    if all(col in df.columns for col in ["tp_all", "fp_all", "fn_all"]):
        tp_all = pd.to_numeric(df["tp_all"], errors="coerce").fillna(0).sum()
        fp_all = pd.to_numeric(df["fp_all"], errors="coerce").fillna(0).sum()
        fn_all = pd.to_numeric(df["fn_all"], errors="coerce").fillna(0).sum()
    else:
        tp_all = per_class_df["TP"].sum()
        fp_all = per_class_df["FP"].sum()
        fn_all = per_class_df["FN"].sum()
    p = (tp_all / (tp_all + fp_all)) if (tp_all + fp_all) > 0 else 0.0
    r = (tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    micro_df = pd.DataFrame(
        [
            {
                "TP": int(tp_all),
                "FP": int(fp_all),
                "FN": int(fn_all),
                "Precisao": p,
                "Recall": r,
                "F1": f1,
            }
        ],
        index=["micro"],
    )
    return per_class_df, micro_df


def make_rankings(df_en, img_col, class_cols, top_n):
    ranks = {}
    for c in class_cols:
        top_count = df_en.sort_values(c, ascending=False).head(top_n)[
            [img_col, c, "TOTAL"]
        ]
        top_rate = df_en.sort_values(f"rate_{c}", ascending=False).head(top_n)[
            [img_col, f"rate_{c}", "TOTAL"]
        ]
        ranks[c] = {"count": top_count, "rate": top_rate}
    return ranks


def save_enriched_csv(df_en, outdir):
    path = os.path.join(outdir, "counts_enriched.csv")
    df_en.to_csv(path, index=False)
    return path


def save_metrics_csv(per_class_df, micro_df, outdir):
    if per_class_df is None:
        return None
    path = os.path.join(outdir, "metrics_summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        per_class_df.to_csv(f, index=False)
        f.write("\n")
        micro_df.to_csv(f)
    return path


def save_summary_md(
    summary, ranks, outdir, df_en, img_col, class_cols, per_class_df=None, micro_df=None
):
    md = []
    md.append("# Relatório de contagens (detector)\n")
    st = summary["stats_total"]
    md.append(
        f"- **Imagens**: {st['n_images']}  \n"
        f"- **Total de células**: {summary['grand_total']}  \n"
        f"- **Total por imagem (média ± desvio)**: {st['mean']:.2f} ± {st['std']:.2f}  \n"
        f"- **Mediana**: {st['median']:.2f}  \n"
        f"- **Mín/Máx**: {st['min']} / {st['max']}  \n"
        f"- **Imagens com TOTAL = 0**: {st['n_zero_total']}\n"
    )

    md.append("\n## Totais por classe (micro)")
    md.append(
        "\n| Classe | Total | % do total | Média por imagem | Média taxa por imagem |"
    )
    md.append("\n|---|---:|---:|---:|---:|")
    for c in class_cols:
        md.append(
            f"\n| {c} | {summary['totals'][c]} | {100*summary['prevalence_micro'][c]:.2f}% | "
            f"{summary['macro_means'][c]:.3f} | {100*summary['macro_rate_means'][c]:.2f}% |"
        )

    if per_class_df is not None:
        md.append("\n\n## Métricas agregadas (TP/FP/FN, Precisão, Recall, F1)\n")
        md.append(
            "> Observação: estas métricas refletem **o limiar SCORE_THR** e **o IoU** usados no script de contagem.\n"
        )
        # tabela por classe
        md.append("\n### Por classe\n")
        md.append(
            per_class_df.assign(
                Precisao=(per_class_df["Precisao"] * 100).round(2),
                Recall=(per_class_df["Recall"] * 100).round(2),
                F1=(per_class_df["F1"] * 100).round(2),
            )
            .rename(
                columns={
                    "Precisao": "Precisão (%)",
                    "Recall": "Recall (%)",
                    "F1": "F1 (%)",
                }
            )
            .to_markdown(index=False)
        )

        md.append("\n\n### Micro (global)\n")
        micro_fmt = micro_df.copy()
        micro_fmt[["Precisao", "Recall", "F1"]] = (
            micro_fmt[["Precisao", "Recall", "F1"]] * 100
        ).round(2)
        micro_fmt = micro_fmt.rename(
            columns={"Precisao": "Precisão (%)", "Recall": "Recall (%)", "F1": "F1 (%)"}
        )
        md.append(micro_fmt.to_markdown())

    for c in class_cols:
        md.append(f"\n\n## TOP {TOP_N} imagens por **contagem** de `{c}`\n")
        md.append(ranks[c]["count"].to_markdown(index=False))
        md.append(f"\n\n## TOP {TOP_N} imagens por **taxa** de `{c}`\n")
        md.append(ranks[c]["rate"].to_markdown(index=False))

    path = os.path.join(outdir, "summary_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    return path


def save_precision_md(per_class_df, micro_df, df_raw, img_col, outdir, top_n=15):
    """
    Gera um relatório Markdown focado em Precisão.
    - Tabela por classe: TP/FP/FN, GT (=TP+FN), Pred (=TP+FP), Precisão (%), ordenada por Precisão.
    - Se houver colunas per-image (prec_all, tp_all, fp_all, fn_all), lista TOP-N melhores e piores imagens.
    """
    if per_class_df is None or per_class_df.empty:
        return None

    # --- seção por classe (ordenada por precisão) ---
    tbl = per_class_df.copy()
    tbl["GT"] = tbl["TP"] + tbl["FN"]
    tbl["Pred"] = tbl["TP"] + tbl["FP"]
    tbl["Precisão (%)"] = (tbl["Precisao"] * 100).round(2)
    tbl = tbl[["classe", "TP", "FP", "FN", "GT", "Pred", "Precisão (%)"]].sort_values(
        "Precisão (%)", ascending=False
    )

    md = []
    md.append("# Relatório de Precisão\n")
    if micro_df is not None and not micro_df.empty:
        p_micro = float(micro_df.loc["micro", "Precisao"]) * 100.0
        md.append(f"- **Precisão global (micro)**: **{p_micro:.2f}%**\n")

    md.append("\n## Precisão por classe (ordenado)\n")
    md.append(tbl.to_markdown(index=False))

    # --- seção por imagem (se disponível) ---
    has_per_image = all(
        c in df_raw.columns for c in ["prec_all", "tp_all", "fp_all", "fn_all"]
    )
    if has_per_image:
        dfi = df_raw[[img_col, "prec_all", "tp_all", "fp_all", "fn_all"]].copy()
        dfi["prec_%"] = (
            pd.to_numeric(dfi["prec_all"], errors="coerce").fillna(0.0) * 100
        ).round(2)

        top = (
            dfi.sort_values("prec_all", ascending=False)
            .head(top_n)
            .rename(
                columns={
                    img_col: "image",
                    "tp_all": "TP",
                    "fp_all": "FP",
                    "fn_all": "FN",
                    "prec_%": "Precisão (%)",
                }
            )[["image", "TP", "FP", "FN", "Precisão (%)"]]
        )

        worst = (
            dfi.sort_values("prec_all", ascending=True)
            .head(top_n)
            .rename(
                columns={
                    img_col: "image",
                    "tp_all": "TP",
                    "fp_all": "FP",
                    "fn_all": "FN",
                    "prec_%": "Precisão (%)",
                }
            )[["image", "TP", "FP", "FN", "Precisão (%)"]]
        )

        md.append(f"\n\n## TOP {top_n} imagens com **maior** precisão\n")
        md.append(top.to_markdown(index=False))

        md.append(f"\n\n## TOP {top_n} imagens com **menor** precisão\n")
        md.append(worst.to_markdown(index=False))
    else:
        md.append(
            "\n\n> Observação: colunas por imagem (`prec_all`, `tp_all`, `fp_all`, `fn_all`) não foram encontradas no CSV; seção por imagem omitida.\n"
        )

    path = os.path.join(outdir, "precision_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    return path


def make_plots(df_en, img_col, class_cols, outdir, per_class_df=None):
    totals = [int(df_en[c].sum()) for c in class_cols]
    plt.figure()
    plt.bar(class_cols, totals)
    plt.title("Totais por classe (micro)")
    plt.xlabel("Classe")
    plt.ylabel("Total de detecções")
    p1 = os.path.join(outdir, "plot_totals_por_classe.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    plots = [p1]
    pos_cols = [c for c in class_cols if c.lower() in POS_NAMES]
    if pos_cols:
        c = pos_cols[0]
        rates = df_en[f"rate_{c}"].to_numpy()
        plt.figure()
        plt.hist(rates, bins=20)
        plt.title(f"Histograma da taxa de {c} por imagem")
        plt.xlabel(f"% {c}")
        plt.ylabel("n imagens")
        p2 = os.path.join(outdir, f"plot_hist_rate_{c}.png")
        plt.tight_layout()
        plt.savefig(p2, dpi=160)
        plt.close()
        plots.append(p2)

    if per_class_df is not None and not per_class_df.empty:
        plt.figure()
        plt.bar(per_class_df["classe"], (per_class_df["F1"] * 100.0))
        plt.title("F1 por classe (%)")
        plt.xlabel("Classe")
        plt.ylabel("F1 (%)")
        p3 = os.path.join(outdir, "plot_f1_por_classe.png")
        plt.tight_layout()
        plt.savefig(p3, dpi=160)
        plt.close()
        plots.append(p3)

        plt.figure()
        plt.bar(per_class_df["classe"], (per_class_df["Precisao"] * 100.0))
        plt.title("Precisão por classe (%)")
        plt.xlabel("Classe")
        plt.ylabel("Precisão (%)")
        p4 = os.path.join(outdir, "plot_precision_por_classe.png")
        plt.tight_layout()
        plt.savefig(p4, dpi=160)
        plt.close()
        plots.append(p4)

    return plots


def main():
    df, img_col, class_cols, metric_class_names = load_counts(INPUT_CSV)
    df_en = enrich(df, img_col, class_cols)

    per_class_df, micro_df = aggregate_metrics(df, metric_class_names)

    enriched_csv = save_enriched_csv(df_en, OUTPUT_DIR)
    metrics_csv = save_metrics_csv(per_class_df, micro_df, OUTPUT_DIR)
    summary = summarize(df_en, img_col, class_cols)
    ranks = make_rankings(df_en, img_col, class_cols, TOP_N)
    report_md = save_summary_md(
        summary, ranks, OUTPUT_DIR, df_en, img_col, class_cols, per_class_df, micro_df
    )
    precision_md = save_precision_md(
        per_class_df, micro_df, df, img_col, OUTPUT_DIR, top_n=TOP_N
    )

    plots = make_plots(df_en, img_col, class_cols, OUTPUT_DIR, per_class_df)

    print("Arquivos gerados:")
    print(" - CSV enriquecido:", enriched_csv)
    if metrics_csv:
        print(" - Métricas agregadas:", metrics_csv)
    print(" - Relatório MD   :", report_md)
    if precision_md:
        print(" - Relatório de Precisão:", precision_md)
    for p in plots:
        print(" - Figura         :", p)


if __name__ == "__main__":
    main()
