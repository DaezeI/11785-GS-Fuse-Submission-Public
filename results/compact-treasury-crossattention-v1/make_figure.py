"""Render the two references and four matched cells from sanitized aggregate metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODELS = (
    ("No change", "no_change", "#147D72"),
    ("Frozen yields only", "frozen_yields", "#708092"),
    ("Pooled · gate off", "pooled_matched", "#3C64B1"),
    ("Cross-attention · gate off", "cross_matched", "#D57649"),
    ("Pooled · gate on", "pooled_gate_matched", "#3C64B1"),
    ("Cross-attention · gate on", "cross_gate_matched", "#D57649"),
)

def main():
    metrics = (json.loads((HERE/"metrics.json").read_text(encoding="utf-8")))["mean_validation_metrics"]
    labels, keys, colors = zip(*MODELS)
    vals = [metrics[k]["mae_bp"] for k in keys]
    positions = [0,1,2.25,3.25,4.50,5.50]
    fig, ax = plt.subplots(figsize=(9.7,5.2),dpi=160)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=.31,right=.96,top=.80,bottom=.22)
    fig.text(.065,.927,"No change beat all four text variants",weight="bold",size=19,color="#182A3C")
    fig.text(.065,.867,"Compact Treasury pilot · event-weighted validation · lower is better",size=10.8,color="#506175")
    ax.set_axisbelow(True)
    ax.grid(axis="x",color="#E4EAF0",linewidth=1)
    ax.barh(positions,vals,color=colors,height=.62)
    ax.set_yticks(positions,labels,color="#283B4E",fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0,13.5)
    ax.set_xticks(range(0,13,3))
    ax.tick_params(axis="both",length=0,labelcolor="#506175",pad=7)
    ax.set_xlabel("Five-session MAE (basis points)",color="#506175",labelpad=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y,value in zip(positions,vals):
        ax.text(value+.14,y,f"{value:.3f}",va="center",fontsize=10,weight="semibold",color="#182A3C")
    fig.text(.065,.064,"12 validation statements · mean over seeds 23, 24, 25 · no held-out test",size=9.4,color="#506175")
    fig.savefig(HERE/"validation-mae.png",dpi=160,metadata={"Software":"GS-Fuse experiment figure"})
    plt.close(fig)

if __name__=="__main__":
    main()
