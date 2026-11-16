"""Paper Figs. 3 and 4 (fig:pertalker, fig:gains) as vector PDFs, plus the CSVs behind them.

Usage:  python src/muffled_asr/oracle_filter/make_paper_figs.py [--out DIR]     (default: reports/paper-figs)

Fig. 3  fig_pertalker.pdf  Geng primary talkers (28): gap = WER(M)-WER(U) vs residual = WER(M)-WER(F),
        panels -10 / 0 / 10 dB (thr 6 dB, post-covering SNR), scored from results/p0/hyps_* via score.load.
Fig. 4  fig_gains.pdf      oracle band gains (datasets/p0/band_gains.csv, thr 6 dB) per mask; human talker
        mean + range vs mannikin response (filters.mannikin_band: mean of 6 refits / no-mask, same mic).
CSVs    pertalker_wer.csv, band_gains_human.csv, mannikin_bands.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, P0
from score import load, wer

# Paper colours (xcolor base names used in Figs. 1-2): black U, teal F, orange M, violet R
C = {"U": "#000000", "F": "#008080", "M": "#FF8000", "R": "#800080"}
SNRS = (-10, 0, 10)
YLIM_GAIN = (-24, 16)  # spec said -20..+15; widened so human minima (-23 dB) and mannikin A cheek peak (15.5 dB) are not clipped

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "lines.linewidth": 0.9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def pertalker():
    d = pd.concat([load("geng_lowsnr"), load("geng")], ignore_index=True)
    rows = []
    for snr in (-10, -5, 0, 5, 10):
        s = d[d.primary & (d.snr == snr) & (d.threshold_db == 6) & (d.snr_convention == "post")]
        for spk, x in s.groupby("speaker"):
            w = {c: wer(x[x.condition == c]) for c in "UFMR"}
            rows.append({"talker": spk, "snr": snr, **{f"wer_{c}": round(w[c], 2) for c in "UFMR"},
                         "gap": round(w["M"] - w["U"], 2), "residual": round(w["M"] - w["F"], 2)})
    return pd.DataFrame(rows)


def fig_pertalker(pt, out):
    fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.6), sharex=True, sharey=True)
    lim = (-20, 25)
    for ax, snr in zip(axs, SNRS):
        x = pt[pt.snr == snr]
        ax.plot([-30, 30], [-30, 30], "--", color="0.45", lw=0.6, zorder=1)
        ax.axhline(0, color="0.45", lw=0.5, zorder=1)
        ax.scatter(x.gap, x.residual, s=7, color=C["M"], edgecolor="k", linewidth=0.3, zorder=3)
        r = np.corrcoef(x.gap, x.residual)[0, 1]
        ax.text(0.04, 0.97, f"r = {r:.2f}\n{(x.gap > 0).sum()}/{len(x)} gap > 0", transform=ax.transAxes,
                va="top", ha="left", fontsize=6)
        ax.set_title((f"{snr:+d} dB" if snr else "0 dB").replace("-", "\u2212"), pad=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xticks([-20, 0, 20]); ax.set_yticks([-20, 0, 20])
    axs[0].set_ylabel("WER(M) − WER(F) (pts)")
    axs[1].set_xlabel("Mask gap WER(M) − WER(U) (pts)", labelpad=1)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.21, top=0.9, wspace=0.12)
    fig.savefig(out / "fig_pertalker.pdf"); fig.savefig(out / "fig_pertalker.png")
    plt.close(fig)


def gains():
    bg = pd.read_csv(P0 / "band_gains.csv")
    meta = json.loads((P0 / "filters_meta.json").read_text())
    prim = {k for k, v in meta.items() if str(v.get("primary")) == "True"}
    hum = bg[(bg.dataset != "mannikin") & (bg.threshold_db == 6.0) & bg.pair_id.isin(prim)]
    man = bg[bg.dataset == "mannikin"].copy()
    man["refits_on"] = man.pair_id.map(lambda p: meta[p]["refits_on"])
    man["refits_off"] = man.pair_id.map(lambda p: meta[p]["refits_off"])
    return hum, man


def fig_gains(hum, man, out):
    fig, axs = plt.subplots(2, 6, figsize=(7, 3), sharex=True, sharey=True)
    def summ(ds, ch, m):
        return hum[(hum.dataset == ds) & (hum.channel == ch) & (hum["mask"] == m)].groupby("band_hz").gain_db.agg(["mean", "min", "max", "count"])
    for i, ch in enumerate(("cheek", "far")):
        for j, m in enumerate("ABCDE"):
            ax = axs[i, j]; h = summ("coxh", ch, m); f = h.index.values
            mk = man[(man.channel == ch) & (man["mask"] == m)].set_index("band_hz").gain_db
            ax.axhline(0, color="0.6", lw=0.4, zorder=1)
            ax.fill_between(f, h["min"], h["max"], color=C["F"], alpha=.25, lw=0, zorder=2)
            ax.plot(f, h["mean"], color=C["F"], zorder=4)
            ax.plot(mk.index, mk.values, "--k", lw=0.8, zorder=5)
            if ch == "far":
                hc = summ("coxh", "cheek", m)
                ax.plot(hc.index, hc["mean"], ":", color=C["F"], lw=0.8, zorder=3)
            if i == 0: ax.set_title(f"Cox mask {m}", pad=2)
        axs[i, 0].set_ylabel(("Cheek mic" if ch == "cheek" else "1 m mic") + "\nGain G(b) (dB)")
    g = summ("geng", "mono", "surgical"); ax = axs[0, 5]
    ax.axhline(0, color="0.6", lw=0.4)
    ax.fill_between(g.index, g["min"], g["max"], color=C["F"], alpha=.25, lw=0)
    ax.plot(g.index, g["mean"], color=C["F"])
    ax.set_title(f"Geng surgical (n={int(g['count'].iloc[0])})", pad=2)
    lg = axs[1, 5]; lg.axis("off")
    hs = [mpl.patches.Patch(color=C["F"], alpha=.25, lw=0), plt.Line2D([], [], color=C["F"]),
          plt.Line2D([], [], ls="--", color="k", lw=0.8), plt.Line2D([], [], ls=":", color=C["F"], lw=0.8)]
    lg.legend(hs, ["Human talker range", "Human talker mean", "Mannikin (same mic)", "Human cheek mean\n(1 m row only)"],
              loc="center", frameon=False, handlelength=1.8)
    for ax in axs.flat:
        ax.set_xscale("log"); ax.set_xlim(100, 6300); ax.set_ylim(*YLIM_GAIN)
        ax.set_xticks([100, 300, 1000, 3000]); ax.set_xticklabels(["0.1", "0.3", "1", "3"])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.set_yticks([-20, -10, 0, 10])
    for ax in axs[1, :5]: ax.set_xlabel("Frequency (kHz)", labelpad=1)
    axs[0, 5].tick_params(labelbottom=True); axs[0, 5].set_xlabel("Frequency (kHz)", labelpad=1)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.93, wspace=0.08, hspace=0.28)
    fig.savefig(out / "fig_gains.pdf"); fig.savefig(out / "fig_gains.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "reports/paper-figs"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    pt = pertalker(); pt.to_csv(out / "pertalker_wer.csv", index=False)
    hum, man = gains()
    hum[["pair_id", "dataset", "speaker", "mask", "channel", "band_hz", "gain_db"]].rename(columns={"band_hz": "band_center_hz"}).to_csv(out / "band_gains_human.csv", index=False)
    man[["mask", "channel", "band_hz", "gain_db", "refits_on", "refits_off"]].rename(columns={"band_hz": "band_center_hz"}).to_csv(out / "mannikin_bands.csv", index=False)
    fig_pertalker(pt, out); fig_gains(hum, man, out)
    for snr in SNRS:
        x = pt[pt.snr == snr]
        print(f"{snr:+d} dB: n={len(x)} r={np.corrcoef(x.gap, x.residual)[0, 1]:.3f} gap>0 {(x.gap > 0).sum()}")
    print("wrote", out)


if __name__ == "__main__": main()
