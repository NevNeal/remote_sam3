#!/usr/bin/env python3
"""
From the per-segment model answers to one flower-type call per family and
character, and the figure that compares the families.

    python all_genera/pv99/analysis/family_types.py

Inputs: export/pv99/analysis/morphology/photos.csv (morph_classify.py) and
family_reference.csv (the textbook expectation per family, next to this file).

Per segment, per character: the call is the most probable option among the
real (non-abstain) options, or "cant_tell" when the abstain options together
hold >= ABSTAIN_CUT of the probability.

Per family, per character (the decision rules; methodology_pdf.py prints them):

  n/a          the reference says the character does not apply (e.g. symmetry
               of a grass spikelet): no call.
  confirmed    the reference lists one state and >= CONFIRM of the segments that
               could be told were called that state.
  supported    one reference state, called in >= SUPPORT but < CONFIRM.
  conflict     one reference state, called in < SUPPORT: the reference state is
               kept as the family's call and the conflict is listed for review.
  variable     the reference lists several states: the call is the states seen
               in >= SEEN of the segments, most frequent first; it is
               "confirmed" if those reference states together cover >= CONFIRM.
  image only   no reference row for the family: the image majority is the call
               when it holds >= CONFIRM of >= MIN_TOLD segments, else "uncertain".

Writes into export/pv99/analysis/morphology/:

  family_shares.csv      family x character x option: share of segments
  family_calls.csv       family x character: reference, image shares, call, status
  family_summary.csv     one row per family: the calls as one line
  calibration.csv        per character: how often a single segment's call
                         matches a single-state reference (the model's accuracy
                         where the textbook leaves no doubt)
  flower_types_by_family.png   the comparison figure
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from morph_questions import QUESTIONS, option_keys  # noqa: E402

ABSTAIN_CUT = 0.5
CONFIRM = 0.5
SUPPORT = 0.25
SEEN = 0.10
MIN_TOLD = 30

LABELS = {
    "solitary": "solitary", "raceme_spike": "raceme/spike", "panicle_cyme": "panicle/cyme",
    "umbel": "umbel", "head": "capitulum", "spadix": "spadix", "spikelet": "spikelet",
    "catkin": "catkin", "cyathium": "cyathium", "actinomorphic": "actinomorphic",
    "zygomorphic": "zygomorphic", "disymmetric": "disymmetric", "asymmetric": "asymmetric",
    "petals": "petals", "tepals": "tepals", "sepals_bracts": "sepals/bracts",
    "stamens": "stamens", "reduced": "reduced", "free": "free", "fused": "fused",
    "none": "no petals", "open": "open/rotate", "bell": "campanulate", "urn": "urceolate",
    "tubular": "tubular", "salverform": "salverform", "lipped": "bilabiate",
    "pea": "papilionaceous", "orchid": "orchid", "cross": "cruciform", "spur": "spur",
    "corona": "corona", "hood": "hood", "nothing": "no spur/corona/hood", "3_6": "3/6", "4_8": "4/8",
    "5_10": "5/10", "many": ">10", "radiate": "radiate", "discoid": "discoid",
    "ligulate": "ligulate", "not_head": "not a head", "cant_tell": "can't tell",
}


def real_options(q):
    return [k for k in option_keys(q) if k not in q["abstain"]]


def segment_calls(photos):
    """photo x character -> option key or 'cant_tell'."""
    out = photos[["photo_id", "family", "genus"]].copy()
    for q in QUESTIONS:
        real = real_options(q)
        P = photos[[f"{q['key']}__{k}" for k in real]].to_numpy(float)
        abstain = photos[[f"{q['key']}__{k}" for k in q["abstain"]]].to_numpy(float).sum(1)
        call = np.array(real, dtype=object)[P.argmax(1)]
        out[q["key"]] = np.where(abstain >= ABSTAIN_CUT, "cant_tell", call)
    return out


def shares(calls):
    rows = []
    for family, g in calls.groupby("family"):
        for q in QUESTIONS:
            counts = g[q["key"]].value_counts()
            for k in real_options(q) + ["cant_tell"]:
                rows.append(dict(family=family, character=q["key"], option=k,
                                 segments=int(counts.get(k, 0)), n=len(g),
                                 share=counts.get(k, 0) / len(g)))
    return pd.DataFrame(rows)


def parse_ref(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return [] if value == "-" else value.split("|")


def decide(ref_states, told_share, n_told):
    """told_share: option -> share among segments that could be told."""
    if ref_states == []:
        return "-", "n/a", "the reference says this character does not apply"
    ranked = sorted(told_share.items(), key=lambda kv: -kv[1])
    top, top_share = ranked[0] if ranked else ("cant_tell", 0.0)
    if n_told == 0:
        if ref_states:
            return "|".join(ref_states), "reference only", \
                "no segment could be told; reference kept"
        return "uncertain", "uncertain", "no reference and no segment could be told"
    if ref_states is None:
        if top_share >= CONFIRM and n_told >= MIN_TOLD:
            return top, "image only", f"no reference; {top_share:.0%} of {n_told} segments"
        return "uncertain", "uncertain", \
            f"no reference; image majority {LABELS[top]} only {top_share:.0%} of {n_told}"
    if len(ref_states) == 1:
        s = told_share.get(ref_states[0], 0.0)
        status = "confirmed" if s >= CONFIRM else "supported" if s >= SUPPORT else "conflict"
        why = f"{s:.0%} of {n_told} segments called {LABELS[ref_states[0]]}"
        if status == "conflict":
            why += f"; the image majority was {LABELS[top]} ({top_share:.0%})"
        return ref_states[0], status, why
    seen = [k for k in ref_states if told_share.get(k, 0.0) >= SEEN]
    seen.sort(key=lambda k: -told_share.get(k, 0.0))
    cover = sum(told_share.get(k, 0.0) for k in ref_states)
    if not seen:
        seen = [max(ref_states, key=lambda k: told_share.get(k, 0.0))]
    status = "variable, confirmed" if cover >= CONFIRM else "variable, conflict"
    why = ", ".join(f"{LABELS[k]} {told_share.get(k, 0.0):.0%}" for k in ref_states) + \
        f" of {n_told} segments (reference states cover {cover:.0%})"
    if cover < CONFIRM:
        why += f"; the image majority was {LABELS[top]} ({top_share:.0%})"
    return "|".join(seen), status, why


def family_calls(calls, reference):
    ref = reference.set_index("family")
    rows = []
    for family, g in calls.groupby("family"):
        for q in QUESTIONS:
            told = g[g[q["key"]] != "cant_tell"][q["key"]]
            n_told = len(told)
            told_share = (told.value_counts() / n_told).to_dict() if n_told else {}
            ref_states = parse_ref(ref.at[family, q["key"]]) if family in ref.index else None
            call, status, why = decide(ref_states, told_share, n_told)
            top = max(told_share, key=told_share.get) if told_share else "cant_tell"
            rows.append(dict(
                family=family, character=q["key"], label=q["label"], n=len(g), n_told=n_told,
                reference="" if ref_states is None else ("-" if ref_states == [] else
                                                         "|".join(ref_states)),
                image_majority=top, image_majority_share=round(told_share.get(top, 0.0), 3),
                call=call, status=status, rationale=why))
    return pd.DataFrame(rows)


def summary(table, order):
    rows = []
    for family in order:
        g = table[table["family"] == family].set_index("character")
        parts = []
        for q in QUESTIONS:
            r = g.loc[q["key"]]
            if r["status"] in ("n/a", "uncertain") or (q["key"] == "head_type"
                                                       and r["call"] == "not_head"):
                continue
            if q["key"] == "accessory" and r["call"] == "nothing":
                continue
            text = " / ".join(LABELS[k] for k in str(r["call"]).split("|"))
            flag = "" if "confirmed" in r["status"] or r["status"] == "image only" else \
                "?" if r["status"] in ("supported", "reference only") else "!"
            parts.append(text + flag)
        n_conf = int(g["status"].str.contains("confirmed").sum())
        n_apply = int((g["status"] != "n/a").sum())
        rows.append(dict(family=family, flower_type=" · ".join(parts),
                         characters_confirmed=f"{n_conf}/{n_apply}",
                         conflicts=", ".join(g.index[g["status"].str.contains("conflict")])))
    return pd.DataFrame(rows)


def calibration(calls, reference):
    ref = reference.set_index("family")
    rows = []
    for q in QUESTIONS:
        hit = told = abst = total = 0
        for family, g in calls.groupby("family"):
            if family not in ref.index:
                continue
            states = parse_ref(ref.at[family, q["key"]])
            if not states or len(states) != 1:
                continue
            total += len(g)
            t = g[g[q["key"]] != "cant_tell"]
            told += len(t)
            abst += len(g) - len(t)
            hit += int((t[q["key"]] == states[0]).sum())
        rows.append(dict(character=q["key"], label=q["label"], segments=total, told=told,
                         abstained=abst, matches=hit,
                         accuracy=round(hit / told, 3) if told else np.nan,
                         abstain_rate=round(abst / total, 3) if total else np.nan))
    return pd.DataFrame(rows)


def figure(share_table, reference, order, out):
    ref = reference.set_index("family")
    cols = []
    for q in QUESTIONS:
        for k in real_options(q) + ["cant_tell"]:
            cols.append((q, k))
    M = np.zeros((len(order), len(cols)))
    lookup = share_table.set_index(["family", "character", "option"])["share"]
    for i, fam in enumerate(order):
        for j, (q, k) in enumerate(cols):
            M[i, j] = lookup.get((fam, q["key"], k), 0.0)

    cell = 0.30
    gap = 0.18
    xs, x = [], 0.0
    for j, (q, k) in enumerate(cols):
        if j and cols[j - 1][0] is not q:
            x += gap
        xs.append(x)
        x += cell
    width_in = x + 2.6
    height_in = cell * len(order) + 2.9
    fig = plt.figure(figsize=(width_in, height_in), dpi=200)
    fig.patch.set_facecolor(common.SURFACE)
    ax = fig.add_axes([2.3 / width_in, 0.25 / height_in, x / width_in,
                       cell * len(order) / height_in])
    ax.set_xlim(0, x)
    ax.set_ylim(len(order) * cell, 0)
    ax.axis("off")
    blue = LinearSegmentedColormap.from_list("b", common.BLUE_RAMP)
    grey = LinearSegmentedColormap.from_list("g", ["#fcfcfb", "#8a8984"])
    for i, fam in enumerate(order):
        y = i * cell
        for j, (q, k) in enumerate(cols):
            v = M[i, j]
            colour = grey(v) if k == "cant_tell" else blue(v)
            ax.add_patch(Rectangle((xs[j] + 0.01, y + 0.01), cell - 0.02, cell - 0.02,
                                   facecolor=colour, edgecolor="none"))
            states = parse_ref(ref.at[fam, q["key"]]) if fam in ref.index else None
            if states and k in states:
                lw = 1.6 if k == states[0] else 0.9
                ax.add_patch(Rectangle((xs[j] + 0.03, y + 0.03), cell - 0.06, cell - 0.06,
                                       facecolor="none", edgecolor=common.ORANGE, linewidth=lw))
            if v >= 0.2:
                ax.text(xs[j] + cell / 2, y + cell / 2, f"{v * 100:.0f}", ha="center",
                        va="center", fontsize=5.5,
                        color="#ffffff" if v >= 0.55 and k != "cant_tell" else common.INK)
        if states is None and fam not in ref.index:
            pass
        n = int(share_table[(share_table["family"] == fam)]["n"].iloc[0])
        ax.text(-0.08, y + cell / 2, f"{fam}  ({n})", ha="right", va="center", fontsize=7.5,
                color=common.INK if fam in ref.index else common.INK_2)
    # column labels and character headers
    start = 0
    for q in QUESTIONS:
        n_opt = len(real_options(q)) + 1
        x0, x1 = xs[start], xs[start + n_opt - 1] + cell
        ax.text((x0 + x1) / 2, -1.55, q["label"], ha="center", va="bottom", fontsize=8.5,
                color=common.INK, fontweight="bold", clip_on=False)
        ax.plot([x0, x1], [-1.48, -1.48], color=common.INK_2, linewidth=0.8, clip_on=False)
        for j in range(start, start + n_opt):
            ax.text(xs[j] + cell / 2, -0.06, LABELS[cols[j][1]], rotation=60, ha="left",
                    va="bottom", fontsize=6.3, color=common.INK_2, clip_on=False,
                    rotation_mode="anchor")
        start += n_opt
    fig.text(0.25 / width_in, 1 - 0.28 / height_in,
             "Flower types identified in the top 100 SAM3 segments of the 30 largest pv99 families",
             fontsize=13, fontweight="bold", color=common.INK, va="top")
    fig.text(0.25 / width_in, 1 - 0.62 / height_in,
             "Cell = % of the family's segments the model called that type (blue; grey = can't "
             "tell). Orange outline = textbook state for the family (thick = commonest). "
             "Rows by number of pv99 photos; (n) = segments. Grey family name = no textbook row.",
             fontsize=7.5, color=common.INK_2, va="top")
    fig.savefig(out, facecolor=common.SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    args = parser.parse_args()
    project = common.project_dir(args.project)
    out = common.analysis_dir(project, "morphology")
    photos = pd.read_csv(out / "photos.csv")
    photos = photos[photos["error"].fillna("") == ""]
    reference = pd.read_csv(HERE / "family_reference.csv")
    segs = pd.read_csv(common.analysis_dir(project, "top_segments") / "top_segments.csv")
    order = (segs.drop_duplicates("family").sort_values("family_rank")["family"].tolist())
    order = [f for f in order if f in set(photos["family"])]

    calls = segment_calls(photos)
    calls.to_csv(out / "segment_calls.csv", index=False)
    share_table = shares(calls)
    share_table.to_csv(out / "family_shares.csv", index=False)
    table = family_calls(calls, reference)
    table.to_csv(out / "family_calls.csv", index=False)
    summ = summary(table, order)
    summ.to_csv(out / "family_summary.csv", index=False)
    cal = calibration(calls, reference)
    cal.to_csv(out / "calibration.csv", index=False)
    figure(share_table, reference, order, out / "flower_types_by_family.png")
    print(cal.to_string(index=False))
    print()
    print(summ.to_string(index=False, max_colwidth=90))
    print(f"\n           : {out}")


if __name__ == "__main__":
    main()
