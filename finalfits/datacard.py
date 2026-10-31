"""Combine datacard from the yields tables of all categories (:mod:`finalfits.datacard_yields`).

The yield variations become lnN factors ``[down/nominal, up/nominal]`` (or one factor for symmetric
weights) per process and category; theory systematics are split into tiers (``shape``: migrations
within a process, ``norm``, ...). Processes below a fraction of their category yield can be
pruned. :func:`clean_datacard` removes unphysical lnN values afterwards.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import pickle
import re
from dataclasses import dataclass

import pandas as pd

from finalfits import analysis
from finalfits.datacard_yields import experimental_systematics_for, factory_type
from finalfits.trees2ws import SQRTS

logger = logging.getLogger(__name__)

LINE_BREAK = "-" * 130
SCALE_GROUPINGS = [[0, 8], [1, 7], [3, 5]]  # weight_LHEScal pairs (muR, muF) up/down
MIN_EVENTS = 100  # fewer simulated events: no systematic variations (except lumi)


@dataclass
class DatacardOptions:
    years: str  # comma separated years with eras, e.g. 2023preBPix,2023postBPix
    variable: str = ""
    skip_cow_corr: bool = False
    do_systematics: bool = False
    do_mc_stat: bool = False
    prune: bool = False
    prune_threshold: float = 0.001
    true_yield: bool = False
    mass: str = "125"
    analysis: str = "STXS"


def load_yields(yields_dir: str) -> pd.DataFrame:
    """The yields tables of all categories (alphabetical) as one DataFrame."""
    data = pd.DataFrame()
    for path in sorted(glob.glob(f"{yields_dir}/*.pkl")):
        with open(path, "rb") as f:
            data = pd.concat([data, pickle.load(f)], ignore_index=True, axis=0, sort=False)
    return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Systematic factors


def _signal(data):
    return (data["type"] == "sig") & (~data["cat"].str.contains("NOTAG"))


def _value_from_json(row, uncertainties, name):
    proc = row["proc"]
    for year in ("2016", "2017", "2018", "2022preEE", "2022postEE"):
        proc = re.sub(f"_{year}_{analysis.decayMode}", "", proc)
    if proc not in uncertainties:
        return "-"
    value = uncertainties[proc][name]
    return value if isinstance(value, list) else [value]


def _lumi_value(values: dict):
    """Per-year lumi uncertainty: the full year label, its prefix or its digits, or an inclusive key."""

    def resolve(row):
        label = str(row.get("year", ""))
        candidates = [
            label,
            label.split("_", 1)[0] if "_" in label else label,
            "".join(c for c in label if c.isdigit()),
        ]
        for key in [*candidates, "Run3", "combined", "all"]:
            if key and key in values:
                return values[key]
        return "-"

    return resolve


def add_constant_systematic(data: pd.DataFrame, syst: dict, years: list[str]) -> pd.DataFrame:
    name, value = syst["name"], syst["value"]
    if syst["correlateAcrossYears"] == 1:
        data[name] = "-"
        mask = _signal(data)
        if isinstance(value, str) and "json" in value:
            with open(value) as f:
                uncertainties = json.load(f)
            sig = data["type"] == "sig"
            data.loc[sig, name] = data[sig].apply(lambda x: _value_from_json(x, uncertainties, name), axis=1)
        elif isinstance(value, dict):
            resolve = (
                _lumi_value(value) if name.startswith("lumi") else lambda row: value.get(str(row.get("year", "")), "-")
            )
            data.loc[mask, name] = data.loc[mask].apply(resolve, axis=1)
        else:
            data.loc[mask, name] = value
    elif syst["correlateAcrossYears"] == -1:
        data[name] = "-"
        for year in years:
            data.loc[_signal(data) & (data["year"] == year), name] = value[year]
    else:
        for year in years:
            data[f"{name}_{year}"] = "-"
            data.loc[_signal(data) & (data["year"] == year), f"{name}_{year}"] = value[year]
    return data


def compare_yield(row, kind: str, name: str, mode: str = "default") -> list[float]:
    """lnN factor(s) of a systematic for one process x category; [1, 1] if a yield is zero."""
    unit = [1.0, 1.0] if kind in ("a_w", "a_h") else [1.0]
    if row["nominal_yield"] == 0:
        return unit
    keys = [f"{name}_down", f"{name}_up"] if kind in ("a_w", "a_h") else [name]
    if mode != "default":
        if row["proc_nominal_yield"] == 0 or any(row[f"proc_{k}_yield"] == 0 for k in keys):
            return unit

    if mode in ("default", "ishape"):
        if kind == "a_h":
            # histogram variations are not centred on the nominal yield: compare to their midpoint, and
            # leave out unphysical (non-positive) yields of weighted events
            down, up = row[f"{name}_down_yield"], row[f"{name}_up_yield"]
            if down <= 0.0 or up <= 0.0 or 0.5 * (down + up) <= 0.0:
                return [1.0, 1.0]
            midpoint = 0.5 * (down + up)
            ratios = [down / midpoint, up / midpoint]
            return [1.0, 1.0] if min(ratios) <= 0.0 else ratios
        return [row[f"{k}_yield"] / row["nominal_yield"] for k in keys]
    if mode == "shape":
        return [
            (row[f"{k}_yield"] / row["nominal_yield"]) / (row[f"proc_{k}_yield"] / row["proc_nominal_yield"])
            for k in keys
        ]
    if mode == "norm":
        return [
            (row[f"proc_{k}_yield"] / row["proc_nominal_yield"])
            / (row[f"proc_s0_{k}_yield"] / row["proc_s0_nominal_yield"])
            for k in keys
        ]
    if mode == "inorm":
        return [row[f"proc_{k}_yield"] / row["proc_nominal_yield"] for k in keys]
    if mode == "inc":
        return [row[f"proc_s0_{k}_yield"] / row["proc_s0_nominal_yield"] for k in keys]
    raise ValueError(f"Theory systematic tier {mode} is not supported")


def experimental_factory(data, systematics, types, years):
    for s in systematics:
        if s["type"] == "constant":
            continue
        if s["correlateAcrossYears"]:
            data[s["name"]] = "-"
        else:
            for year in years:
                data[f"{s['name']}_{year}"] = "-"
    for s in systematics:
        if s["type"] == "constant":
            continue
        kind = types[s["name"]]
        if s["correlateAcrossYears"]:
            mask = _signal(data)
            data.loc[mask, s["name"]] = data[mask].apply(
                lambda x, kind=kind, name=s["name"]: compare_yield(x, kind, name), axis=1
            )
        else:
            for year in years:
                mask = _signal(data) & (data["year"] == year)
                data.loc[mask, f"{s['name']}_{year}"] = data[mask].apply(
                    lambda x, kind=kind, name=s["name"]: compare_yield(x, kind, name), axis=1
                )
    return data


def _yield_keys(name, kind):
    return [f"{name}_up", f"{name}_down"] if kind in ("a_w", "a_h") else [name]


def theory_factory(data, systematics, types, years, skip_cow_corr):
    corr = "" if skip_cow_corr else "_COWCorr"
    factories = [s for s in systematics if s["type"] != "constant"]
    # yields summed per production mode (and year) and per process
    for proc_s0 in data[data["type"] == "sig"].proc_s0.unique():
        for year in years:
            mask = (data["proc_s0"] == proc_s0) & (data["year"] == year)
            data.loc[mask, "proc_s0_nominal_yield"] = data[mask][f"nominal_yield{corr}"].sum()
            for s in factories:
                for key in _yield_keys(s["name"], types[s["name"]]):
                    data.loc[mask, f"proc_s0_{key}_yield"] = data[mask][f"{key}_yield{corr}"].sum()
    for proc in data[data["type"] == "sig"].proc.unique():
        mask = data["proc"] == proc
        data.loc[mask, "proc_nominal_yield"] = data[mask][f"nominal_yield{corr}"].sum()
        for s in factories:
            for key in _yield_keys(s["name"], types[s["name"]]):
                data.loc[mask, f"proc_{key}_yield"] = data[mask][f"{key}_yield{corr}"].sum()
    for s in factories:
        if "mnorm" in s["tiers"]:
            raise NotImplementedError("Merged STXS normalisation (mnorm) systematics are not supported")
        for tier in s["tiers"]:
            data[f"{s['name']}_{tier}"] = "-"
    for s in factories:
        kind = types[s["name"]]
        mask = (data["type"] == "sig") & (data["nominal_yield"] != 0)
        if "THU_ggH" in s["name"]:
            mask &= data["proc"].str.contains("ggH")
        for tier in s["tiers"]:
            data.loc[mask, f"{s['name']}_{tier}"] = data[mask].apply(
                lambda x, kind=kind, name=s["name"], tier=tier: compare_yield(x, kind, name, mode=tier), axis=1
            )
    return data


def group_systematics(data, systematics, prefix, groupings):
    """Combine pairs of single weights (e.g. scale weights) into up/down systematics <prefix>_gr<i>."""
    for index, (first, second) in enumerate(groupings):
        s0 = next((s for s in systematics if s["name"] == f"{prefix}_{first}"), None)
        s1 = next((s for s in systematics if s["name"] == f"{prefix}_{second}"), None)
        if s0 is None or s1 is None:
            logger.warning("No systematic for prefix %s and group %s", prefix, (first, second))
            continue
        for tier in s0["tiers"]:
            if tier == "mnorm":
                continue
            c0, c1 = f"{s0['name']}_{tier}", f"{s1['name']}_{tier}"
            group = f"{prefix}_gr{index}" + ("" if tier == "shape" else f"_{tier}")
            data[group] = "-"
            mask = (data[c0] != "-") & (data[c1] != "-")
            data.loc[mask, group] = data[mask].apply(lambda x, c0=c0, c1=c1: [x[c0][0], x[c1][0]], axis=1)
        s0["name"] = s0["name"].replace(f"{prefix}_{first}", f"{prefix}_gr{index}")
        s0["title"] = s0["title"].replace(f"{prefix}_{first}", f"{prefix}_gr{index}")
        systematics.remove(s1)
    return data, systematics


def prune(data, opts: DatacardOptions) -> pd.DataFrame:
    """Flag (prune = 1) signal processes below prune_threshold of their category yield, and NOTAG."""
    data["prune"] = 0
    sig = data["type"] == "sig"
    if opts.true_yield:
        # expected events: XS * BR * eff x acc * lumi
        data["xsbr"] = "-"
        xsbr = analysis.extractXSBR(data, mass=opts.mass, analysis=opts.analysis)
        data.loc[sig, "xsbr"] = data[sig].apply(lambda x: xsbr[f"XS_{x['procOriginal']}"] * xsbr["BR"], axis=1)
        data["ea"] = "-"
        data.loc[sig, "ea"] = data.loc[sig, "nominal_yield"]
        data.loc[sig, "true_yield"] = data[sig].apply(lambda x: x["xsbr"] * x["ea"] * x["rate"], axis=1)
        totals = {cat: data[(data["cat"] == cat) & sig].true_yield.sum() for cat in data.cat.unique()}
        yields = data["true_yield"]
    else:
        totals = {cat: data[(data["cat"] == cat) & sig].nominal_yield.sum() for cat in data.cat.unique()}
        yields = pd.to_numeric(data["nominal_yield"], errors="coerce")
    small = yields < opts.prune_threshold * data.apply(lambda x: totals[x["cat"]], axis=1)
    data.loc[small & sig & (~data["cat"].str.contains("NOTAG")), "prune"] = 1
    data.loc[data["cat"].str.contains("NOTAG"), "prune"] = 1
    return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Writing


def _calendar_years(years: str) -> list[str]:
    return sorted({m.group() for y in years.split(",") if (m := re.search(r"\d{4}", y))})


def _is_zmmg_2022_2023(year: str) -> bool:
    return str(year).strip().startswith(("2022", "2023", "2223"))


def _is_zmmg_2024(year: str) -> bool:
    return str(year).strip().startswith("2024")


def _format_value(line, value, title, proc, cat, n_events):
    """Append one lnN entry: '-' for no effect, below MIN_EVENTS events (except lumi) or negative values."""
    if isinstance(value, str):
        return line + "%-15s " % value
    if not isinstance(value, list):
        raise TypeError(f"Systematic {title}: value {value!r} for ({proc}, {cat}) is neither a string nor a list")
    few = n_events < MIN_EVENTS and "lumi" not in title
    if len(value) == 1:
        if abs(value[0] - 1) < 0.0005 or few:
            return line + "%-15s " % "-"
        if value[0] < 0.0:
            logger.warning("Systematic %s: negative variation for (%s, %s)", title, proc, cat)
            return line + "%-15s " % "-"
        return line + "%-15.3f " % value[0]
    if len(value) == 2:
        if (abs(value[0] - 1) < 0.0005 and abs(value[1] - 1) < 0.0005) or few:
            return line + "%-15s " % "-"
        if value[0] < 0.0 or value[1] < 0.0:
            logger.warning("Systematic %s: negative variation for (%s, %s)", title, proc, cat)
            return line + "%-15s " % "-"
        if abs(value[0] * value[1] - 1) < 0.0005:
            return line + "%-15.3f " % value[1]
        return line + "%-15s " % ("%.3f/%.3f" % (value[0], value[1]))
    return line


def _rows(data):
    for cat in data.cat.unique():
        for _, row in data[data["cat"] == cat].iterrows():
            if row["proc"] != "data_obs":
                yield cat, row


def _systematic_value(row, column, tier):
    if column in row.index:
        return row[column]
    if tier == "shape" and f"{column}_shape" in row.index:
        return row[f"{column}_shape"]
    return "-"


def write_preamble(f, opts):
    f.write(f"CMS HGG Datacard - {opts.years} - {SQRTS}\n")
    f.write("Auto-generated by flashggFinalFits/Datacard/makeDatacard.py\n")
    f.write("Run with: combine\n")
    f.write("---------------------------------------------\n")
    f.write("imax *\njmax *\nkmax *\n")
    f.write("---------------------------------------------\n")


def write_processes(f, data, opts):
    f.write("\n")
    for cat in data.cat.unique():
        for year in _calendar_years(opts.years):
            for _, row in data[data["cat"] == cat].iterrows():
                if year not in str(row["year"]) and row["year"] != "merged":
                    continue
                model = row["model"]
                if row["proc"] == "bkg_mass":
                    index = model.rfind("_13TeV_bkgshape")
                    if index == -1:
                        continue
                    model = f"{model[:index]}_{year}_13TeV_bkgshape"
                f.write("shapes      %-55s %-40s %s %s\n" % (row["proc"], row["cat"], row["modelWSFile"], model))
    lines = {
        k: "%-30s" % label
        for k, label in [
            ("bin_cat", "bin"),
            ("obs", "observation"),
            ("bin", "bin"),
            ("proc", "process"),
            ("id", "process"),
            ("rate", "rate"),
        ]
    }
    for cat in data.cat.unique():
        lines["bin_cat"] += "%-55s " % cat
        lines["obs"] += "%-55s " % "-1"
        signal_id = 0
        for _, row in data[data["cat"] == cat].iterrows():
            if row["proc"] == "data_obs":
                continue
            lines["bin"] += "%-55s " % cat
            lines["proc"] += "%-55s " % row["proc"]
            if row["proc"] == "bkg_mass":
                lines["id"] += "%-55s " % "1"
            else:
                lines["id"] += "%-55s " % signal_id
                signal_id -= 1
            lines["rate"] += ("%-55.1f " if row["rate"] == 1.0 else "%-55.7f ") % row["rate"]
    f.write("\n")
    # every line of this block, the breaks included, loses its last character (the trailing space)
    for key in [None, "bin_cat", "obs", None, "bin", "proc", "id", "rate", None]:
        f.write((LINE_BREAK if key is None else lines[key])[:-1] + "\n")
    f.write("\n")


def write_systematic(f, data, syst, opts):
    if syst["type"] == "signal_shape":
        title = syst["title"]
        if syst.get("name") in ("ScaleEBZmmg", "ScaleEEZmmg") or title.endswith(("ScaleEBZmmg", "ScaleEEZmmg")):
            years = opts.years.split(",")
            if any(_is_zmmg_2022_2023(y) for y in years):
                f.write("%-70s  param    %-6s %-6s\n" % (f"{title}_2022_2023", syst["mean"], syst["sigma"]))
            if any(_is_zmmg_2024(y) for y in years):
                f.write("%-70s  param    %-6s %-6s\n" % (f"{title}_2024", syst["mean"], syst["sigma"]))
        elif syst["mode"] in ("scales", "smears"):
            for year in opts.years.split(","):
                f.write("%-70s  param    %-6s %-6s\n" % (f"{title}_{year}", syst["mean"], syst["sigma"]))
        else:
            f.write("%-70s  param    %-6s %-6s\n" % (title, syst["mean"], syst["sigma"]))
        return
    for tier in [t for t in syst.get("tiers", []) if t != "mnorm"] or [""]:
        suffix = "" if tier in ("", "shape") else f"_{tier}"
        if syst["correlateAcrossYears"] in (1, -1):
            title = f"{syst['title']}{suffix}"
            line = "%-50s  %-10s    " % (title, syst["prior"])
            for cat, row in _rows(data):
                value = _systematic_value(row, f"{syst['name']}{suffix}", tier)
                line = _format_value(line, value, title, row["proc"], cat, row["numEvents"])
            f.write(line[:-1] + "\n")
        else:
            for year in opts.years.split(","):
                title = f"{syst['title']}{suffix}_{year}"
                line = "%-50s  %-10s    " % (title, syst["prior"])
                for cat, row in _rows(data):
                    value = _systematic_value(row, f"{syst['name']}{suffix}_{year}", "shape")
                    line = _format_value(line, value, title, row["proc"], cat, row["numEvents"])
                f.write(line[:-1] + "\n")


def write_mc_stat(f, data, opts):
    for stat_cat in data.cat.unique():
        for year in opts.years.split(","):
            mask = (data["year"] == year) & (data["cat"] == stat_cat) & (data["type"] == "sig")
            sumw, sumw2 = data[mask]["nominal_yield"].sum(), data[mask]["sumw2"].sum()
            value = [1.0] if sumw <= 0 else [1 + (math.sqrt(sumw2) / sumw)]
            title = f"MCStat_{year}_{stat_cat}"
            line = "%-50s  %-10s    " % (title, "lnN")
            for cat, row in _rows(data):
                if year not in row["proc"] or row["proc"] == "bkg_mass":
                    line += "%-15s " % "-"
                    continue
                line = _format_value(line, value if cat == stat_cat else "-", title, row["proc"], cat, row["numEvents"])
            f.write(line[:-1] + "\n")


def write_pdf_indices(f, data, opts):
    f.write("\n")
    for cat in data[~data["cat"].str.contains("NOTAG")].cat.unique():
        for year in _calendar_years(opts.years):
            f.write("%-55s  discrete\n" % f"pdfindex_{cat}_{year}_13TeV")


def make_datacard(yields_dir: str, output_file: str, opts: DatacardOptions, dataframe_file: str | None = None):
    """Write the datacard of the yields in ``yields_dir``; returns the final DataFrame."""
    data = load_yields(yields_dir)
    years = opts.years.split(",")
    experimental, theory, signal_shape = analysis.datacard_systematics()
    experimental = experimental_systematics_for(opts.variable, experimental)
    if opts.do_systematics:
        mask = _signal(data)
        experimental_types = {
            s["name"]: "a_h" if s["name"] == "JetHEM" else factory_type(data[mask], s)
            for s in experimental
            if s["type"] == "factory"
        }
        theory_types = {s["name"]: factory_type(data[mask], s) for s in theory if s["type"] == "factory"}
        for s in experimental:
            if s["type"] == "constant":
                data = add_constant_systematic(data, s, years)
        data = experimental_factory(data, experimental, experimental_types, years)
        for s in theory:
            if s["type"] == "constant":
                data = add_constant_systematic(data, s, years)
        data = theory_factory(data, theory, theory_types, years, opts.skip_cow_corr)
        data, theory = group_systematics(data, theory, "weight_LHEScal", SCALE_GROUPINGS)
        for s in theory:
            s["title"] = s["title"].replace("scaleWeight", "scale")
    if opts.prune:
        data = prune(data, opts)
    if dataframe_file:
        os.makedirs(os.path.dirname(dataframe_file), exist_ok=True)
        with open(dataframe_file, "wb") as f:
            pickle.dump(data, f)

    written = data[data["prune"] == 0] if opts.prune else data
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        write_preamble(f, opts)
        write_processes(f, written, opts)
        if opts.do_systematics:
            for group in (experimental, theory, signal_shape):
                for s in group:
                    write_systematic(f, written, s, opts)
                if group is not signal_shape:
                    f.write(LINE_BREAK + "\n")
        if opts.do_mc_stat:
            f.write(LINE_BREAK + "\n")
            write_mc_stat(f, written, opts)
        f.write(LINE_BREAK + "\n")
        write_pdf_indices(f, written, opts)
    return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cleaning


def _is_diagonal(proc: str, cat: str) -> bool:
    """Whether a signal process belongs to the category (the reco bin of its particle-level bin)."""
    if proc.startswith(("ggh_", "vbf_", "vh_", "tth_")):
        parts = proc.split("_")
        if parts and parts[-1] == "hgg":
            parts = parts[:-1]
        if parts and parts[-1].startswith(("20", "22")):
            parts = parts[:-1]
        if parts and parts[-1] in ("in", "out"):
            parts = parts[:-1]
        if parts and parts[-1].startswith(("20", "22")):
            parts = parts[:-1]
        return len(parts) >= 2 and f"RECO_{'_'.join(parts[1:])}" in cat
    rules = {
        "qqH_hgg": "RECO_VBFTOPO",
        "ttH_hgg": "TTH",
        "WH_had_hgg": "RECO_VHHAD",
        "ZH_had_hgg": "RECO_VHHAD",
        "WH_lep_hgg": "LEP",
        "ZH_lep_hgg": "LEP",
    }
    return any(p in proc and c in cat for p, c in rules.items())


def _clean_value(name, effect, proc, cat, opts):
    values = effect.split("/")
    if len(values) not in (1, 2):
        raise ValueError(f"lnN value {effect} of {name} should have one or two entries")
    keep = "lumi" in name
    if opts["remove_non_diagonal"] and not _is_diagonal(proc, cat):
        keep_value = keep or "bkg_mass" in proc
        if len(values) == 1:
            return "- " if values[0] == "-" or not keep_value else "%1.3f " % float(values[0])
        return "%1.3f/%1.3f " % (float(values[0]), float(values[1])) if keep_value else "- "
    if len(values) == 1:
        if values[0] == "-":
            return "- "
        value = float(values[0])
        if not keep and (value < 1.0 / opts["factor"] or value > opts["factor"]):
            return "- "
        return "%1.3f " % value
    low, high = float(values[0]), float(values[1])
    if keep:
        return "%1.3f/%1.3f " % (low, high)
    if low < 1.0 / opts["factor"] or low > opts["factor"]:
        low = 1
    if high < 1.0 / opts["factor"] or high > opts["factor"]:
        high = 1
    if name in opts["symmetrize"] and min(abs(low - 1.0), abs(high - 1.0)) < 0.003 and abs(low - high) >= 0.010:
        # one-sided: symmetric kappa of the side that deviates, in the direction of the up variation
        return "%1.3f " % (high if abs(high - 1.0) >= abs(low - 1.0) else 1.0 / low)
    if opts["remove_double_sided"] and high > 1.000001 and low > 1.000001:
        return "%1.3f " % max(high, low)
    if opts["remove_double_sided"] and high < 0.999999 and low < 0.999999:
        return "%1.3f " % min(high, low)
    return "%1.3f/%1.3f " % (low, high)


def clean_datacard(
    datacard: str,
    output: str | None = None,
    factor: float = 5.0,
    remove_double_sided=False,
    remove_non_diagonal=False,
    symmetrize: str = "",
) -> str:
    """Remove lnN values beyond factor (or below 1/factor), optionally symmetrise one-sided ones.

    Writes ``<datacard>_cleaned.txt`` by default.
    """
    output = output or datacard.replace(".txt", "_cleaned.txt")
    opts = {
        "factor": float(factor),
        "remove_double_sided": remove_double_sided,
        "remove_non_diagonal": remove_non_diagonal,
        "symmetrize": [s for s in symmetrize.split(",") if s],
    }
    procs, cats, lnn_names = [], [], set()
    with open(datacard) as f_in, open(output, "w") as f_out:
        for line in f_in:
            values = line.split()
            if len(values) < 2 or values[1] != "lnN" and values[0] not in ("process", "bin"):
                f_out.write(line)
                continue
            if values[0] == "process":
                procs = procs or values[1:]
                f_out.write(line)
                continue
            if values[0] == "bin":
                cats = values[1:]
                f_out.write(line)
                continue
            name = values[0]
            lnn_names.add(name)
            cleaned = line.split("lnN")[0] + "lnN   "
            for i, effect in enumerate(values[2:]):
                cleaned += _clean_value(name, effect, procs[i], cats[i], opts)
            f_out.write(cleaned + "\n")
    missing = [s for s in opts["symmetrize"] if s not in lnn_names]
    if missing:
        logger.warning("Nuisances to symmetrise without lnN line in %s: %s", datacard, ", ".join(missing))
    return output
