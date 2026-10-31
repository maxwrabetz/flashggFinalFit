"""Statistical interpretation with combine.

combine, combineTool.py and hadd run as external commands; this module builds those commands (the law tasks of
Combine/law_combine.py run them) and holds the helpers around them:

* :func:`text2workspace_command`: datacard -> workspace with the physics model of ``Combine/models.py``
* Asimov fits: :func:`asimov_first_step_command` (best fit seeding the scans) and :func:`asimov_scan_command`
* fits to data: :func:`observed_best_fit_command` and :func:`observed_scan_command`
* impacts in three steps: initial fit, one fit per nuisance (``*_impacts_parameter_fit_command``) and
  :func:`impacts_collect_command`
* :func:`hesse_command`: robust Hesse for the covariance and correlation matrices
* toys for the bands of the mass distributions: :func:`toy_generate_command`, :func:`toy_fit_command`,
  :func:`toy_throw_command`
* goodness of fit of the differential measurement: :func:`pvalue_fit_command` and :func:`pvalue`
* :func:`parameter_range_for_poi`, :func:`scan_grid_configuration`, :func:`modelconfig_nuisances`,
  :func:`free_parameters`, :func:`fitted_pdf_indices`, ...

The order of the options in the commands is the one of the original law tasks. All scans and stat-only fits
start from the snapshot ``MultiDimFit`` of a best fit; ``inclusive=True`` selects the commands of the inclusive
measurement (one POI ``r``), else the POIs are the particle-level bins of the differential variable.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess

from finalfits import analysis

logger = logging.getLogger(__name__)

HIGGS_MASS = "125.07"
COMMON_T2W_OPTIONS = f"-m {HIGGS_MASS} higgsMassRange=122,128"
SNAPSHOT = "MultiDimFit"

MULTIDIMFIT = ["-M", "MultiDimFit"]
MASS = ["-m", HIGGS_MASS]
FREEZE_MH = ["--freezeParameters", "MH"]
FREEZE_STAT_ONLY = ["--freezeParameters", "allConstrainedNuisances,MH"]
MINIMIZER = ["--cminDefaultMinimizerStrategy=0", "--cminFallbackAlgo", "Minuit2,Migrad,1:10"]
# discrete profiling of the background envelopes (pdfindex categories)
DISCRETE_PROFILING = [
    token
    for flag in (
        "MINIMIZER_freezeDisassociatedParams",
        "MINIMIZER_multiMin_hideConstants",
        "MINIMIZER_multiMin_maskConstraints",
        "MINIMIZER_multiMin_maskChannels=2",
    )
    for token in ("--X-rtd", flag)
]
ASIMOV = ["-t", "-1"]
TOY_OPTIONS = ["--toysFrequentist", "--bypassFrequentistFit"]


def physics_model_options(mode: str) -> list[str]:
    """text2workspace options of a physics model (e.g. NJ: one POI per particle-level bin)."""
    models = analysis.combine_models()
    if mode not in models:
        raise KeyError(f"Physics model {mode} is not defined in Combine/models.py")
    # the models are written for the shell, with quoted --PO arguments
    return shlex.split(models[mode])


def text2workspace_command(datacard: str, workspace: str, mode: str, common_options: str = COMMON_T2W_OPTIONS):
    return ["text2workspace.py", datacard, "-o", workspace, *shlex.split(common_options), *physics_model_options(mode)]


def run(command: list[str]) -> None:
    """Run a combine command, raising if it fails (its output is logged)."""
    logger.info("Running: %s", shlex.join(command))
    result = subprocess.run(command, text=True, capture_output=True)
    logger.debug(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stderr[-5000:]}")


def output_file(name: str, method: str = "MultiDimFit") -> str:
    """File name combine writes for ``-n <name> -M <method>`` at the Higgs mass."""
    return f"higgsCombine{name}.{method}.mH{HIGGS_MASS}.root"


# ------------------------------------------------------------------------------------------------ fit set-up


def parameter_range_for_poi(range_expression, poi, range_overrides=None):
    """Map a generic ``r=min,max`` range to the POI scanned in this branch.

    Combine separates multiple parameter ranges with colons. Literal ranges for parameters other than ``r`` are
    retained, while a generic ``r`` entry is replaced by the differential POI name, e.g. ``r_PTH_350p0_450p0``.
    An optional per-POI override (``min,max`` or a full range expression) allows narrow refinement scans for
    selected differential bins without changing the range used by all other bins.
    """
    if range_overrides and poi in range_overrides:
        bounds = str(range_overrides[poi]).strip()
        range_expression = bounds if "=" in bounds else f"{poi}={bounds}"

    mapped_ranges = []
    for item in str(range_expression).split(":"):
        item = item.strip()
        if not item:
            continue
        name, separator, bounds = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid parameter range expression: {item}")
        mapped_ranges.append(f"{poi if name.strip() == 'r' else name.strip()}={bounds.strip()}")
    if not mapped_ranges:
        raise ValueError("Parameter range expression must not be empty")
    return ":".join(mapped_ranges)


def scan_grid_configuration(n_points, grid_points, first_point, last_point):
    """Return the full grid size and the subset of grid indices to evaluate.

    By default all ``n_points`` are evaluated. A larger ``grid_points`` value together with first/last indices
    permits a high-resolution local refinement while the parameter range still contains the global best fit
    needed for a consistently normalized deltaNLL.
    """
    n_points, grid_points, first_point, last_point = (int(n) for n in (n_points, grid_points, first_point, last_point))
    total_points = grid_points if grid_points > 0 else n_points
    if first_point < 0 and last_point < 0:
        return total_points, list(range(n_points))
    if first_point < 0 or last_point < first_point or last_point >= total_points:
        raise ValueError(
            "invalid scan grid subset: require 0 <= first_point <= last_point "
            f"< {total_points}, got {first_point}..{last_point}"
        )
    return total_points, list(range(first_point, last_point + 1))


def _parse_exclude(exclude):
    """Literal names and regular expressions (``rgx{...}`` as in combineTool.py, or ``/.../``) of a csv list."""
    names, patterns = set(), []
    for token in (t.strip() for t in (exclude or "").split(",")):
        if not token:
            continue
        if token.startswith("rgx{") and token.endswith("}"):
            pattern = token[4:-1]
        elif len(token) >= 2 and token.startswith("/") and token.endswith("/"):
            pattern = token[1:-1]
        else:
            names.add(token)
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            names.add(token)
    return names, patterns


def filter_excluded(names, exclude):
    """Drop the names matching the exclude expression of ``combine_impacts.exclude``."""
    excluded, patterns = _parse_exclude(exclude)
    return [n for n in names if n not in excluded and not any(p.search(n) for p in patterns)]


def _model_config(workspace_file):
    import ROOT

    f = ROOT.TFile.Open(workspace_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open workspace file: {workspace_file}")
    w = f.Get("w")
    if not w:
        raise RuntimeError(f"Workspace 'w' not found in: {workspace_file}")
    config = w.genobj("ModelConfig")
    if not config:
        raise RuntimeError(f"ModelConfig not found in workspace 'w' in: {workspace_file}")
    return f, config


def _floating(variables, pois):
    return [
        v.GetName()
        for v in variables
        if v.GetName() not in pois and not v.isConstant() and v.InheritsFrom("RooRealVar")
    ]


def modelconfig_nuisances(workspace_file, pois, exclude=""):
    """Sorted floating nuisances of the ModelConfig, without POIs and excluded ones.

    Only these have a meaningful +-1 sigma crossing for the impacts; the background envelope parameters
    (env_pdf_*) are no nuisances of the ModelConfig.
    """
    f, config = _model_config(workspace_file)
    nuisances = config.GetNuisanceParameters()
    names = _floating(nuisances, pois) if nuisances else []
    f.Close()
    return sorted(filter_excluded(names, exclude))


def free_parameters(workspace_file, pois):
    """Floating parameters of the pdf other than the POIs, as the --allPars of combineTool.py Impacts."""
    f, config = _model_config(workspace_file)
    names = _floating(config.GetPdf().getParameters(config.GetObservables()), pois)
    f.Close()
    return names


def fitted_pdf_indices(fit_file, macro):
    """``pdfindex_X=i,...`` of a fit result (Combine/checkPdfIdx.C on its limit tree), None if it cannot be read."""
    result = subprocess.run(f"root -l -q '{macro}(\"{fit_file}\")'", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Could not read the pdf indices of %s: %s", fit_file, result.stderr)
        return None
    # the macro prints the values on a line starting with X (after the "Processing ..." line of ROOT)
    marked = [line for line in result.stdout.splitlines() if line.startswith("X")]
    if not marked:
        logger.error("Could not read the pdf indices of %s: %s", fit_file, result.stderr)
        return None
    return marked[-1][1:].strip().rstrip(",")


def best_fit_values(fit_file, pois):
    """``poi=value`` of the first entry of the limit tree of a fit, for --setParameters."""
    import ROOT

    f = ROOT.TFile(fit_file)
    tree = f.Get("limit")
    if not tree:
        raise RuntimeError(f"Tree 'limit' not found in {fit_file}")
    values = []
    for poi in pois:
        if not hasattr(tree, poi):
            raise RuntimeError(f"Branch '{poi}' not found in the tree of {fit_file}")
        tree.GetEntry(0)
        values.append(f"{poi}={getattr(tree, poi)}")
    return ",".join(values)


def snapshot_value(workspace_file, poi, snapshot=SNAPSHOT):
    """Value of a POI in a snapshot of the workspace saved by a fit."""
    import ROOT

    f = ROOT.TFile(workspace_file)
    w = f.Get("w")
    w.loadSnapshot(snapshot)
    return w.var(poi).getVal()


# ------------------------------------------------------------------------------------------------ Asimov fits


def _save_index(save_index):
    return ["--saveSpecifiedIndex", save_index] if save_index else []


def asimov_first_step_command(workspace, branch, save_index, poi=None, set_parameters=""):
    """Asimov best fit; its workspace (with the snapshot and pdf indices) seeds the scans. ``poi=None``: inclusive."""
    command = [
        "combine", *MULTIDIMFIT, workspace, *FREEZE_MH, *MASS, "-n", f"firstStep_{branch}", *MINIMIZER,
        "--expectSignal", "1", "--saveWorkspace", *DISCRETE_PROFILING, *ASIMOV,
    ]  # fmt: skip
    if poi is None:
        return [*command, "--saveFitResult", "--saveSpecifiedIndex", save_index, "--floatOtherPOIs", "1"]
    return [
        *command, "-P", poi, "--saveFitResult", "--floatOtherPOIs", "1", *_save_index(save_index),
        "--setParameters", set_parameters,
    ]  # fmt: skip


def asimov_scan_command(
    first_step, poi, point, n_points, parameter_ranges, set_parameters, stat_only=False, save_index="", inclusive=False
):
    """One point of the Asimov likelihood scan of a POI, from the first-step snapshot (stat-only: nuisances frozen)."""
    name = f"AsimovPostFitScan{'Stat' if stat_only else 'Fit'}_{poi}.POINTS.{point}.{point}"
    grid = [
        *(FREEZE_STAT_ONLY if stat_only else FREEZE_MH), *MASS, "-n", name, *MINIMIZER,
        "--algo", "grid", "--points", str(n_points), "--expectSignal", "1", *DISCRETE_PROFILING, *ASIMOV,
        "-P", "r" if inclusive else poi, "--firstPoint", str(point), "--lastPoint", str(point),
    ]  # fmt: skip
    if inclusive:
        return [
            "combine", *MULTIDIMFIT, "-d", first_step, "--snapshotName", SNAPSHOT, *grid, "--floatOtherPOIs", "1",
            "--alignEdges", "1", "--setParameterRanges", parameter_ranges, "--saveSpecifiedNuis", "all",
            "--setParameters", set_parameters,
        ]  # fmt: skip
    return [
        "combineTool.py", *MULTIDIMFIT, "-d", first_step, *grid, "--saveFitResult", "--floatOtherPOIs", "1",
        "--alignEdges", "1", "--snapshotName", SNAPSHOT, "--setParameters", set_parameters,
        "--setParameterRanges", parameter_ranges, *_save_index(save_index),
    ]  # fmt: skip


def hadd_command(target, sources):
    return ["hadd", "-f", target, *sources]


def scan_plot_command(scan, stat_scan, poi, output, main_label, translate, launcher=("plot1DScan.py",)):
    """plot1DScan.py of CombineHarvester: scan with all uncertainties and the stat-only one."""
    return [
        *launcher, scan, "-o", output, "--POI", poi, "--others", f"{stat_scan}:stat-only:2",
        "--main-label", main_label, "--translate", translate,
    ]  # fmt: skip


# ------------------------------------------------------------------------------------------------ fits to data


def observed_best_fit_command(
    workspace, poi, prefit_tolerance, stat_only=False, inclusive=False, r_range=("0.7", "1.6"), save_index=""
):
    """Best fit (singles) to data whose snapshot seeds the scans; ``stat_only``: refit starting from that snapshot."""
    name = f"DataPostFitBestFit{'Stat' if stat_only else ''}_{poi}"
    r_options = ["--rMin", str(r_range[0]), "--rMax", str(r_range[1])]
    tolerance = ["--cminApproxPreFitTolerance", str(prefit_tolerance)]
    if stat_only:
        tool = "combine" if inclusive else "combineTool.py"
        command = [tool, *MULTIDIMFIT, workspace, *FREEZE_MH, *MASS, "-n", name, *MINIMIZER, "--algo", "singles"]
        snapshot = ["--snapshotName", SNAPSHOT, "-w", "w"]
        if inclusive:
            return [
                *command, *r_options, *DISCRETE_PROFILING, "-P", poi, "--floatOtherPOIs", "1", "--saveWorkspace",
                "--saveFitResult", *snapshot, *tolerance,
            ]  # fmt: skip
        return [
            *command, *DISCRETE_PROFILING, "-P", poi, "--saveFitResult", "--floatOtherPOIs", "1", "--saveWorkspace",
            *snapshot,
        ]  # fmt: skip
    if inclusive:
        return [
            "combine", *MULTIDIMFIT, "-d", workspace, *FREEZE_MH, *MASS, "-n", name, *MINIMIZER, "--algo", "singles",
            *DISCRETE_PROFILING, "-P", "r", "--floatOtherPOIs", "1", "--saveWorkspace", "--saveFitResult",
            *tolerance, *r_options,
        ]  # fmt: skip
    return [
        "combine", *MULTIDIMFIT, workspace, *FREEZE_MH, *MASS, "-n", name, *MINIMIZER, "--algo", "singles",
        *DISCRETE_PROFILING, "-P", poi, *tolerance, "--saveFitResult", "--floatOtherPOIs", "1", "--saveWorkspace",
        "--saveSpecifiedIndex", save_index,
    ]  # fmt: skip


def observed_scan_command(
    snapshot_file,
    poi,
    point,
    n_points,
    stat_only=False,
    inclusive=False,
    r_range=("0.7", "1.6"),
    parameter_ranges="",
    prefit_tolerance=0.01,
    save_index="",
    set_parameters="",
):
    """One point of the likelihood scan of a POI in data, from the snapshot of the (stat-only) best fit."""
    name = f"DataPostFitScan{'Stat' if stat_only else 'Fit'}_{poi}.POINTS.{point}.{point}"
    head = [
        "combine" if inclusive else "combineTool.py", *MULTIDIMFIT, "-d", snapshot_file,
        *(FREEZE_STAT_ONLY if stat_only else FREEZE_MH), *MASS, "-n", name, *MINIMIZER, "--algo", "grid",
    ]  # fmt: skip
    points = ["--points", str(n_points)]
    grid = ["--alignEdges", "1", "--firstPoint", str(point), "--lastPoint", str(point)]
    if inclusive:
        r_options = ["--rMin", str(r_range[0]), "--rMax", str(r_range[1])]
        return [
            *head, *([*r_options, *points] if stat_only else [*points, *r_options]), *DISCRETE_PROFILING,
            "-P", poi if stat_only else "r", "--floatOtherPOIs", "1", "--saveWorkspace", "--saveFitResult",
            "--snapshotName", SNAPSHOT, *grid, *([] if stat_only else ["--setParameterRanges", parameter_ranges]),
            "--cminApproxPreFitTolerance", str(prefit_tolerance), "-w", "w",
        ]  # fmt: skip
    return [
        *head, *points, *DISCRETE_PROFILING, "-P", poi, "--saveWorkspace", "--saveFitResult", "--floatOtherPOIs", "1",
        "--snapshotName", SNAPSHOT, *grid,
        *([] if stat_only else ["--saveSpecifiedIndex", save_index, "--setParameters", set_parameters]), "-w", "w",
    ]  # fmt: skip


# ------------------------------------------------------------------------------------------------ impacts


def asimov_impacts_initial_fit_command(workspace, set_parameters, pois=None):
    """First impacts step on the Asimov data set; ``pois=None``: inclusive (combineTool.py --doInitialFit)."""
    tail = [*MINIMIZER, *DISCRETE_PROFILING, *ASIMOV, "--setParameters", set_parameters]
    if pois is None:
        return [
            "combineTool.py", "-M", "Impacts", "-d", workspace, "--doInitialFit", "--robustFit", "1",
            *FREEZE_MH, *MASS, *tail,
        ]  # fmt: skip
    return [
        "combine", *MULTIDIMFIT, "-d", workspace, "--algo", "singles", "--redefineSignalPOIs", ",".join(pois),
        *FREEZE_MH, *MASS, "-n", "_initialFit_Test", *tail,
    ]  # fmt: skip


def observed_impacts_initial_fit_command(workspace, pois=None, prefit_tolerance=0.01, set_parameter=1.0):
    """First impacts step on data; ``pois=None``: inclusive (combineTool.py --doInitialFit)."""
    if pois is None:
        return [
            "combineTool.py", "-M", "Impacts", "-d", workspace, "--doInitialFit", "--robustFit", "1",
            *FREEZE_MH, *MASS, *MINIMIZER, *DISCRETE_PROFILING,
            "--cminApproxPreFitTolerance", str(prefit_tolerance), "--setParameter", str(set_parameter),
        ]  # fmt: skip
    return [
        "combine", *MULTIDIMFIT, "-d", workspace, "--algo", "singles", "--redefineSignalPOIs", ",".join(pois),
        *FREEZE_MH, *MASS, "--robustFit", "1", "-n", "_initialFit_Test", *MINIMIZER, *DISCRETE_PROFILING,
    ]  # fmt: skip


def _impacts_parameter_fit(workspace, parameter, pois, options):
    return [
        "combine", *MULTIDIMFIT, "-d", workspace, "--algo", "impact", "--redefineSignalPOIs", ",".join(pois),
        *options[0], "--floatOtherPOIs", "1", "--saveInactivePOI", "1", "--robustFit", "1",
        "-n", f"_paramFit_Test_{parameter}", *MINIMIZER, *DISCRETE_PROFILING, *options[1],
    ]  # fmt: skip


def asimov_impacts_parameter_fit_command(workspace, parameter, pois, set_parameters):
    """Second impacts step on the Asimov data set: fits with one nuisance at its +-1 sigma values."""
    return _impacts_parameter_fit(
        workspace, parameter, pois, ([*FREEZE_MH, *MASS, "-P", parameter], [*ASIMOV, "--setParameters", set_parameters])
    )


def observed_impacts_parameter_fit_command(workspace, parameter, pois, best_fit, inclusive_options=None):
    """Second impacts step on data, starting from the best-fit POIs; ``inclusive_options``: inclusive fit options."""
    fit = [*FREEZE_MH, *MASS, "-P", parameter]
    best = ["--setParameters", best_fit]
    if inclusive_options is not None:
        return _impacts_parameter_fit(workspace, parameter, pois, ([*fit, *best], inclusive_options))
    return _impacts_parameter_fit(workspace, parameter, pois, ([*best, *fit], []))


def impacts_collect_command(
    workspace, output="impacts/impacts.json", named=None, exclude="", set_parameters=None, freeze_mh=True
):
    """Third impacts step: collect the fits of the nuisances (``named``, default all free parameters) in a json."""
    command = ["combineTool.py", "-M", "Impacts", "-d", workspace, *(FREEZE_MH if freeze_mh else []), *MASS]
    command += ["-o", output]
    if named is not None:
        command += ["--named", ",".join(named)]
    if exclude:
        command += ["--exclude", exclude]
    if set_parameters is not None:
        command += ["--setParameters", set_parameters]
    return command


def correct_impacts_command(script, impacts_json, exclude=""):
    """Plots/correctImpacts.py: drop MH and the background model parameters from the impacts json."""
    command = ["python3", script, "--impactsJson", impacts_json, "--frozenParam", "MH", "--dropBkgModelParams"]
    return command + (["--exclude", exclude] if exclude else [])


def plot_impacts_command(impacts_json, output, poi=None, translate=None, blind=False):
    """plotImpacts.py of CombineHarvester."""
    command = ["plotImpacts.py", "-i", impacts_json, "-o", output]
    if poi is not None:
        command += ["--POI", poi]
    if translate is not None:
        command += ["--translate", translate]
    return command + (["--blind"] if blind else [])


# ------------------------------------------------------------------------------------------------ Hesse


def hesse_command(workspace, name, save_index, asimov_parameters=None):
    """Best fit with robust Hesse (saved) for the covariance matrix; Asimov if ``asimov_parameters`` is given."""
    command = [
        "combine", *MULTIDIMFIT, workspace, *FREEZE_MH, *MASS, "-n", name, "--saveWorkspace", "--saveFitResult",
        "--saveSpecifiedIndex", save_index, "--floatOtherPOIs", "1", "--robustHesse", "1", "--robustHesseSave", "1",
        *MINIMIZER, *DISCRETE_PROFILING,
    ]  # fmt: skip
    if asimov_parameters is None:
        return command
    return [*command, "--X-rtd", "MINIMIZER_skipDiscreteIterations", *ASIMOV, "--setParameters", asimov_parameters]


def corr_matrix_command(
    script,
    input_json,
    mode,
    hesse_file,
    output,
    translate,
    year,
    covariance=False,
    observed=False,
    no_preliminary=False,
):
    """Plots/makeCorrMatrix.py: correlation (or covariance) matrix of the POIs from the robust Hesse."""
    command = [
        "python3", script, "--inputJson", input_json, "--mode", mode, "--input", hesse_file, "--output", output,
        "--translate", translate, "--year", year,
    ]  # fmt: skip
    flags = (("--doCov", covariance), ("--doObserved", observed), ("--noPreliminary", no_preliminary))
    return command + [flag for flag, enabled in flags if enabled]


# ------------------------------------------------------------------------------------------------ toys


def toy_generate_command(toy, input_file, set_parameters, snapshot=None):
    """Toy of the S+B model (one per job, random seed); ``snapshot``: from the post-fit workspace, else prefit."""
    options = [
        "--saveWorkspace", *TOY_OPTIONS, "-t", "1", "-s", "-1", "-n", f"_{toy}_gen_step",
        "--setParameters", set_parameters,
    ]  # fmt: skip
    if snapshot is not None:
        return ["combine", input_file, "-M", "GenerateOnly", *MASS, *options, "--snapshotName", snapshot]
    return ["combine", "-M", "GenerateOnly", "-d", input_file, *MASS, *options]


def toy_fit_command(toy, poi, method, set_parameters, postfit=False):
    """Fit of the generated toy (gen_<toy>.root)."""
    head = [
        "combine", f"gen_{toy}.root", *MASS, "-M", method, "-P", poi, "--floatOtherPOIs=1", "--saveWorkspace",
        *TOY_OPTIONS, "-t", "1",
    ]  # fmt: skip
    seed = ["-s", "-1", "-n", f"_{toy}_fit_step"]
    minimizer = ["--cminDefaultMinimizerStrategy", "0", *DISCRETE_PROFILING]
    parameters = ["--setParameters", set_parameters]
    return [*head, *parameters, *seed, *minimizer] if postfit else [*head, *seed, *minimizer, *parameters]


def toy_throw_command(toy, snapshot, set_parameters):
    """Asimov data set of the toy fit (fit_<toy>.root) with the signal set by ``set_parameters``."""
    return [
        "combine", f"fit_{toy}.root", *MASS, "--snapshotName", snapshot, "-M", "GenerateOnly", "--saveToys",
        *TOY_OPTIONS, "-t", "-1", "-n", f"_{toy}_throw_step", "--setParameters", set_parameters,
    ]  # fmt: skip


def mgg_best_fit_command(workspace, poi, method):
    """Best fit to data whose workspace (snapshot ``method``) is used for the post-fit mass distributions."""
    return [
        "combine", "--floatOtherPOIs", "1", "-P", poi, *FREEZE_MH, "--saveInactivePOI", "1", "--saveWorkspace",
        "--saveSpecifiedNuis", "all", "--cminDefaultMinimizerStrategy", "0", *DISCRETE_PROFILING,
        "-M", method, *MASS, "-d", workspace, "-n", f"_bestfit_syst_obs_{poi}",
    ]  # fmt: skip


# ------------------------------------------------------------------------------------------------ p-value


def pvalue_fit_command(scan_file, fixed_point_pois):
    """Fit at the fixed point (all POIs at their SM values) from the snapshot of a scan, for :func:`pvalue`."""
    return [
        "combine", *MULTIDIMFIT, scan_file, "--algo", "fixed", *DISCRETE_PROFILING, "--cminDefaultMinimizerStrategy=0",
        "--cminFallbackAlgo", "Minuit2,Simplex,0:0.1", "--cminFallbackAlgo", "Minuit2,Combined,0:0.1", *FREEZE_MH,
        "--fixedPointPOIs", f"{fixed_point_pois},MH={HIGGS_MASS}", "-n", ".pvalue", *MASS, "--saveWorkspace",
    ]  # fmt: skip


def pvalue(fixed_point_file, n_pois):
    """Compatibility with the SM: chi2 probability of 2 deltaNLL of the fixed point with one dof per POI."""
    import uproot
    from scipy.stats import chi2

    if not os.path.isfile(fixed_point_file):
        raise FileNotFoundError(f"{fixed_point_file} does not exist, the fit at the SM point did not run or failed")
    delta_nll = uproot.open(fixed_point_file)["limit"].arrays()["deltaNLL"]
    # entry 0 is the best fit, entry 1 the fixed (SM) point
    if len(delta_nll) < 2:
        raise RuntimeError(f"{fixed_point_file} has no fixed point entry, the fit at the SM point failed")
    return 1 - chi2(n_pois).cdf(2 * delta_nll[1])


def round_significant(value, digits=2):
    return round(value, digits - int(f"{value:.1e}".split("e")[1]) - 1)
