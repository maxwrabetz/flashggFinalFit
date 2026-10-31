"""law tasks of the Combine step: workspace, Asimov and observed fits and scans, impacts, Hesse, toys and p-value.

The combine commands are built by finalfits.combine and run as external commands; the tasks resolve the paths of
the analysis config, run the commands in Combine/runFits_<variable>/<step>/ of the output directory and, for
batch_flavor slurm/psi (PSI Tier-3), run in the job scratch and copy from / to the storage element.
"""

import copy
import errno
import functools
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings

import law
import yaml

from finalfits import combine
from finalfits.combine import output_file, parameter_range_for_poi, scan_grid_configuration

from Background.law_background import Background
from commonObjects import BMW, combineVariableDict, lumiMap
from commonTools import extractListOfCatsFromHiggsDNAAllData
from Datacard.law_datacard import MakeDatacard
from framework import HTCondorWorkflow, SlurmWorkflow, Task
from pdfindex_utils import extract_pdf_indices, update_override_file

# Suppress // UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero // warning.
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

# Ensure SLURM_JOB_ID is set for local workflows that still use /scratch.
if "SLURM_JOB_ID" not in os.environ:
    os.environ["SLURM_JOB_ID"] = f"local_{os.getpid()}"

PSI_XROOTD = "root://t3dcachedb03.psi.ch:1094/"


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def convert_boolean_string(string):
    return string in (True, "True", "true")


def get_lumi_label(year):
    if year in lumiMap:
        lumi = lumiMap[year]
    else:
        lumi = sum(lumiMap[y] for y in re.split(r"[,_]", str(year)) if y in lumiMap)
    return f"{lumi:.1f} fb^{{-1}} (13.6 TeV)"


def execute_command(command, return_output=False, shell=False):
    """Run a (shell) command for the file handling; failures are only printed."""
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, shell=shell, env=os.environ)
        print("Script output:", result.stdout)
        print("Script executed successfully.")
        if return_output:
            return (result.stdout).split("\n")[0]
    except subprocess.CalledProcessError as e:
        print("Error executing script:", e.stderr)


def run_command(command, check=True):
    """Run combine (or a plot script), printing its output; a failure raises unless ``check=False``."""
    print("Running:", shlex.join(command))
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Script output:", result.stdout)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error executing script:", e.stderr)
        if check:
            raise


def _t3_transfer(src, dst, action):
    # files of a directory on the PSI storage element (via xrootd) or on a local disk
    file_list = subprocess.check_output(["xrdfs", "root://t3dcachedb03.psi.ch", "ls", src]).decode().splitlines()
    print(file_list)
    for file_path in file_list:
        filename = file_path.split("/")[-1]
        if "bkgfTest-Data" in filename:
            continue
        if "/pnfs" in file_path:
            src_file = f"{PSI_XROOTD}{file_path}"
            dest_file = f"{PSI_XROOTD}{dst}/{filename}"
            if action == "copy":
                execute_command([f"xrdcp -rf {src_file} {dest_file}"], shell=True)
            else:
                execute_command([f"xrdfs {PSI_XROOTD} mv -f {src_file} {dest_file}"], shell=True)
        else:
            command = "cp -rf" if action == "copy" else "mv -f"
            execute_command([f"{command} {file_path} {dst}/{filename}"], shell=True)


def manually_copy_t3(src, dst):
    _t3_transfer(src, dst, "copy")


def manually_move_t3(src, dst):
    _t3_transfer(src, dst, "move")


@functools.cache
def _parse_config(path, mtime):
    with open(path) as file:
        return yaml.safe_load(file)


def load_analysis_config(year, variable):
    # config/<year>_<variable>.yml, or config/<year>_inclusive.yml for the inclusive analysis
    name = f"{year}_{variable}.yml" if variable else f"{year}_inclusive.yml"
    path = os.path.join(os.environ["ANALYSIS_PATH"], "config", name)
    # law evaluates the outputs of every branch, parse the large config once (copies, they are modified)
    return copy.deepcopy(_parse_config(path, os.path.getmtime(path)))


@functools.cache
def workspace_pdf_indices(workspace):
    """pdfindex categories of a workspace (merged categories do not have the usual cat0/1/2 ones)."""
    return extract_pdf_indices(workspace)


def batch_options(step_config):
    """Workflow and batch parameters of a step of the analysis config (combine_fit, combine_impacts, ...)."""
    return dict(
        workflow=step_config["execution"],
        slurm_partition=step_config["batchPartition"],
        slurm_memory=step_config["batchMemory"],
        slurm_max_runtime=step_config["batchMaxRuntime"],
        htcondor_partition=step_config["batchPartition"],
        htcondor_memory=step_config["batchMemory"],
        htcondor_max_runtime=step_config["batchMaxRuntime"],
    )


def targets(paths):
    return [law.LocalFileTarget(path) for path in paths]


class ScanRootFileTarget(law.LocalFileTarget):
    """A scan-point target that rejects truncated ROOT files from killed jobs."""

    def exists(self):
        return super().exists() and os.path.getsize(self.path) >= 1024


class _CombineTask:
    """Paths of the analysis and PSI Tier-3 staging shared by the tasks (a mixin, not a task itself)."""

    @property
    def analysis_config(self):
        return load_analysis_config(self.year, self.variable)

    @property
    def inclusive(self):
        return self.variable == ""

    @property
    def out_dir(self):
        return self.output_dir if self.output_dir != "" else self.analysis_config["outputFolder"]

    @property
    def version_name(self):
        return self.variable if self.variable != "" else "inclusive"

    @property
    def fit_folder(self):
        return "runFits_mu_fiducial" if self.inclusive else f"runFits_{self.variable}"

    def fit_path(self, *parts, base=None):
        """<output (or job scratch) dir>/Combine/runFits_<variable>/<parts>"""
        return os.path.join(base or self.out_dir, "Combine", self.fit_folder, *parts)

    @property
    def datacard_name(self):
        return f"Datacard_{self.year}" if self.inclusive else f"Datacard_{self.variable}_{self.year}"

    @property
    def workspace_path(self):
        return os.path.join(self.out_dir, "Combine", f"{self.datacard_name}.root")

    @property
    def variable_dict(self):
        return combineVariableDict(self.variable, self.year)

    @property
    def pois(self):
        return ["r"] if self.inclusive else self.variable_dict["paramStrNoOne"]

    @property
    def poi_defaults(self):
        """All POIs at 1, for --setParameters."""
        return ",".join(self.variable_dict["paramStr"])

    def common_params(self):
        return dict(output_dir=self.out_dir, variable=self.variable, year=self.year, batch_flavor=self.batch_flavor)

    def fitted_pdf_indices(self, poi):
        """pdfindex values of the Asimov first step of a POI (Combine/checkPdfIdx.C)."""
        first_step = self.fit_path("asimov", output_file(f"firstStep_{poi}"))
        return combine.fitted_pdf_indices(first_step, f"{os.environ['ANALYSIS_PATH']}/Combine/checkPdfIdx.C")

    def first_step_pdf_indices(self):
        """pdfindex values of the Asimov first step of the first POI, None if it does not exist."""
        poi = self.pois[0]
        first_step = self.fit_path("asimov", output_file(f"firstStep_{poi}"))
        if os.path.exists(first_step):
            return self.fitted_pdf_indices(poi)
        print(f"First-step file not found for PDF index extraction: {first_step}")
        return None

    # -------------------------------------------------------------------------------- PSI Tier-3 (slurm/psi)
    @property
    def is_psi(self):
        return self.batch_flavor == "slurm/psi"

    @staticmethod
    def set_scratch():
        # Have to use /scratch/<user>/<job id> for slurm/psi
        os.environ["TARGET_PATH"] = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}"
        return os.environ["TARGET_PATH"]

    def make_fit_dir(self, sub):
        """Create Combine/runFits_<variable>/<sub> and return the directory the job writes to.

        slurm/psi: on the storage element (xrootd, unless the output is on /work) and in the job scratch.
        """
        out = self.out_dir
        if not self.is_psi:
            execute_command([f"mkdir -p {out}/Combine/{self.fit_folder}/{sub}"], shell=True)
            return out
        if "/work" in out:
            execute_command([f"mkdir -p {out}/Combine/{self.fit_folder}/{sub}"], shell=True)
        else:
            execute_command([f"xrdfs {PSI_XROOTD} mkdir -p {out}/Combine/{self.fit_folder}/{sub}"], shell=True)
        scratch = self.set_scratch()
        execute_command([f"mkdir -p $TARGET_PATH/Combine/{self.fit_folder}/{sub}"], shell=True)
        return scratch

    def stage_in(self, source, destination=""):
        """slurm/psi: copy runFits_<variable>/<source> of the output to the job scratch."""
        if not self.is_psi:
            return
        out = self.out_dir
        src = f"{out}/Combine/{self.fit_folder}/{source}"
        dst = f"{os.environ['TARGET_PATH']}/Combine/{self.fit_folder}{destination}"
        execute_command(["cp", "-rf", src, dst] if "/work" in out else ["xrdcp", "-rf", f"{PSI_XROOTD}/{src}", dst])

    def stage_out(self):
        """slurm/psi: copy the outputs of the job scratch to the output (don't forget to VOMS!) and clean up."""
        if not self.is_psi:
            return
        out = self.out_dir
        source = f"{os.environ['TARGET_PATH']}/Combine/"
        if "/work" in out:
            execute_command(["cp", "-rf", source, out])
        else:
            execute_command(["xrdcp", "-rf", source, f"{PSI_XROOTD}/{out}"])
        shutil.rmtree(os.environ["TARGET_PATH"])

    def local_copy(self, path, description):
        """Copy a required input into the working directory, return the path to use."""
        if not os.path.exists(path):
            raise RuntimeError(f"Required {description} not found: {path}")
        local = os.path.join(os.getcwd(), os.path.basename(path))
        if os.path.abspath(local) != os.path.abspath(path):
            shutil.copy2(path, local)
        return local if os.path.exists(local) else path

    def fit_settings(self):
        """Prefit tolerance and r range of the fits to data (combine_fit), with their defaults."""
        fit_config = self.analysis_config.get("combine_fit", {})
        r_range = (fit_config.get("rMin", 0.7), fit_config.get("rMax", 1.6))
        return fit_config.get("cminApproxPreFitTolerance", 0.01), r_range


def _exit_inclusive(task):
    print(f"Running {task} for inclusive does not make sense. Please specify a variable.")
    sys.exit(1)


# ======================================================================================== workspace


class PrepareTheDirectory(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Combine/ with the datacard and the signal and background models it points to."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        config = self.analysis_config
        params = dict(self.common_params(), version=self.version_name)
        reqs["MakeDatacard"] = MakeDatacard(**params, **batch_options(config["datacard_yields"]))
        reqs["Background"] = Background(**params, **batch_options(config["backgroundScriptCfg"]))
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def model_folders(self, config):
        return (config["datacard_yields"]["sigModelWSDir"].split("/")[-2],
                config["datacard_yields"]["bkgModelWSDir"].split("/")[-2])  # fmt: skip

    def output(self):
        config = self.analysis_config
        combine_dir = os.path.join(self.out_dir, "Combine")
        input_path = config["inputFiles"]["Trees2WSData"]
        cats = config["backgroundScriptCfg"]["cats"]
        if cats == "auto":
            cats = extractListOfCatsFromHiggsDNAAllData(input_path)
        ext = config[f"packaged_{self.year}"]["ext"]
        signal, background = self.model_folders(config)

        paths = [combine_dir, os.path.join(combine_dir, self.fit_folder)]
        if signal == background:
            model = os.path.join(combine_dir, signal)
            paths += [model, os.path.join(model, "background"), os.path.join(model, "signal")]
        else:
            paths += [os.path.join(combine_dir, signal), os.path.join(combine_dir, signal, "signal")]
            paths += [os.path.join(combine_dir, background), os.path.join(combine_dir, background, "background")]
        for cat in cats.split(","):
            paths.append(os.path.join(combine_dir, background, "background", f"CMS-HGG_multipdf_{cat}.root"))
            paths.append(os.path.join(combine_dir, signal, "signal", f"CMS-HGG_sigfit_packaged{ext}_{cat}.root"))
        paths.append(os.path.join(combine_dir, f"{self.datacard_name}.txt"))
        return targets(paths)

    def run(self):
        config = self.analysis_config
        out = self.out_dir
        combine_dir = os.path.join(out, "Combine")

        def mkdir(path):
            if self.is_psi:
                execute_command([f"xrdfs {PSI_XROOTD} mkdir -p {path}"], shell=True)
            else:
                safe_mkdir(path)

        if self.is_psi:
            mkdir(os.path.join(combine_dir, self.fit_folder))
        else:
            safe_mkdir(combine_dir)
            safe_mkdir(os.path.join(combine_dir, self.fit_folder))
        signal, background = self.model_folders(config)
        for folder in [signal] if signal == background else [signal, background]:
            mkdir(os.path.join(combine_dir, folder))
        signal_dst = os.path.join(combine_dir, signal, "signal")
        background_dst = os.path.join(combine_dir, background, "background")
        mkdir(signal_dst)
        mkdir(background_dst)

        # Copying the models
        background_src = os.path.join(out, "Background", f"outdir_{config['backgroundScriptCfg']['ext']}")
        signal_src = os.path.join(out, f"outdir_packaged{config[f'packaged_{self.year}']['ext']}/")
        if self.is_psi:
            manually_copy_t3(background_src, background_dst)
            manually_copy_t3(signal_src, signal_dst)
        else:
            shutil.copytree(background_src, background_dst, dirs_exist_ok=True)
            shutil.copytree(signal_src, signal_dst, dirs_exist_ok=True)

        # Signal models named *_<year>.root (relative to the working directory, as it always was)
        for file_path in glob.glob(f"{signal}/signal/*_{self.year}.root"):
            if os.path.isfile(file_path):
                new_name = file_path.replace(f"_{self.year}.root", ".root")
                os.rename(file_path, new_name)
                print(f"Renamed {file_path} to {new_name}")

        # The cleaned datacard if it exists, else the datacard
        datacard = os.path.join(out, "Datacards", f"{self.datacard_name}.txt")
        cleaned = os.path.join(out, "Datacards", f"{self.datacard_name}_cleaned.txt")
        source = cleaned if os.path.exists(cleaned) else datacard
        destination = os.path.join(combine_dir, f"{self.datacard_name}.txt")
        if self.is_psi:
            execute_command([f"xrdcp -rf {PSI_XROOTD}{source} {PSI_XROOTD}{destination}"], shell=True)
        else:
            shutil.copy2(source, destination)
        print("Combine directory sucessfully prepared.")


class RunText2Workspace(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Datacard -> workspace (scripts/combine.py text2workspace) with the physics model of the variable."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        reqs["PrepareTheDirectory"] = PrepareTheDirectory(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        return targets([self.workspace_path])

    def run(self):
        config = self.analysis_config
        out = self.out_dir
        name = self.datacard_name
        models = [config["datacard_yields"]["sigModelWSDir"], config["datacard_yields"]["bkgModelWSDir"]]
        if self.is_psi:
            work_dir = self.set_scratch()
            for folder in models:
                execute_command([f"mkdir -p $TARGET_PATH/Combine/{folder}"], shell=True)
            execute_command(["mkdir -p $TARGET_PATH/Combine/t2w_jobs"], shell=True)
            # the datacard and the models it points to
            execute_command(["xrdcp", "-rf", f"{PSI_XROOTD}/{out}/Combine/{name}.txt", f"{work_dir}/Combine"])
            for folder in models:
                destination = f"{work_dir}/Combine/{folder.split('/')[-2]}"
                execute_command(["xrdcp", "-rf", f"{PSI_XROOTD}/{out}/Combine/{folder}", destination])
        else:
            work_dir = out
        datacards_dir = os.path.join(work_dir, "Combine")

        script = os.path.join(os.environ["ANALYSIS_PATH"], "scripts/combine.py")
        model = "mu_fiducial" if self.inclusive else self.variable
        datacard, workspace = (os.path.join(datacards_dir, f"{name}.{ext}") for ext in ("txt", "root"))
        run_command(["python3", script, "text2workspace", datacard, workspace, "--model", model], check=False)

        if self.is_psi:
            execute_command([f"xrdcp -rf {datacards_dir}/{name}.root {PSI_XROOTD}/{out}/Combine/"], shell=True)
            shutil.rmtree(work_dir)

        # Persist the pdfindex values of the workspace so downstream cat-merged fits can reuse them
        pdf_indices = extract_pdf_indices(os.path.join(out, "Combine", f"{name}.root"))
        if pdf_indices:
            override_path = os.path.join(os.environ["ANALYSIS_PATH"], "config", "pdfindex_overrides.json")
            update_override_file(override_path, self.year, self.version_name, pdf_indices)


# ======================================================================================== Asimov fits


class AsimovFitCategoryFirstStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Asimov best fit per POI, saving the workspace with the fitted pdf indices that seeds the scans."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    cats = law.Parameter(description="Current category")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return dict(enumerate(["r"] if self.inclusive else self.cats.split(",")))

    def output(self):
        branch = self.branch_data
        return targets([
            self.fit_path("asimov"),
            self.fit_path("asimov", output_file(f"firstStep_{branch}")),
            self.fit_path("asimov", f"multidimfitfirstStep_{branch}.root"),
        ])  # fmt: skip

    def run(self):
        branch = self.branch_data
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("asimov")
        os.chdir(self.fit_path("asimov", base=work_dir))

        if self.inclusive:
            # all BMW categories of all years (also for combined years like 2022_2023)
            save_index = ",".join(f"pdfindex_{bmw}_{year}_13TeV" for bmw in BMW for year in self.year.split("_"))
            command = combine.asimov_first_step_command(workspace, branch, save_index)
        else:
            save_index = ",".join(workspace_pdf_indices(workspace))
            command = combine.asimov_first_step_command(
                workspace, branch, save_index, poi=branch, set_parameters=self.poi_defaults
            )
        run_command(command)

        self.stage_out()
        os.chdir(cwd)


class CreateAsimovFitFirstStep(_CombineTask, law.Task):
    """All Asimov first steps (one AsimovFitCategoryFirstStep workflow)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def requires(self):
        if self.inclusive:
            cats, version = ["r"], "inclusive_v1"  # cats is not used by the inclusive first step
        else:
            cats, version = ",".join(self.pois), f"{self.variable}_v1"
        impact_config = self.analysis_config["combine_impacts"]
        return [
            AsimovFitCategoryFirstStep(
                **self.common_params(), cats=cats, version=version, **batch_options(impact_config)
            )
        ]

    def output(self):
        return targets([self.fit_path("asimov")])

    def run(self):
        return True


class _AsimovScan(_CombineTask):
    """Asimov likelihood scans, one branch per scan point (per POI and point with ``cats``, so that all bins of a
    differential measurement run in one workflow). A larger ``grid_points`` with ``first_point``/``last_point``
    evaluates a subset of a finer grid, ``range_profile`` selects per-POI ranges of
    combine_fit.asimov_parameterRangeProfiles."""

    stat_only = False

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        reqs["CreateAsimovFitFirstStep"] = CreateAsimovFitFirstStep(**self.common_params())
        return reqs

    def create_branch_map(self):
        _, points = scan_grid_configuration(self.nPoints, self.grid_points, self.first_point, self.last_point)
        if not self.cats:
            return dict(enumerate(points))
        cats = [cat.strip() for cat in self.cats.split(",") if cat.strip()]
        return dict(enumerate((cat, point) for cat in cats for point in points))

    def _current_branch_info(self):
        # (POI, point), the POI is the cat parameter for workflows of a single POI
        data = self.branch_data
        if isinstance(data, tuple):
            return data
        if not self.cat:
            raise ValueError("Category not set for branch without tuple data")
        return (self.cat, data)

    def scan_name(self, cat, point):
        return f"AsimovPostFitScan{'Stat' if self.stat_only else 'Fit'}_{cat}.POINTS.{point}.{point}"

    def output(self):
        cat, point = self._current_branch_info()
        return [
            law.LocalFileTarget(self.fit_path("asimov")),
            ScanRootFileTarget(self.fit_path("asimov", output_file(self.scan_name(cat, point)))),
        ]

    def run(self):
        cat, point = self._current_branch_info()
        total_points, _ = scan_grid_configuration(self.nPoints, self.grid_points, self.first_point, self.last_point)
        fit_config = self.analysis_config["combine_fit"]
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("asimov")
        os.chdir(self.fit_path("asimov", base=work_dir))

        pdf_index = self.fitted_pdf_indices(cat)
        first_step = self.fit_path("asimov", output_file(f"firstStep_{cat}"))
        if self.inclusive:
            set_parameters = f"{pdf_index}" if convert_boolean_string(self.set_pdfidx_inclusives) else "r=1"
            command = combine.asimov_scan_command(
                first_step, cat, point, total_points, f"{fit_config['setParameterRange']}", set_parameters,
                stat_only=self.stat_only, inclusive=True,
            )  # fmt: skip
            run_command(command, check=self.stat_only)
        else:
            set_parameters = f"{self.poi_defaults},{pdf_index}" if pdf_index else self.poi_defaults
            profiles = fit_config.get("asimov_parameterRangeProfiles", {})
            ranges = parameter_range_for_poi(
                fit_config["setParameterRange"], cat, profiles.get(str(self.range_profile), {})
            )
            command = combine.asimov_scan_command(
                first_step, cat, point, total_points, ranges, set_parameters, stat_only=self.stat_only,
                save_index=",".join(workspace_pdf_indices(workspace)),
            )  # fmt: skip
            run_command(command)

        self.stage_out()
        os.chdir(cwd)


class AsimovFitCategorySyst(_AsimovScan, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Asimov likelihood scan with all uncertainties."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    cat = law.Parameter(default="", description="Current category")
    cats = law.Parameter(default="", description="Comma separated list of categories to process")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    grid_points = law.Parameter(default=0, description="Total number of points defining the full scan grid")
    first_point = law.Parameter(default=-1, description="First full-grid point index to run")
    last_point = law.Parameter(default=-1, description="Last full-grid point index to run")
    range_profile = law.Parameter(
        default="", description="Optional named Asimov parameter-range profile from the config"
    )
    set_pdfidx_inclusives = law.Parameter(default=False, description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")


class AsimovFitCategoryStat(_AsimovScan, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Stat-only Asimov likelihood scan (all constrained nuisances frozen at the first-step values)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    cat = law.Parameter(default="", description="Current category")
    cats = law.Parameter(default="", description="Comma separated list of categories to process")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")
    grid_points = law.Parameter(default=0, description="Total number of points defining the full scan grid")
    first_point = law.Parameter(default=-1, description="First full-grid point index to run")
    last_point = law.Parameter(default=-1, description="Last full-grid point index to run")
    range_profile = law.Parameter(
        default="", description="Optional named Asimov parameter-range profile from the config"
    )
    set_pdfidx_inclusives = law.Parameter(default=False, description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    stat_only = True


class CreateAsimovFit(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Merge the Asimov scan points and plot the scans (plot1DScan.py)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    set_pdfidx_inclusives = law.Parameter(default=False, description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        params = dict(
            self.common_params(),
            nPoints=fit_config["asimov_numPoints"],
            set_pdfidx_inclusives=self.set_pdfidx_inclusives,
            **batch_options(fit_config),
        )
        if self.inclusive:
            reqs["AsimovFitCategorySyst"] = AsimovFitCategorySyst(**params, cat="r", version="inclusive_v1")
            reqs["AsimovFitCategoryStat"] = AsimovFitCategoryStat(**params, cat="r", version="inclusive_v1")
        else:
            # all POIs in one workflow (single submission)
            cats, version = ",".join(self.pois), f"{self.variable}_v1"
            reqs["AsimovFitCategorySyst_all"] = AsimovFitCategorySyst(**params, cats=cats, version=version)
            reqs["AsimovFitCategoryStat_all"] = AsimovFitCategoryStat(**params, cats=cats, version=version)
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        paths = [self.fit_path("asimov", "scans")]
        for cat in self.pois:
            paths += [
                self.fit_path("asimov", f"higgsCombineAsimovPostFitScan{kind}_{cat}.root") for kind in ("Fit", "Stat")
            ]
            paths += [self.fit_path("asimov", "scans", f"scan_{cat}.{ext}") for ext in ("root", "pdf", "png")]
        return targets(paths)

    def run(self):
        n_points = self.analysis_config["combine_fit"]["asimov_numPoints"]
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("asimov/scans")
        os.chdir(self.fit_path("asimov", base=work_dir))
        self.stage_in("asimov")

        plot_script = ("python3", f"{os.environ['CMSSW_BASE']}/bin/{os.environ['SCRAM_ARCH']}/plot1DScan.py")
        for cat in self.pois:
            scans = {}
            for kind in ("Fit", "Stat"):
                scans[kind] = self.fit_path("asimov", f"higgsCombineAsimovPostFitScan{kind}_{cat}.root", base=work_dir)
                points = [
                    self.fit_path("asimov", output_file(f"AsimovPostFitScan{kind}_{cat}.POINTS.{i}.{i}"), base=work_dir)
                    for i in range(n_points)
                ]
                run_command(combine.hadd_command(scans[kind], points), check=kind == "Fit")

            os.chdir(self.fit_path("asimov", "scans", base=work_dir))
            command = combine.scan_plot_command(
                scans["Fit"], scans["Stat"], cat, f"scan_{cat}", "Expected",
                os.path.join(os.environ["ANALYSIS_PATH"], "Combine", "pois.json"), launcher=plot_script,
            )  # fmt: skip
            run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


# ======================================================================================== Asimov impacts


class AsimovImpactFirstStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts on the Asimov data set, first step: initial fit."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(impact_config)
        )
        reqs["CreateAsimovFitFirstStep"] = CreateAsimovFitFirstStep(**self.common_params())
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        return targets([self.fit_path("impact"), self.fit_path("impact", output_file("_initialFit_Test"))])

    def run(self):
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact")
        os.chdir(self.fit_path("impact", base=work_dir))

        pdf_index = self.first_step_pdf_indices()
        base = "r=1" if self.inclusive else self.poi_defaults
        set_parameters = f"{base},{pdf_index}" if pdf_index else base
        pois = None if self.inclusive else self.pois
        run_command(combine.asimov_impacts_initial_fit_command(self.workspace_path, set_parameters, pois), check=False)

        self.stage_out()
        os.chdir(cwd)


class AsimovImpactSecondStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts on the Asimov data set, second step: one branch per nuisance of the ModelConfig.

    Only constrained nuisances: the free background envelope parameters (env_pdf_*, combineTool.py --allPars)
    often have no +-1 sigma crossing in multi-POI fits ("Closed range without finding crossing!").
    """

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["AsimovImpactFirstStep"] = AsimovImpactFirstStep(
            **self.common_params(), version=self.version_name, **batch_options(impact_config)
        )
        return reqs

    def create_branch_map(self):
        config = self.analysis_config
        exclude = config["combine_impacts"].get("exclude", "") if "combine_impacts" in config else ""
        return dict(enumerate(combine.modelconfig_nuisances(self.workspace_path, self.pois, exclude=exclude)))

    def output(self):
        parameter = self.branch_data
        return targets([self.fit_path("impact"), self.fit_path("impact", output_file(f"_paramFit_Test_{parameter}"))])

    def run(self):
        parameter = self.branch_data
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact")
        os.chdir(self.fit_path("impact", base=work_dir))

        pdf_index = self.first_step_pdf_indices()
        base = "r=1" if self.inclusive else self.poi_defaults
        set_parameters = f"{base},{pdf_index}" if pdf_index else base
        command = combine.asimov_impacts_parameter_fit_command(
            self.workspace_path, parameter, self.pois, set_parameters
        )
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class AsimovImpactThirdStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts on the Asimov data set, third step: collect the fits and plot the impacts."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["AsimovImpactSecondStep"] = AsimovImpactSecondStep(
            **self.common_params(), version=self.version_name, **batch_options(impact_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        impacts = self.fit_path("impact", "impacts")
        if self.inclusive:
            paths = [impacts, os.path.join(impacts, "impacts.pdf")]
            paths.append(os.path.join(impacts, "impacts_corrected_dropBkgModelParams.json"))
        else:
            paths = [path for cat in self.pois for path in (impacts, os.path.join(impacts, f"impacts_{cat}.pdf"))]
        return targets([*paths, os.path.join(impacts, "impacts.json")])

    def run(self):
        config = self.analysis_config
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact/impacts")
        self.stage_in("impact")
        os.chdir(self.fit_path("impact", base=work_dir))

        pdf_index = self.first_step_pdf_indices()
        base = "r=1" if self.inclusive else self.poi_defaults
        set_parameters = f"{base},{pdf_index}" if pdf_index else None
        # the nuisances of the ModelConfig (not all free parameters, the default of combineTool.py)
        exclude = config.get("combine_impacts", {}).get("exclude", "")
        named = combine.modelconfig_nuisances(workspace, self.pois, exclude=exclude)
        if not named:
            raise RuntimeError(
                f"No nuisance parameters found for impacts in {workspace} (variable={self.variable}, "
                f"year={self.year}). Check workspace ModelConfig nuisances and combine_impacts.exclude='{exclude}'."
            )

        if self.inclusive:
            run_command(combine.impacts_collect_command(workspace, named=named, set_parameters=set_parameters))
            impacts_json = self.fit_path("impact", "impacts", "impacts.json", base=work_dir)
            correct_script = os.path.join(os.environ["ANALYSIS_PATH"], "Plots", "correctImpacts.py")
            run_command(combine.correct_impacts_command(correct_script, impacts_json))
            corrected = "impacts/impacts_corrected_dropBkgModelParams.json"
            run_command(combine.plot_impacts_command(corrected, "impacts/impacts"))
        else:
            command = combine.impacts_collect_command(
                workspace, named=named, exclude=config["combine_impacts"]["exclude"], set_parameters=set_parameters
            )
            run_command(command, check=False)
            for cat in self.pois:
                run_command(combine.plot_impacts_command("impacts/impacts.json", f"impacts/impacts_{cat}", poi=cat))

        self.stage_out()
        os.chdir(cwd)


# ======================================================================================== Asimov Hesse


class AsimovCovCorrHesse(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Asimov best fit with robust Hesse (differential only)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        if self.inclusive:
            _exit_inclusive("AsimovCovCorrHesse")
        hesse_config = self.analysis_config["combine_hesse"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.variable, **batch_options(hesse_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        if self.inclusive:
            _exit_inclusive("AsimovCovCorrHesse")
        hesse = self.fit_path("hesse")
        files = ["robustHessefirstStep.root", "multidimfitfirstStep.root", output_file("firstStep")]
        return targets([hesse, *(os.path.join(hesse, name) for name in files)])

    def run(self):
        if self.inclusive:
            _exit_inclusive("AsimovCovCorrHesse")
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("hesse")
        os.chdir(self.fit_path("hesse", base=work_dir))

        save_index = ",".join(workspace_pdf_indices(workspace) or self.variable_dict["pdfIndeces"])
        command = combine.hesse_command(workspace, "firstStep", save_index, asimov_parameters=self.poi_defaults)
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


def _corr_matrix_commands(task, hesse_file, output, observed):
    """Correlation and covariance matrix of Plots/makeCorrMatrix.py."""
    plots = os.path.join(os.environ["ANALYSIS_PATH"], "Plots")
    return [
        combine.corr_matrix_command(
            os.path.join(plots, "makeCorrMatrix.py"),
            os.path.join(plots, "inputs_robustHesse.json"),
            task.variable,
            hesse_file,
            output,
            os.path.join(plots, "poi_differential_hesse_noLabels.json"),
            task.year,
            covariance=covariance,
            observed=observed,
            no_preliminary=convert_boolean_string(task.noPreliminary),
        )  # fmt: skip
        for covariance in (False, True)
    ]


class AsimovCovCorr(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Correlation and covariance matrices of the Asimov fit (Plots/makeCorrMatrix.py)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        config = self.analysis_config
        if self.inclusive:
            _exit_inclusive("AsimovCovCorr")
        reqs["AsimovCovCorrHesse"] = AsimovCovCorrHesse(
            **self.common_params(), version=self.variable, **batch_options(config["combine_hesse"])
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        if self.inclusive:
            _exit_inclusive("AsimovCovCorr")
        plots = self.fit_path("hesse", "Plots")
        names = [f"{m}Matrix_{self.variable}_syst.{ext}" for m in ("corr", "cov") for ext in ("pdf", "png")]
        return targets([plots, *(os.path.join(plots, name) for name in names)])

    def run(self):
        if self.inclusive:
            return True
        work_dir = self.make_fit_dir("hesse/Plots")
        self.stage_in("hesse", "/")

        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], "Plots"))
        hesse_file = self.fit_path("hesse", "robustHessefirstStep.root")
        for command in _corr_matrix_commands(self, hesse_file, self.fit_path("hesse", "Plots", base=work_dir), False):
            run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


# ======================================================================================== fits to data


class UnblindedFitSystSingle(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Best fit to data per POI, saving the workspace (snapshot) that seeds the scans."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return dict(enumerate(self.pois))

    def output(self):
        cat = self.branch_data
        return targets([self.fit_path("dataFit"), self.fit_path("dataFit", output_file(f"DataPostFitBestFit_{cat}"))])

    def run(self):
        cat = self.branch_data
        tolerance, r_range = self.fit_settings()
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("dataFit")
        os.chdir(self.fit_path("dataFit", base=work_dir))

        if self.inclusive:
            command = combine.observed_best_fit_command(workspace, cat, tolerance, inclusive=True, r_range=r_range)
        else:
            save_index = ",".join(self.variable_dict["pdfIndeces"])
            command = combine.observed_best_fit_command(workspace, cat, tolerance, save_index=save_index)
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedFitStatSingle(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Refit of the best fit to data from its snapshot, seeding the stat-only scans."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        params = dict(self.common_params(), version=self.version_name)
        fit_options = batch_options(self.analysis_config["combine_fit"])
        reqs["RunT2WS"] = RunText2Workspace(**params, **fit_options)
        reqs["UnblindedFitSystSingle"] = UnblindedFitSystSingle(**params, **fit_options)
        return reqs

    def create_branch_map(self):
        return dict(enumerate(self.pois))

    def output(self):
        return targets([self.fit_path("dataFit", output_file(f"DataPostFitBestFitStat_{self.branch_data}"))])

    def run(self):
        cat = self.branch_data
        tolerance, r_range = self.fit_settings()
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("dataFit")
        os.chdir(self.fit_path("dataFit", base=work_dir))

        best_fit = self.fit_path("dataFit", output_file(f"DataPostFitBestFit_{cat}"))
        snapshot = self.local_copy(best_fit, "best fit snapshot")
        command = combine.observed_best_fit_command(
            snapshot, cat, tolerance, stat_only=True, inclusive=self.inclusive, r_range=r_range
        )
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class _ObservedScan(_CombineTask):
    """Likelihood scan in data, one branch per POI and scan point."""

    stat_only = False

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        best_fit = UnblindedFitStatSingle if self.stat_only else UnblindedFitSystSingle
        reqs[best_fit.__name__] = best_fit(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return dict(enumerate((cat, point) for cat in self.pois for point in range(int(self.nPoints))))

    def run(self):
        cat, point = self.branch_data
        config = self.analysis_config
        tolerance, r_range = self.fit_settings()
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("dataFit")
        os.chdir(self.fit_path("dataFit", base=work_dir))

        n_points = int(self.nPoints)
        if self.stat_only:
            best_fit = self.fit_path("dataFit", output_file(f"DataPostFitBestFitStat_{cat}"))
            snapshot = self.local_copy(best_fit, "stat-only best fit snapshot")
            command = combine.observed_scan_command(
                snapshot, cat, point, n_points, stat_only=True, inclusive=self.inclusive, r_range=r_range,
                prefit_tolerance=tolerance,
            )  # fmt: skip
        elif self.inclusive:
            best_fit = self.fit_path("dataFit", output_file(f"DataPostFitBestFit_{cat}"))
            command = combine.observed_scan_command(
                best_fit, cat, point, n_points, inclusive=True, r_range=r_range,
                parameter_ranges=f"{config['combine_fit']['setParameterRange']}", prefit_tolerance=tolerance,
            )  # fmt: skip
        else:
            best_fit = self.fit_path("dataFit", output_file(f"DataPostFitBestFit_{cat}"))
            command = combine.observed_scan_command(
                best_fit, cat, point, n_points, save_index=",".join(self.variable_dict["pdfIndeces"]),
                set_parameters=self.poi_defaults,
            )  # fmt: skip
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedFitCategorySyst(_ObservedScan, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Likelihood scan in data with all uncertainties."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def output(self):
        cat, point = self.branch_data
        scan = output_file(f"DataPostFitScanFit_{cat}.POINTS.{point}.{point}")
        return targets([self.fit_path("dataFit"), self.fit_path("dataFit", scan)])


class UnblindedFitCategoryStat(_ObservedScan, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Stat-only likelihood scan in data."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    nPoints = law.Parameter(default=30, description="Number of points for the LL scan")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    stat_only = True

    def output(self):
        cat, point = self.branch_data
        return targets([self.fit_path("dataFit", output_file(f"DataPostFitScanStat_{cat}.POINTS.{point}.{point}"))])


class CreateUnblindedFit(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Merge the scan points in data and plot the observed scans (plot1DScan.py)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        params = dict(
            self.common_params(),
            nPoints=fit_config["unblindedFit_numPoints"],
            version=self.version_name,
            **batch_options(fit_config),
        )
        reqs["UnblindedFitCategorySyst"] = UnblindedFitCategorySyst(**params)
        reqs["UnblindedFitCategoryStat"] = UnblindedFitCategoryStat(**params)
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        scans = self.fit_path("dataFit", "scans")
        return targets(
            [
                scans,
                *(os.path.join(scans, f"scan_{cat}_observed.{e}") for cat in self.pois for e in ("root", "pdf", "png")),
            ]
        )

    def run(self):
        n_points = int(self.analysis_config["combine_fit"]["unblindedFit_numPoints"])
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("dataFit/scans")
        os.chdir(self.fit_path("dataFit", base=work_dir))
        job_dir = self.fit_path("dataFit", base=work_dir)
        self.stage_in("dataFit")

        def scan(kind, cat, suffix=""):
            return os.path.join(job_dir, output_file(f"DataPostFitScan{kind}_{cat}{suffix}"))

        for cat in self.pois:
            for kind in ("Fit", "Stat"):
                points = [scan(kind, cat, f".POINTS.{i}.{i}") for i in range(n_points)]
                try:
                    run_command(combine.hadd_command(scan(kind, cat), points))
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error executing hadd for {kind} scan {cat}: {e.stderr}") from e

        for cat in self.pois:
            for kind in ("Fit", "Stat"):
                if not os.path.exists(scan(kind, cat)) or os.path.getsize(scan(kind, cat)) == 0:
                    raise RuntimeError(f"Required scan output is missing: {scan(kind, cat)}")

        for cat in self.pois:
            command = combine.scan_plot_command(
                scan("Fit", cat), scan("Stat", cat), cat, f"scans/scan_{cat}_observed", "Observed",
                os.path.join(os.environ["ANALYSIS_PATH"], "Combine", "pois.json"),
            )  # fmt: skip
            run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


# ======================================================================================== Hesse and impacts in data


class UnblindedCovCorrHesse(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Best fit to data with robust Hesse (differential only)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        if self.inclusive:
            return []
        hesse = self.fit_path("hesse")
        files = ["robustHessefirstStep_data.root", "multidimfitfirstStep_data.root", output_file("firstStep_data")]
        return targets([hesse, *(os.path.join(hesse, name) for name in files)])

    def run(self):
        if self.inclusive:
            _exit_inclusive("UnblindedCovCorrHesse")
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("hesse")
        os.chdir(self.fit_path("hesse", base=work_dir))

        save_index = ",".join(self.variable_dict["pdfIndeces"])
        run_command(combine.hesse_command(self.workspace_path, "firstStep_data", save_index), check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedCovCorr(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Correlation and covariance matrices of the fit to data (Plots/makeCorrMatrix.py)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    noPreliminary = law.Parameter(default=False, description="Flag, if final plot should bear the Preliminary.")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        hesse_config = self.analysis_config["combine_hesse"]
        reqs["UnblindedCovCorrHesse"] = UnblindedCovCorrHesse(
            **self.common_params(), version="r" if self.inclusive else self.variable, **batch_options(hesse_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        plots = self.fit_path("hesse", "Plots", "data")
        names = [f"{m}Matrix_{self.variable}_syst_obs.{ext}" for m in ("corr", "cov") for ext in ("pdf", "png")]
        return targets([plots, *(os.path.join(plots, name) for name in names)])

    def run(self):
        if self.inclusive:
            _exit_inclusive("UnblindedCovCorrHesse")
        work_dir = self.make_fit_dir("hesse/Plots/data")
        self.stage_in("hesse", "/")

        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ["ANALYSIS_PATH"], "Plots"))
        hesse_file = self.fit_path("hesse", "robustHessefirstStep_data.root", base=work_dir)
        output = self.fit_path("hesse", "Plots", "data", base=work_dir)
        for command in _corr_matrix_commands(self, hesse_file, output, True):
            run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedImpactFirstStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts in data, first step: initial fit."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(impact_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        unblinded = self.fit_path("impact", "unblinded")
        return targets([unblinded, os.path.join(unblinded, output_file("_initialFit_Test"))])

    def run(self):
        impact_config = self.analysis_config.get("combine_impacts", {})
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact/unblinded")
        os.chdir(self.fit_path("impact", "unblinded", base=work_dir))

        if self.inclusive:
            command = combine.observed_impacts_initial_fit_command(
                self.workspace_path,
                prefit_tolerance=impact_config.get("cminApproxPreFitTolerance", 0.01),
                set_parameter=impact_config.get("setParameters", 1.000),
            )
        else:
            command = combine.observed_impacts_initial_fit_command(self.workspace_path, pois=self.pois)
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedImpactSecondStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts in data, second step: one branch per free parameter (as combineTool.py Impacts)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["UnblindedImpactFirstStep"] = UnblindedImpactFirstStep(
            **self.common_params(), version=self.version_name, **batch_options(impact_config)
        )
        return reqs

    def create_branch_map(self):
        return dict(enumerate(combine.free_parameters(self.workspace_path, self.pois)))

    def output(self):
        unblinded = self.fit_path("impact", "unblinded")
        return targets([unblinded, os.path.join(unblinded, output_file(f"_paramFit_Test_{self.branch_data}"))])

    def run(self):
        parameter = self.branch_data
        impact_config = self.analysis_config.get("combine_impacts", {})
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact/unblinded")
        os.chdir(self.fit_path("impact", "unblinded", base=work_dir))

        initial_fit = self.fit_path("impact", "unblinded", output_file("_initialFit_Test"))
        try:
            best_fit = combine.best_fit_values(initial_fit, self.pois)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if self.inclusive:
            options = [
                "--cminApproxPreFitTolerance", f"{impact_config.get('cminApproxPreFitTolerance', 0.01)}",
                "--stepSize", f"{impact_config.get('stepSize', 0.05)}",
                "--setCrossingTolerance", f"{impact_config.get('setCrossingTolerance', 0.00005)}",
                "--robustHesse", "1",
            ]  # fmt: skip
        else:
            options = None
        command = combine.observed_impacts_parameter_fit_command(
            self.workspace_path, parameter, self.pois, best_fit, inclusive_options=options
        )
        run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


class UnblindedImpactThirdStep(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Impacts in data, third step: collect the fits and plot the impacts."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        impact_config = self.analysis_config["combine_impacts"]
        reqs["UnblindedImpactSecondStep"] = UnblindedImpactSecondStep(
            **self.common_params(), version="r" if self.inclusive else self.variable, **batch_options(impact_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        impacts = self.fit_path("impact", "unblinded", "impacts")
        corrected = os.path.join(impacts, "impacts_corrected_dropBkgModelParams.json")
        if self.inclusive:
            paths = [impacts, os.path.join(impacts, "impacts_unblinded.pdf"), corrected]
        else:
            paths = [
                p
                for cat in self.pois
                for p in (impacts, os.path.join(impacts, f"impacts_unblinded_{cat}.pdf"), corrected)
            ]
        return targets([*paths, os.path.join(impacts, "impacts.json")])

    def run(self):
        impact_config = self.analysis_config["combine_impacts"]
        workspace = self.workspace_path
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("impact/unblinded/impacts")
        os.chdir(self.fit_path("impact", "unblinded", base=work_dir))
        self.stage_in("impact/unblinded", "/impact")

        impacts = self.fit_path("impact", "unblinded", "impacts", base=work_dir)
        corrected = os.path.join(impacts, "impacts_corrected_dropBkgModelParams.json")
        correct_script = os.path.join(os.environ["ANALYSIS_PATH"], "Plots", "correctImpacts.py")
        blind = bool(impact_config["not_show_POI"])
        if self.inclusive:
            commands = [
                combine.impacts_collect_command(workspace, freeze_mh=False),
                combine.correct_impacts_command(correct_script, os.path.join(impacts, "impacts.json")),
                combine.plot_impacts_command(corrected, "impacts/impacts_unblinded", blind=blind),
            ]
        else:
            exclude = impact_config["exclude"]
            translate = os.path.join(os.environ["ANALYSIS_PATH"], "Combine", "pois.json")
            commands = [
                combine.impacts_collect_command(workspace, exclude=exclude),
                combine.correct_impacts_command(correct_script, os.path.join(impacts, "impacts.json"), exclude=exclude),
                *(
                    combine.plot_impacts_command(
                        corrected, f"impacts/impacts_unblinded_{cat}", poi=cat, translate=translate, blind=blind
                    )
                    for cat in self.pois
                ),
            ]
        for command in commands:
            run_command(command, check=False)

        self.stage_out()
        os.chdir(cwd)


# ======================================================================================== mass distributions


class MggBestFit(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Best fit to data per POI for the post-fit mass distributions."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        toy_config = self.analysis_config["combine_mggToys"]
        reqs["RunT2WS"] = RunText2Workspace(
            **self.common_params(), version=self.version_name, **batch_options(toy_config)
        )
        return reqs

    def create_branch_map(self):
        return dict(enumerate(self.pois))

    def output(self):
        cat = self.branch_data
        models = self.fit_path("postFit", f"SplusBModels_{cat}")
        return targets(
            [self.fit_path("postFit"), models, os.path.join(models, output_file(f"_bestfit_syst_obs_{cat}"))]
        )

    def run(self):
        cat = self.branch_data
        method = f"{self.analysis_config['combine_mggToys']['loadSnapshot']}"
        cwd = os.getcwd()
        work_dir = self.make_fit_dir(f"postFit/SplusBModels_{cat}")
        os.chdir(self.fit_path("postFit", f"SplusBModels_{cat}", base=work_dir))

        run_command(combine.mgg_best_fit_command(self.workspace_path, cat, method), check=False)

        self.stage_out()
        os.chdir(cwd)


def _move_files(pattern, destination, psi):
    """Move the files matching a pattern (combine adds the seed to the toy file names)."""
    sources = glob.glob(pattern)
    if not sources:
        print("No files found matching the pattern.")
    for source in sources:
        try:
            print(f"Moving {source} to {destination}")
            # On the PSI Tier-3 the toys can be on the storage element which needs special handling
            if psi and ("/pnfs" in source or "/pnfs" in destination):
                manually_move_t3(source, destination)
            else:
                shutil.move(source, destination)
            print("File moved successfully.")
        except Exception as e:
            print(f"Error moving file {source}: {e}")


class MggToyGeneration(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Toys for the uncertainty bands of the (pre- or post-fit) mass distributions: generate, fit and throw the
    Asimov data set of the fit (with the signal of the POI set to 0); one branch per toy and POI."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    is_postfit = law.Parameter(
        default=False, description="Flag that signifies if toys are created for postfit mass distributions."
    )

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    @property
    def postfit(self):
        return convert_boolean_string(self.is_postfit)

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        toy_options = batch_options(self.analysis_config["combine_mggToys"])
        if self.postfit:
            reqs["MggBestFit"] = MggBestFit(
                **self.common_params(), version="r" if self.inclusive else self.variable, **toy_options
            )
        else:
            reqs["RunT2WS"] = RunText2Workspace(**self.common_params(), version=self.version_name, **toy_options)
        return reqs

    def create_branch_map(self):
        n_toys = int(self.analysis_config["combine_mggToys"]["nToys"])
        return dict(enumerate((toy, cat) for toy in range(n_toys) for cat in self.pois))

    def toys_path(self, cat, *parts, base=None):
        return self.fit_path("postFit" if self.postfit else "preFit", f"SplusBModels_{cat}", "toys", *parts, base=base)

    def output(self):
        toy, cat = self.branch_data
        paths = [] if self.postfit else [self.fit_path("preFit"), self.fit_path("preFit", f"SplusBModels_{cat}")]
        toys = self.toys_path(cat)
        paths += [toys, os.path.join(toys, "filechecker"), os.path.join(toys, "filechecker", f"toy_{toy}_ok.txt")]
        return targets([*paths, os.path.join(toys, f"toy_{toy}.root")])

    def run(self):
        toy, cat = self.branch_data
        snapshot = f"{self.analysis_config['combine_mggToys']['loadSnapshot']}"
        fit = "postFit" if self.postfit else "preFit"
        cwd = os.getcwd()
        work_dir = self.make_fit_dir(f"{fit}/SplusBModels_{cat}/toys/filechecker")
        os.chdir(self.toys_path(cat, base=work_dir))

        if self.postfit:
            best_fit = self.fit_path(fit, f"SplusBModels_{cat}", output_file(f"_bestfit_syst_obs_{cat}"))
            signal = f"{cat}={combine.snapshot_value(best_fit, cat, snapshot)}"
            steps = [
                ("gen", combine.toy_generate_command(toy, best_fit, signal, snapshot=snapshot)),
                ("fit", combine.toy_fit_command(toy, cat, snapshot, signal, postfit=True)),
                ("throw", combine.toy_throw_command(toy, snapshot, "r=0" if self.inclusive else f"{cat}=0")),
            ]
        else:
            signal = "r=1" if self.inclusive else self.poi_defaults
            steps = [
                ("gen", combine.toy_generate_command(toy, self.workspace_path, signal)),
                ("fit", combine.toy_fit_command(toy, cat, snapshot, signal)),
                (
                    "throw",
                    combine.toy_throw_command(toy, snapshot, "r=0" if self.inclusive else signal.replace("=1", "=0")),
                ),
            ]

        # On the PSI Tier-3 the toys of an output on the storage element are written to the job scratch
        toy_dir = self.toys_path(
            cat, base=os.environ["TARGET_PATH"] if self.is_psi and "/pnfs" in self.out_dir else None
        )
        renamed = {"gen": f"gen_{toy}.root", "fit": f"fit_{toy}.root", "throw": f"toy_{toy}.root"}
        for step, command in steps:
            run_command(command, check=False)
            pattern = os.path.join(toy_dir, f"higgsCombine_{toy}_{step}_step*.root")
            _move_files(pattern, os.path.join(toy_dir, renamed[step]), self.is_psi)

        for file_path in (os.path.join(toy_dir, renamed["gen"]), os.path.join(toy_dir, renamed["fit"])):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"{file_path} removed successfully.")
                else:
                    print(f"{file_path} does not exist.")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

        # Mark toys with content (> 1000 bytes)
        try:
            if os.path.getsize(os.path.join(toy_dir, renamed["throw"])) > 1000:
                with open(os.path.join(toy_dir, "filechecker", f"toy_{toy}_ok.txt"), "w"):
                    pass
        except OSError:
            print("Error creating file. Probably I/O error.")
            return False

        self.stage_out()
        os.chdir(cwd)


class MggDistribution(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Pre- or post-fit S+B mass distributions per POI (Plots/makeSplusBModelPlot.py)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")
    is_postfit = law.Parameter(
        default=False, description="Flag that signifies if toys are created for postfit mass distributions."
    )

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    @property
    def postfit(self):
        return convert_boolean_string(self.is_postfit)

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        config = self.analysis_config
        mgg_options = batch_options(config["combine_mggToys"])
        label = self.variable if self.variable != "" else "r"
        if config["combine_mggToys"]["doBands"]:
            reqs["MggToyGeneration"] = MggToyGeneration(
                **self.common_params(), is_postfit=self.postfit,
                version=f"{label}_postfit" if self.postfit else f"{label}_prefit", **mgg_options,
            )  # fmt: skip
        elif self.postfit:
            reqs["CreateUnblindedFit"] = CreateUnblindedFit(
                **self.common_params(), version=f"{label}_prefit", **batch_options(config["combine_fit"])
            )
        else:
            reqs["RunT2WS"] = RunText2Workspace(**self.common_params(), version=self.version_name, **mgg_options)
        return reqs

    def create_branch_map(self):
        return dict(enumerate(self.pois))

    def reco_categories(self, cat):
        """Reconstructed categories (with BMW) of a POI; per year (Y<yy>_) for combined years."""
        if self.inclusive:
            cats = ["cat0", "cat1", "cat2"]
        else:
            cats = [c for c in self.variable_dict["catsStrWithBMW"] if "_".join(cat.split("_")[2:]) in c]
        if "_" in self.year:
            cats = [f"Y{year[-2:]}_{c}" for year in self.year.split("_") for c in cats]
        return cats

    def output(self):
        cat = self.branch_data
        fit = "postFit" if self.postfit else "preFit"
        models = self.fit_path(fit, f"SplusBModels_{cat}")
        reco = self.reco_categories(cat)
        paths = [self.fit_path(fit, "jsons"), self.fit_path(fit, "jsons", f"catsWeights_sospb_{cat}_CMS_hgg_mass.json")]
        paths += [os.path.join(models, f"{cat}_{c}_CMS_hgg_mass.{ext}") for ext in ("pdf", "png") for c in reco]
        paths += [
            os.path.join(models, f"{cat}_{c}_CMS_hgg_mass.{ext}") for c in ("all", "wall") for ext in ("pdf", "png")
        ]
        return targets(paths)

    def run(self):
        cat = self.branch_data
        config = self.analysis_config
        fit = "postFit" if self.postfit else "preFit"
        main_dir = os.getcwd()
        work_dir = self.make_fit_dir(f"{fit}/SplusBModels_{cat}/")
        self.stage_in(f"{fit}/SplusBModels_{cat}/toys", f"/{fit}/SplusBModels_{cat}/")
        os.chdir(self.fit_path(fit, base=work_dir))

        plots = os.path.join(os.environ["ANALYSIS_PATH"], "Plots")
        if self.postfit:
            snapshot = f"{config['combine_mggToys']['loadSnapshot']}"
            inputs = [self.fit_path(fit, f"SplusBModels_{cat}", output_file(f"_bestfit_syst_obs_{cat}"))]
            inputs += ["--loadSnapshot", snapshot]
            region = ["--unblind"]
        else:
            inputs = [self.workspace_path]
            region = ["--blindingRegion", "125,125"]
        command = [
            "python3", os.path.join(plots, "makeSplusBModelPlot.py"), "--inputWSFile", *inputs,
            "--cats", ",".join(self.reco_categories(cat)), "--lumiLabel", get_lumi_label(self.year), "--doZeroes",
            *region, "--translateCats", os.path.join(plots, "cats.json"), "--doSumCategories", "--doCatWeights",
            "--saveWeights", "--ext", f"_{cat}", "--POI", f"{cat}",
        ]  # fmt: skip
        if config["combine_mggToys"]["doBands"]:
            command += ["--doBands", "--doToyVeto", "--saveToyYields"]
        run_command(command, check=False)
        if self.postfit:
            os.chdir("..")

        self.stage_out()
        os.chdir(main_dir)


# ======================================================================================== p-value


class PValueCalculation(_CombineTask, Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    """Compatibility of the differential measurement with the SM (fit at the SM point of all POIs)."""

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def workflow_requires(self):
        reqs = dict(super().workflow_requires())
        fit_config = self.analysis_config["combine_fit"]
        reqs["CreateUnblindedFit"] = CreateUnblindedFit(
            **self.common_params(), version=self.version_name, **batch_options(fit_config)
        )
        return reqs

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        if self.inclusive:
            return []
        return targets([self.fit_path("dataFit", output_file(".pvalue")), self.fit_path("pvalue.txt")])

    def run(self):
        cwd = os.getcwd()
        work_dir = self.make_fit_dir("dataFit")
        os.chdir(self.fit_path("dataFit", base=work_dir))

        pvalue_file = self.fit_path("dataFit", output_file(".pvalue"), base=work_dir)
        if not os.path.isfile(pvalue_file):
            print("The pvalue file does not exist in the current directory. Creating it...")
            scan = self.fit_path("dataFit", output_file(f"DataPostFitScanFit_{self.variable_dict['paramStrNoOne'][0]}"))
            run_command(combine.pvalue_fit_command(scan, self.poi_defaults), check=False)
        else:
            print("The pvalue file exists in the current directory. Skipping the creation.")

        n_bins = len(self.variable_dict["paramStrNoOne"])
        os.chdir("../")
        pvalue = combine.round_significant(combine.pvalue(os.path.realpath(pvalue_file), n_bins))
        with open("pvalue.txt", "w") as f:
            f.write(f"{pvalue}")
        print("P-value written to pvalue.txt")
        print(f"The p-value of the variable {self.variable} is: {pvalue}")

        self.stage_out()
        os.chdir(cwd)
