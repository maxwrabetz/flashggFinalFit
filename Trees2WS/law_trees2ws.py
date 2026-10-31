# law tasks for the Trees2WS step of the signal MC; the conversion itself is finalfits.trees2ws
import glob
import os
import shutil
import subprocess

import law
import yaml

from finalfits import trees2ws
from finalfits.utils import setup_logging

from commonObjects import allErasMap, input_masses, production_modes
from framework import HTCondorWorkflow, SlurmWorkflow, Task

PSI_XROOTD = "root://t3dcachedb03.psi.ch:1094/"


def execute_command(command, return_output=False, shell=False):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, shell=shell, env=os.environ)
        print("Script output:", result.stdout)
        print("Script executed successfully.")
        if return_output:
            return (result.stdout).split("\n")[0]
    except subprocess.CalledProcessError as e:
        print("Error executing script:", e.stderr)


def convert_boolean_string(string):
    return string in (True, "True", "true")


def load_analysis_config(year, variable):
    # config/<year>_<variable>.yml, or config/<year>_inclusive.yml for the inclusive analysis
    name = f"{year}_{variable}.yml" if variable else f"{year}_inclusive.yml"
    with open(os.path.join(os.environ["ANALYSIS_PATH"], "config", name)) as f:
        return yaml.safe_load(f)


def era_output_dir(output_dir, year, era, variable):
    era_suffix = "" if era in ["", "None"] else era
    if variable == "":
        return os.path.join(output_dir, f"input_output_{year}{era_suffix}")
    return os.path.join(output_dir, f"input_output_{variable}_{year}{era_suffix}")


class Trees2WSSingleProcess(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    # One branch per signal MC file: production mode x mass point
    input_paths = law.Parameter(description="Directory with the <process>_M-<mass>_<era>/ input directories")
    era = law.Parameter(default="", description="Current era.")
    output_dir = law.Parameter(description="Path to the output directory")
    variable = law.Parameter(default="", description="Differential variable, empty for the inclusive analysis")
    year = law.Parameter(default="2022", description="Year including the era, e.g. 2023preBPix")

    batch_flavor = law.Parameter(default="slurm", description="Batch system to use")

    def trees2ws_config(self):
        return trees2ws.Trees2WSConfig.from_dict(load_analysis_config(self.year[:4], self.variable)["trees2wsCfg"])

    def create_branch_map(self):
        # Input directories are named <process>_M-<mass>_<era>, or <process>_M-<mass>_<year+era>
        for suffix in (self.era, self.year):
            branches = [
                (mode, mass, input_path)
                for mode, process in production_modes
                for mass in input_masses
                for input_path in glob.glob(f"{self.input_paths}/{process}_M-{mass}_{suffix}/*.root")
            ]
            if branches:
                break
        return dict(enumerate(branches))

    def filechecker(self, label):
        mode, mass, _ = self.branch_data
        return os.path.join(self.output_dir, "filechecker", f"{mode}_{mass}_{label}.txt")

    def output(self):
        mode, _, input_path = self.branch_data
        targets = []
        for split in trees2ws.signal_splits(self.trees2ws_config(), mode, self.variable):
            targets.append(law.LocalFileTarget(self.filechecker(split.label)))
            targets.append(law.LocalFileTarget(str(trees2ws.signal_output_path(self.output_dir, input_path, split))))
        return targets

    def run(self):
        setup_logging()
        mode, mass, input_path = self.branch_data

        # On the PSI Tier 3, the jobs write to the local scratch and copy the outputs at the end
        psi = self.batch_flavor == "slurm/psi"
        if psi:
            execute_command([f"xrdfs {PSI_XROOTD} mkdir -p {os.path.join(self.output_dir, 'filechecker')}"], shell=True)
            work_dir = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}/{os.path.basename(self.output_dir)}"
        else:
            work_dir = self.output_dir

        paths = trees2ws.convert_signal_file(
            input_path, work_dir, mode, mass, self.year, self.trees2ws_config(), self.variable
        )

        # The filechecker marks workspaces that are not empty (> 2000 bytes)
        os.makedirs(os.path.join(work_dir, "filechecker"), exist_ok=True)
        for label, path in paths.items():
            if os.path.getsize(path) > 2000:
                open(os.path.join(work_dir, "filechecker", f"{mode}_{mass}_{label}.txt"), "w").close()

        if psi:
            if "/work" in self.output_dir:
                execute_command([f"cp -rf {work_dir}/ {self.output_dir}/"], shell=True)
            else:
                execute_command([f"xrdcp -rf {work_dir}/* {PSI_XROOTD}{self.output_dir}/"], shell=True)
            shutil.rmtree(work_dir)


class Trees2WS(law.Task):
    # Converts the signal MC of all eras of a year and collects the workspaces in ws_signal/
    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used for output folder naming")
    year = law.Parameter(default="2022", description="Year")

    batch_flavor = law.Parameter(default="slurm", description="Batch system to use")

    def era_output_dirs(self):
        config = load_analysis_config(self.year, self.variable)
        output_dir = self.output_dir or config["outputFolder"]
        eras = allErasMap.get(f"{self.year}", [""])
        return config, [(era, era_output_dir(output_dir, self.year, era, self.variable)) for era in eras]

    def requires(self):
        config, era_dirs = self.era_output_dirs()
        batch = config["trees2wsCfg"]
        tasks = []
        for i, (era, era_dir) in enumerate(era_dirs, start=1):
            tasks.append(
                Trees2WSSingleProcess(
                    input_paths=config["inputFiles"]["Trees2WS"],
                    era=era,
                    year=f"{self.year}{era}",
                    output_dir=era_dir,
                    variable=self.variable,
                    version=f"{self.year}_{self.variable}_v{i}" if self.variable != "" else f"{self.year}_v{i}",
                    workflow=batch["execution"],
                    batch_flavor=self.batch_flavor,
                    slurm_partition=batch["batchPartition"],
                    slurm_memory=batch["batchMemory"],
                    slurm_max_runtime=batch["batchMaxRuntime"],
                    htcondor_partition=batch["batchPartition"],
                    htcondor_memory=batch["batchMemory"],
                    htcondor_max_runtime=batch["batchMaxRuntime"],
                )
            )
        return tasks

    def output(self):
        _, era_dirs = self.era_output_dirs()
        return [law.LocalFileTarget(os.path.join(era_dir, "ws_signal")) for _, era_dir in era_dirs]

    def run(self):
        print("Trees2WS ran through. Moving output to the subdirectory ./ws_signal")
        _, era_dirs = self.era_output_dirs()
        for _, era_dir in era_dirs:
            dst_folder = os.path.join(era_dir, "ws_signal")
            if self.batch_flavor == "slurm/psi":
                execute_command([f"xrdfs {PSI_XROOTD} mkdir -p {dst_folder}"], shell=True)
            else:
                os.makedirs(dst_folder, exist_ok=True)

            for current_file in glob.glob(os.path.join(era_dir, "ws_*", "*")):
                if os.path.dirname(current_file) == dst_folder:
                    continue
                if self.batch_flavor == "slurm/psi":
                    src = current_file.replace("/pnfs", f"{PSI_XROOTD}/pnfs")
                    execute_command([f"xrdcp -rf {src} {dst_folder}"], shell=True)
                else:
                    shutil.copy2(current_file, dst_folder)
        print("Copy completed.")
        return True
