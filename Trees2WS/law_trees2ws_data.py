# law task for the Trees2WS step of the data; the conversion itself is finalfits.trees2ws
import os
import shutil

import law

from finalfits import trees2ws
from finalfits.utils import setup_logging

from framework import HTCondorWorkflow, SlurmWorkflow, Task
from Trees2WS.law_trees2ws import PSI_XROOTD, execute_command, load_analysis_config


class Trees2WSData(Task, HTCondorWorkflow, SlurmWorkflow, law.LocalWorkflow):
    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used for output folder naming")
    year = law.Parameter(default="2022", description="Year")
    batch_flavor = law.Parameter(default="htcondor", description="Batch system to use")

    def create_branch_map(self):
        return {0: 0}

    def ws_dir(self, output_dir):
        name = f"input_output_data_{self.variable}_{self.year}" if self.variable else f"input_output_data_{self.year}"
        return os.path.join(output_dir, "input_output_data", name, "ws")

    def output(self):
        config = load_analysis_config(self.year, self.variable)
        return law.LocalFileTarget(os.path.join(self.ws_dir(self.output_dir or config["outputFolder"]), "allData.root"))

    def run(self):
        setup_logging()
        config = load_analysis_config(self.year, self.variable)
        output_dir = self.output_dir or config["outputFolder"]

        # On the PSI Tier 3, the job writes to the local scratch and copies the output at the end
        psi = self.batch_flavor == "slurm/psi"
        work_dir = f"/scratch/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}" if psi else output_dir

        all_data = trees2ws.convert_data_file(
            config["inputFiles"]["Trees2WSData"],
            os.path.join(self.ws_dir(work_dir), "allData.root"),
            trees2ws.Trees2WSConfig.from_dict(config["trees2wsCfg"]),
        )

        if psi:
            final_ws_dir = self.ws_dir(output_dir)
            execute_command([f"xrdfs {PSI_XROOTD} mkdir -p {final_ws_dir}"], shell=True)
            if "/work" in output_dir:
                execute_command(["cp", "-rf", str(all_data), f"{final_ws_dir}/allData.root"])
            else:
                execute_command(["xrdcp", "-rf", str(all_data), f"{PSI_XROOTD}/{final_ws_dir}/allData.root"])
            shutil.rmtree(work_dir)
