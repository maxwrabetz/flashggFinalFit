# law tasks of the differential spectra; the spectra themselves are finalfits.spectra
import os

import law
import yaml

# CreateAsimovFit, CreateUnblindedFit and PValueCalculation; law/analysis.py also gets the Combine tasks from here
from Combine.law_combine import *  # noqa: F403
from finalfits import spectra
from finalfits.analysis import combineVariableDict

HIGGS_MASS = spectra.HIGGS_MASS


def convert_boolean_string(string):
    return string in (True, "True", "true")


def load_analysis_config(year, variable):
    # config/<year>_<variable>.yml, or config/<year>_inclusive.yml for the inclusive analysis
    name = f"{year}_{variable}.yml" if variable else f"{year}_inclusive.yml"
    with open(os.path.join(os.environ["ANALYSIS_PATH"], "config", name)) as f:
        return yaml.safe_load(f)


class CreateDiffSpectra(law.Task):
    """Create a differential cross-section spectrum in the fiducial phase space.

    Subclasses can select another cross-section family written by the theory-input
    producer by overriding ``cross_section_prefix`` and ``spectrum_output_stem``.
    """

    phase_space = "fiducial"
    cross_section_prefix = "fidXS"
    cross_section_label = r"\text{fid}"
    spectrum_output_stem = "spectra"
    upper_panel_scale = 1.0

    output_dir = law.Parameter(default="", description="Path to the output directory")
    variable = law.Parameter(default="", description="Variable to be used for output folder naming")
    year = law.Parameter(default="2022", description="Year")
    is_unblinded = law.Parameter(
        default=False, description="Flag that signifies if spectrum is created for the unblinded results."
    )
    log_x = law.Parameter(default=False, description="Use a logarithmic x axis for the spectrum.")

    batch_flavor = law.Parameter(default="slurm", description="Special treatment for PSI Slurm batch system")
    batch_system = law.Parameter(default="slurm", description="Batch system to use")

    def spectrum_phase_space(self):
        return spectra.PhaseSpace(
            self.phase_space,
            self.cross_section_prefix,
            self.cross_section_label,
            self.spectrum_output_stem,
            self.upper_panel_scale,
        )

    def analysis_output_dir(self, config):
        return self.output_dir or config["outputFolder"]

    def requires(self):
        config = load_analysis_config(self.year, self.variable)
        output_dir = self.analysis_output_dir(config)
        version = self.variable or "inclusive"
        common = dict(variable=self.variable, output_dir=output_dir, year=self.year, batch_flavor=self.batch_flavor)
        if not convert_boolean_string(self.is_unblinded):
            return {"CreateAsimovFit": CreateAsimovFit(**common, version=version, workflow=self.batch_system)}  # noqa: F405
        hesse = config["combine_hesse"]
        batch = {
            f"{system}_{key}": hesse[f"batch{option}"]
            for system in ("slurm", "htcondor")
            for key, option in (("partition", "Partition"), ("memory", "Memory"), ("max_runtime", "MaxRuntime"))
        }
        return {
            "PValueCalculation": PValueCalculation(**common, version=version, workflow=self.batch_system, **batch),  # noqa: F405
            "CreateUnblindedFit": CreateUnblindedFit(**common, version=version, workflow=self.batch_system),  # noqa: F405
        }

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        # nothing for the inclusive analysis: a spectrum of a single POI makes no sense
        if self.variable == "":
            return []
        config = load_analysis_config(self.year, self.variable)
        fit_dir = os.path.join(self.analysis_output_dir(config), "Combine", f"runFits_{self.variable}")
        names = spectra.output_names(
            self.spectrum_output_stem, convert_boolean_string(self.is_unblinded), convert_boolean_string(self.log_x)
        )
        return [law.LocalFileTarget(os.path.join(fit_dir, name)) for name in names]

    def run(self):
        if self.variable == "":
            raise ValueError("Variable is not set. Please set the variable parameter to a valid value.")
        config = load_analysis_config(self.year, self.variable)
        plot_config = os.path.join(
            os.environ["ANALYSIS_PATH"], "Plots", "Spectra", "config", f"{self.year}_{self.variable}.yml"
        )
        with open(plot_config) as f:
            plotting_config = yaml.safe_load(f)
        spectra.make_spectrum(
            os.path.join(self.analysis_output_dir(config), "Combine", f"runFits_{self.variable}"),
            combineVariableDict(self.variable, self.year)["paramStrNoOne"],  # in the order of the bins
            config,
            plotting_config,
            self.year,
            self.spectrum_phase_space(),
            unblinded=convert_boolean_string(self.is_unblinded),
            log_x=convert_boolean_string(self.log_x),
        )


class CreateTotalDiffSpectra(CreateDiffSpectra):
    """Create differential spectra extrapolated to the total phase space."""

    phase_space = "total"
    cross_section_prefix = "totXS"
    cross_section_label = r"\text{tot}"
    spectrum_output_stem = "spectra_total"
    upper_panel_scale = 1e-3
