"""Analysis-wide definitions that are shared with the steps which are not rewritten yet.

They still live in commonTools/ (commonObjects.py, commonTools.py, XSBRMap.py) and
Signal/tools/replacementMap.py because Combine/, Plots/ and the law tasks use them too. This module
is the only place through which the finalfits package reaches into them, so they can be moved here
once nothing else needs them there.
"""

import importlib.util
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "commonTools") not in sys.path:
    sys.path.append(str(_REPO / "commonTools"))

from commonObjects import (  # noqa: E402
    bkgWSName__,
    combineVariableDict,
    decayMode,
    differentialProcTable_,
    jetVariables,
    lumiMap,
    lumiScaleFactor,
    outputWSName__,
    outputWSNuisanceTitle__,
    outputWSObjectTitle__,
    productionModes,
)
from commonTools import (  # noqa: E402
    dataToProc,
    extractListOfProcs,
    extractWSFileNames,
    procToData,
    procToDatacardName,
)
from XSBRMap import extractXSBR, globalXSBRMap  # noqa: E402


def _load(name: str, path: Path):
    # by path: Signal/tools also holds modules whose names clash with other directories
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


globalReplacementMap = _load("replacementMap", _REPO / "Signal" / "tools" / "replacementMap.py").globalReplacementMap


def luminosity(year) -> float:
    """Integrated luminosity in fb^-1 of a year of lumiMap or of combined years like 2022_2023 or 2022,2023."""
    if str(year) in lumiMap:
        return lumiMap[str(year)]
    return sum(lumiMap[y] for y in re.split(r"[,_]", str(year)))


def combine_models() -> dict:
    """Physics models of text2workspace by name (Combine/models.py)."""
    return _load("models", _REPO / "Combine" / "models.py").models


def datacard_systematics():
    """Fresh copies of (experimental, theory, signal shape) systematics of Datacard/systematics.py."""
    module = _load("systematics", _REPO / "Datacard" / "systematics.py")
    return module.experimental_systematics, module.theory_systematics, module.signal_shape_systematics


__all__ = [
    "bkgWSName__",
    "combineVariableDict",
    "combine_models",
    "dataToProc",
    "datacard_systematics",
    "decayMode",
    "differentialProcTable_",
    "extractListOfProcs",
    "extractWSFileNames",
    "extractXSBR",
    "globalReplacementMap",
    "globalXSBRMap",
    "jetVariables",
    "lumiMap",
    "lumiScaleFactor",
    "luminosity",
    "outputWSName__",
    "outputWSNuisanceTitle__",
    "outputWSObjectTitle__",
    "procToData",
    "procToDatacardName",
    "productionModes",
]
