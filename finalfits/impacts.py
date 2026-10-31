"""Impacts of the nuisance parameters (combineTool.py -M Impacts) with the POI values corrected.

The central POI values of the impacts json are replaced by the midpoint of the up and down variation of a parameter
frozen in the fit (e.g. MH), whose variations do not move the POIs; the impacts are recomputed from them.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def is_bkg_model_param(name: str) -> bool:
    """Parameters of the background model (statistical uncertainties of the fit to data)."""
    return "shapeBkg" in name or "env_pdf" in name


def _replace_central(values: list, central: float) -> list:
    return [v if i in (0, 2) else central for i, v in enumerate(values)]


def correct_impacts(data: dict, frozen_param: str = "MH", drop_bkg_model_params: bool = False) -> dict:
    """The impacts json with the corrected POI values and impacts."""
    pois = [poi["name"] for poi in data["POIs"]]
    corrected = {}
    for param in data["params"]:
        if param["name"] == frozen_param:
            corrected = {poi: 0.5 * (param[poi][0] + param[poi][2]) for poi in pois}
    if not corrected:
        logger.warning(f"Frozen parameter {frozen_param} not found in impacts json. Using POI central values.")
        corrected = {poi["name"]: poi["fit"][1] for poi in data["POIs"]}

    params = []
    for param in data["params"]:
        bkg = is_bkg_model_param(param["name"])
        if drop_bkg_model_params and bkg:
            continue
        info = {key: param[key] for key in ("fit", "prefit", "groups", "name", "type")}
        for poi in pois:
            low, high = param[poi][0], param[poi][2]
            # background model parameters are one sided; otherwise keep the corrected value if it lies between the
            # variations, else take their midpoint
            if bkg or (corrected[poi] - low) * (corrected[poi] - high) <= 0:
                central = corrected[poi]
            else:
                central = 0.5 * (low + high)
            info[poi] = _replace_central(param[poi], central)
        for poi in pois:
            info[f"impact_{poi}"] = max(abs(info[poi][0] - info[poi][1]), abs(info[poi][2] - info[poi][1]))
        params.append(info)
    return {
        "POIs": [
            {"fit": _replace_central(poi["fit"], corrected[poi["name"]]), "name": poi["name"]} for poi in data["POIs"]
        ],
        "method": data["method"],
        "params": params,
    }


def _values(values: list, indent: str) -> list[str]:
    return [f"{indent}{v:.16f}" + ("," if i < len(values) - 1 else "") for i, v in enumerate(values)]


def format_impacts(impacts: dict) -> str:
    """The json text in the layout of combineTool.py (16 decimals), as read by plotImpacts.py."""
    pois = [poi["name"] for poi in impacts["POIs"]]
    lines = ["{", '  "POIs": [']
    for i, poi in enumerate(impacts["POIs"]):
        lines += [
            "    {",
            '      "fit": [',
            *_values(poi["fit"], " " * 8),
            "      ],",
            f'      "name": "{poi["name"]}"',
        ]
        lines.append("    }" + ("," if i < len(pois) - 1 else ""))
    lines += ["  ],", f'  "method": "{impacts["method"]}",', '  "params": [']
    for i, param in enumerate(impacts["params"]):
        lines += ["    {", '      "fit": [', *_values(param["fit"], " " * 8), "      ],"]
        lines.append(f'      "groups": {json.dumps(param["groups"])},')
        lines += [f'      "impact_{poi}": {param[f"impact_{poi}"]:.16f},' for poi in pois]
        lines += [
            f'      "name": "{param["name"]}",',
            '      "prefit": [',
            *_values(param["prefit"], " " * 8),
            "      ],",
        ]
        for poi in pois:
            lines += [f'      "{poi}": [', *_values(param[poi], " " * 8), "      ],"]
        lines.append(f'      "type": "{param["type"]}"')
        lines.append("    }" + ("," if i < len(impacts["params"]) - 1 else ""))
    lines.append("  ]")
    return "\n".join(lines) + "\n}"


def corrected_path(path: str, drop_bkg_model_params: bool = False) -> str:
    """impacts.json -> impacts_corrected[_dropBkgModelParams].json"""
    return f"{os.path.splitext(path)[0]}_corrected{'_dropBkgModelParams' if drop_bkg_model_params else ''}.json"


def write_corrected_impacts(path: str, frozen_param: str = "MH", drop_bkg_model_params: bool = False) -> str:
    """Correct the impacts json at path; returns the path of the corrected json, next to it."""
    with open(path) as f:
        data = json.load(f)
    out = corrected_path(path, drop_bkg_model_params)
    with open(out, "w") as f:
        f.write(format_impacts(correct_impacts(data, frozen_param, drop_bkg_model_params)))
    logger.info(f"Written {out}")
    return out
