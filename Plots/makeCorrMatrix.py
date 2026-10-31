#!/usr/bin/env python3
# Kept for the law tasks that call this path: runs scripts/plot_corr_matrix.py (finalfits.correlation_plots)
import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "plot_corr_matrix.py"),
    run_name="__main__",
)
