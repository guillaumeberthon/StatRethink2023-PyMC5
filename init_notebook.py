import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pymc as pm
import xarray as xr
import arviz as az
from scipy import stats as stats

import utils as utils
from functools import partial
from matplotlib import pyplot as plt
from typing import List, Callable, Optional, Tuple, Any

# ---
# Print imports / aliases
with open(__file__) as f:
    lines = f.readlines()

from colorama import Fore

def print_import(import_line):
    parts = [p.strip() for p in import_line.strip().split(" ") if p.strip()]
    if not parts:
        return

    if parts[0] == 'import':
        module = parts[1]
        alias = parts[3] if len(parts) > 3 else module
        msg = Fore.GREEN + 'import' \
            + Fore.BLUE + f" {module} " \
            + Fore.GREEN + "as" \
            + Fore.BLUE + f" {alias}"
    elif parts[0] == 'from':
        module = parts[1]
        submodule = parts[3]
        alias = parts[5] if len(parts) > 5 else submodule
        msg = Fore.GREEN + 'from' \
            + Fore.BLUE + f" {module} "\
            + Fore.GREEN + 'import' \
            + Fore.BLUE + f" {submodule} " \
            + Fore.GREEN + "as" \
            + Fore.BLUE + f" {alias}"
    else:
        return
    
    print(msg)

print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-'* 44}")
for l in lines:
    if "# ---" in l:
        break
    print_import(l)

from watermark import watermark
print(Fore.RED + f"Watermark:\n{'-'* 10}")
print(Fore.BLUE + watermark())
print(Fore.BLUE + watermark(iversions=True, globals_=globals()))

import warnings
warnings.filterwarnings("ignore")

from matplotlib import style
STYLE = "statistical-rethinking-2023.mplstyle"
style.use(STYLE)
