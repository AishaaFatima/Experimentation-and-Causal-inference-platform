"""Run from the backend folder:  python scripts/check_pymc_speed.py
Prints whether PyTensor found a C++ compiler and how long each PyMC step takes on this machine."""
import logging, sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

t = time.time()
import pytensor
import pymc
print(f"import pymc/pytensor : {time.time() - t:6.1f} s   (pymc {pymc.__version__})")
print(f"C++ compiler (cxx)   : {pytensor.config.cxx!r}   <- empty means none found")

import pandas as pd
from app.services.bayes import bayesian_ab
from app.services.simulator import simulate_ab

lalonde = Path(__file__).resolve().parents[1] / "data" / "lalonde.csv"
for label, df, g, y in [("binary A/B, 1st run ", simulate_ab(1500, "binary", 0.1, 0.02, seed=1), "group", "outcome"),
                        ("binary A/B, 2nd run ", simulate_ab(1500, "binary", 0.1, 0.02, seed=2), "group", "outcome"),
                        ("LaLonde (continuous)", pd.read_csv(lalonde), "treat", "re78")]:
    t = time.time()
    bayesian_ab(df, g, y, draws=300, tune=200, chains=2)
    print(f"{label} : {time.time() - t:6.1f} s", flush=True)
