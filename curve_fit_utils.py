
"""

Script and functions to perform curve-fitting

Author: Johan Blakkisrud

This is only for descriptive reasons and for my own 
use and was hastily put together for the TACTIC
challenge.

Documentation to be written

TODO: 
- Add more fitting functions 
- Flag when fit is too bad (r-square-cutoff?) and go
  to a piecewise-integration instead
- Restruture the test-set-up to a proper function


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
import os
from lmfit import Model
import lmfit
from dataclasses import dataclass, field
import scipy.integrate as si

path_to_tac = "data/phase0/received_data.csv"
data = pd.read_csv(path_to_tac, index_col=False)


@dataclass
class TACData:
    structure_name: str
    time: np.ndarray
    activity: np.ndarray


@dataclass(order=True)
class ModelFit:
    sort_index: float = field(init=False)
    tac: TACData
    model: Model
    integral: float
    result: lmfit.model.ModelResult  # ModelResult
    aic: float

    def __post_init__(self):
        self.sort_index = self.aic


def mono_exp(t, a0, lam):
    return a0*np.exp(t*lam)


def bi_exp(t, a0, lam1, lam2):
    return a0*(np.exp(t*lam1)-np.exp(t*lam2))


def load_tac_data(path_to_file: str):
    """Loads from mem and returns a list of TAC-data"""

    data = pd.read_csv(path_to_file, index_col=False)

    t = data["times"].values

    list_of_structures = (list(data.columns))
    # Remove times
    list_of_structures = [x for x in list_of_structures if "times" not in x]

    tac_collection = []

    for org in list_of_structures:
        a = data[org].values

        tac_collection.append(TACData(structure_name=org, time=t, activity=a))

    return tac_collection


def plot_tac(tac: TACData, path_to_plot):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(tac.time, tac.activity, 'o', label=tac.structure_name)

    ax.set_xlim([0, None])
    ax.set_ylim([0, None])

    plt.legend()

    plt.savefig(path_to_plot)


def fit_data(tac: TACData, fitting_method: str):

    t = tac.time
    a = tac.activity

    if fitting_method == "mono_exp":

        mono_exp_model = Model(mono_exp)
        mono_exp_params = mono_exp_model.make_params(a0=2, lam=-0.01)
        result = mono_exp_model.fit(a, mono_exp_params, t=t)

        # Perform integration with the quad-function

        a0 = result.params["a0"].value
        lam = result.params["lam"].value

        I = (si.quad(mono_exp, 0, np.inf, args=(a0, lam)))[0]

    if fitting_method == "bi_exp":

        bi_exp_model = (Model(bi_exp))
        bi_exp_params = bi_exp_model.make_params(a0=2, lam1=0.01, lam2=-0.01)
        result = bi_exp_model.fit(a, bi_exp_params, t=t)

        # Perform integration

        a0 = result.params["a0"].value
        lam1 = result.params["lam1"].value
        lam2 = result.params["lam2"].value

        I = (si.quad(bi_exp, 0, np.inf, args=(a0, lam1, lam2)))[0]

    model_fit = ModelFit(tac=tac,
                         model=result.model,
                         integral=I,
                         result=result,
                         aic=result.aic)

    return model_fit
