"""

Test file for the curve_fit_utils functions

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
import os
from lmfit import Model
from dataclasses import dataclass

import scipy.integrate as si

import curve_fit_utils as cu
from curve_fit_utils import TACData
from curve_fit_utils import ModelFit

path_to_tac = "data/phase0/received_data.csv"

tac_list = cu.load_tac_data(path_to_tac)

path_to_plots = "output/plots"
path_to_fig = os.path.join(path_to_plots, "testfig.png")

cu.plot_tac(tac_list[0], path_to_fig)

# TODO: Add debug-flag, function and writing to file

for tac in tac_list:

    cand_1 = cu.fit_data(tac, "mono_exp")
    cand_2 = cu.fit_data(tac, "bi_exp")

    candidates = [cand_1, cand_2]

    lowest_aic = min(candidates)

    TIAC_lowest_aic = lowest_aic.integral

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(tac.structure_name)

    t_points = tac.time
    a_points = tac.activity

    result_lowest_aic = lowest_aic.result

    t_syn = np.arange(250)
    a_best_fit = result_lowest_aic.eval(t=t_syn)

    ax.plot(t_points, a_points, 'o')
    ax.plot(t_syn, a_best_fit, '--')

    print(tac.structure_name)
    print(lowest_aic.result.fit_report())

    fname_output = "plot_" + tac.structure_name + ".png"

    path_to_fig = os.path.join(path_to_plots, fname_output)

    plt.savefig(path_to_fig)

print("Have a nice day")
