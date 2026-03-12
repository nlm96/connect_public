"""
Script for making triangle plots from analysed Monte Python jobs

 - Andreas Nygaard (2020)

"""

import argparse
import importlib.util
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


textwidth = 440  # JCAP uses a textwidth of 440 pts
height = 2
width = textwidth / 72.27
fontsize = 11 / 1.2

latex_preamble = [
    r"\usepackage{lmodern}",
    r"\usepackage{amsmath}",
    r"\usepackage{amsfonts}",
    r"\usepackage{mathtools}",
    r"\usepackage{siunitx}",
]
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cmr10",
        "font.size": fontsize,
        "mathtext.fontset": "cm",
        "text.latex.preamble": "\n".join(latex_preamble),
    }
)


"""______________________ Set options and parameters ______________________"""

# Set output path (end with "/")
output_path = (
    "/home/maanson/Speciale/MontePythonCluster/chains/dcdm/Full_planck+lensing/"
)

# Set names as they appear in the parameter file
# names = ["H0", "omega_cdm", "omega_b"]
names = [
    "omega_b",
    "omega_cdm",
    "H0",
    "ln10^{10}A_s",
    "n_s",
    "tau_reio",
    "sigma8",
    "z_reio",
    "Omega_Lambda",
    "YHe",
    "A_s",
    "100*theta_s",
]

# # Set labelnames as you would like them to appear in the figure (supports latex syntax)
labelnames = [
    r"$100\times\omega_{\mathrm{b}}$",
    r"$\omega_{\mathrm{cdm}}$",
    r"$H_0$",
    r"$\ln(10^{10}A_s)$",
    r"$n_s$",
    r"$\tau_{\mathrm{reio}}$",
    r"$\sigma_8$",
    r"$z_{\mathrm{reio}}$",
    r"$\Omega_{\Lambda}$",
    r"$Y_{\mathrm{He}}$",
    r"$10^{-9}A_s$",
    r"$100\theta_s$",
]


# Set x-limits and ticks for each parameter. If a rows is left blanck, python will
# automatically find limits and ticks.
# xlimits = [[65, 70], [0.115, 0.126], [2.16, 2.32]]
xlimits = [[] for _ in names]
# ticklist = [[66, 67.5, 69], [0.117, 0.121, 0.125], [2.2, 2.25, 2.3]]
ticklist = [[] for _ in names]

# Choose which jobs to be used for the figure.
# The jobs must have already been analysed with the '--all' option,
# and their location must be in the chains folder.
# If they are in a subfolder, you must write both the subfolder and job name
# e.g. 'subfolder/jobname'
# JOBnames = ["basePlanck01", "15basePlanck01"]


JOBnames = None

# Each job is given a legend label
# legend_labels = [r"$\Lambda$CDM (PLANCK-2018)", r"$\Lambda$CDM (PLANCK-2015)"]

# Default legend labels
legend_labels = [
    "plot 1",
    "plot 2",
    "plot 3",
    "plot 4",
    "plot 5",
]


# The colors and styles of the 1D posteriors will be given to the different jobs
# in this order
style1d = ["k", "r--", "b-.", "y-.", "g--"]

# The corresponding order for the 95- and 68-level contours
color95 = ["k", "r", "b", "y", "g"]
color68 = ["k", "r", "b", "y", "g"]

# The transparency of the contours
alpha95 = 0.2
alpha68 = 0.4


## layout options
square = True  # should each subfigure be a square in shape?

ticksize = fontsize  # fontsize of ticklabels (default to JCAP fontsize)

only_1D = False  # plot only the 1D posteriors in a horizontal plot

dist = 0  # distance between subplots


""" -------------------"""


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("plot_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def setup_matplotlib(latex_preamble, fontsize):
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "cmr10",
            "font.size": fontsize,
            "mathtext.fontset": "cm",
            "text.latex.preamble": "\n".join(latex_preamble),
        }
    )


def sanitize(name):
    s = (
        name.replace("*", "star")
        .replace("^", "exp")
        .replace("{", "")
        .replace("}", "")
        .replace(" ", "")
        .replace(".", "p")
    )
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _job_base(j):
    return j.split("/")[-1]

def _hist_path(job, pname):
    return f"chains/{job}/plots/{_job_base(job)}_{pname}.hist"

def _hist_exists(job, pname):
    return os.path.exists(_hist_path(job, pname))

def _dat2d_path(job, p1, p2):
    return f"chains/{job}/plots/{_job_base(job)}_2d_{p1}-{p2}.dat"











parser = argparse.ArgumentParser(
    description="""
    Make triangle plots from analyzed MontePython jobs using a Python config file.\n
    The jobs must have already been analyzed with the '--all' option, and their location must be in the chains folder.\n
    If they are in a subfolder, you must write both the subfolder and job name, e.g. 'subfolder/jobname'.
    """
)

parser.add_argument(
    "--JOBnames",
    nargs="+",  # This means that the argument can be repeated multiple times
    required=False,
    default=None,
    help="""
    Job names corresponding to the different MontePython analyses.
    You can provide multiple job names this way: --jobnames "job1" "job2" "job3"
    """,
)

parser.add_argument(
    "--legend_labels",
    nargs="+",
    required=False,
    default=None,
    help="""
    Legend labels for the different jobs.
    You can provide multiple labels this way: --legend_labels "label1" "label2" "label3"
    """,
)

parser.add_argument(
    "--output_path",
    type=str,
    required=False,
    default=None,
    help="Output path for the final plot.",
)

parser.add_argument(
    "--config", type=str, required=True, help="Path to the Python configuration file."
)


args = parser.parse_args()

cfg = load_config(args.config)

# Load the args if they are provided

if args.JOBnames:
    JOBnames = args.JOBnames

# Check if JOBnames are in the config file
# if "JOBnames" in cfg:
if hasattr(cfg, "JOBnames"):
    JOBnames = cfg.JOBnames

if not JOBnames:
    raise ValueError(
        "JOBnames must be provided either as an argument or in the config file"
    )

# Set legend_labels to be the same as JOBnames if not provided

legend_labels = JOBnames

if args.legend_labels:
    legend_labels = args.legend_labels

if hasattr(cfg, "legend_labels"):
    legend_labels = cfg.legend_labels

if hasattr(cfg, "style1d"):
    style1d = cfg.style1d

if hasattr(cfg, "color95"):
    color95 = cfg.color95

if hasattr(cfg, "color68"):
    color68 = cfg.color68

if hasattr(cfg, "alpha95"):
    alpha95 = cfg.alpha95

if hasattr(cfg, "alpha68"):
    alpha68 = cfg.alpha68

if hasattr(cfg, "square"):
    square = cfg.square

if hasattr(cfg, "ticksize"):
    ticksize = cfg.ticksize

if hasattr(cfg, "only_1D"):
    only_1D = cfg.only_1D

if hasattr(cfg, "dist"):
    dist = cfg.dist

if hasattr(cfg, "xlimits"):
    xlimits = cfg.xlimits

if hasattr(cfg, "ticklist"):
    ticklist = cfg.ticklist

if hasattr(cfg, "labelnames"):
    labelnames = cfg.labelnames

if hasattr(cfg, "names"):
    names = cfg.names

if hasattr(cfg, "textwidth"):
    textwidth = cfg.textwidth

if hasattr(cfg, "height"):
    height = cfg.height

width = textwidth / 72.27

if hasattr(cfg, "width"):
    width = cfg.width

if args.output_path:
    output_path = args.output_path

if hasattr(cfg, "output_path"):
    output_path = cfg.output_path

if hasattr(cfg, "fontsize"):
    fontsize = cfg.fontsize

if hasattr(cfg, "latex_preamble"):
    latex_preamble = cfg.latex_preamble

setup_matplotlib(latex_preamble, fontsize)



# --- NEW: resolve alias groups and prune dead axes ---

# 1) Decide groups (prefer name_groups; else emulate groups from names)
if hasattr(cfg, "name_groups"):
    name_groups = cfg.name_groups
    # basic sanity
    if not hasattr(cfg, "labelnames"):
        raise ValueError("When using name_groups, you must define labelnames with same length.")
    if len(cfg.labelnames) != len(name_groups):
        raise ValueError("labelnames must have the same length as name_groups.")
    labelnames = cfg.labelnames
    # xlimits/ticklist may or may not be present; if present, must match length
    if hasattr(cfg, "xlimits"):
        if len(cfg.xlimits) != len(name_groups):
            raise ValueError("xlimits length must match name_groups length.")
        xlimits = cfg.xlimits
    else:
        xlimits = [[] for _ in range(len(name_groups))]
    if hasattr(cfg, "ticklist"):
        if len(cfg.ticklist) != len(name_groups):
            raise ValueError("ticklist length must match name_groups length.")
        ticklist = cfg.ticklist
    else:
        ticklist = [[] for _ in range(len(name_groups))]
else:
    # fallback: behave like your old code (each name is its own group)
    name_groups = [[n] for n in names]

# 2) For each group/axis, check if ANY alias exists in ANY job (via .hist). If not, drop that axis.
active_indices = []
for gi, group in enumerate(name_groups):
    found_any = False
    for Jn in JOBnames:
        if any(_hist_exists(Jn, cand) for cand in group):
            found_any = True
            break
    if found_any:
        active_indices.append(gi)

if not active_indices:
    raise ValueError("None of the requested parameters (via name_groups/names) exist in the supplied jobs.")

# 3) Build canonical axis names from the FIRST alias in each active group.
#    These canonical names are only used to create axis IDs & sanitize.
axis_names   = [name_groups[i][0] for i in active_indices]
labelnames   = [labelnames[i]       for i in active_indices] if 'labelnames' in locals() else labelnames
xlimits      = [xlimits[i]          for i in active_indices]
ticklist     = [ticklist[i]         for i in active_indices]

# 4) For each job & axis, pick the FIRST existing alias for that job (else None).
selected_name = []   # list of dicts: selected_name[job_index][axis_index] = chosen alias or None
for dex, Jn in enumerate(JOBnames):
    per_job = {}
    for new_idx, gi in enumerate(active_indices):
        chosen = None
        for cand in name_groups[gi]:
            if _hist_exists(Jn, cand):
                chosen = cand
                break
        per_job[new_idx] = chosen  # can be None
    selected_name.append(per_job)

# 5) From now on, use axis_names as "names"
names = axis_names
dim   = len(names)



"""____________________________ Automatic from here _____________________________"""


safe_names = [sanitize(name) for name in names]


for j, name in enumerate(safe_names):
    exec("i_" + name + "=j+1")

# dim = len(names)

# Track which job actually produced a 1D line per axis (for legend later)
has_1d = [[False for _ in range(dim)] for _ in range(len(JOBnames))]

# for dex, Jn in enumerate(JOBnames):
#     for name in names:
#         with open(
#             "chains/"
#             + JOBnames[dex]
#             + "/plots/"
#             + JOBnames[dex].split("/")[-1]
#             + "_"
#             + name
#             + ".hist"
#         ) as f:
#             exec(
#                 "h_"
#                 + safe_names[names.index(name)]
#                 + "_"
#                 + Jn.split("/")[-1]
#                 + ' = np.loadtxt(f, delimiter=",", dtype="float", comments="#", skiprows=1, usecols=None)'
#             )

for dex, Jn in enumerate(JOBnames):
    for j, axis_name in enumerate(names):
        pname = selected_name[dex][j]  # the alias that exists for this job (or None)
        if not pname:
            continue
        p = _hist_path(Jn, pname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            exec(
                "h_{safe}_{job} = np.loadtxt(f, delimiter=',', dtype='float', comments='#', skiprows=1, usecols=None)".format(
                    safe=safe_names[j], job=_job_base(Jn)
                )
            )
            has_1d[dex][j] = True
            


line_handles = [None for _ in range(len(JOBnames))]  # for legend

if only_1D:
    fig = plt.figure(figsize=(width, height))
    for j, name in enumerate(names):
        safe_name = safe_names[j]
        exec("ax" + safe_name + "1d = fig.add_subplot(1,dim+1,j+2)")
        if len(xlimits[j]):
            exec("ax" + safe_name + "1d.set_xlim(xlimits[j])")
        if len(ticklist[j]):
            exec("ax" + safe_name + "1d.set_xticks(ticklist[j])")
        # for i, Jn in enumerate(JOBnames):
        #     Jn = Jn.split("/")[-1]
        #     exec(
        #         safe_name
        #         + "_"
        #         + Jn
        #         + "_1d_plot=ax"
        #         + safe_name
        #         + "1d.plot(h_"
        #         + safe_name
        #         + "_"
        #         + Jn
        #         + "[0],h_"
        #         + safe_name
        #         + "_"
        #         + Jn
        #         + "[1],style1d[i])[0]"
        #     )
        

        for i, Jn in enumerate(JOBnames):
            if not has_1d[i][j]:
                continue
            Jb = Jn.split("/")[-1]
            exec(
                "{safe}_{job}_1d_plot = ax{suffix}.plot(h_{safe}_{job}[0], h_{safe}_{job}[1], style1d[{i}])[0]".format(
                    safe=safe_names[j],
                    job=Jb,
                    suffix=(safe_name + "1d"),  # already defined in your code per branch
                    i=i,
                )
            )
            # store a handle for legend if we don't have one for this job yet
            if line_handles[i] is None:
                line_handles[i] = eval(f"{safe_names[j]}_{Jb}_1d_plot")

                
        
        
            exec("ax" + safe_name + "1d.get_yaxis().set_visible(False)")
            exec("ax" + safe_name + "1d.set_xlabel(labelnames[j],labelpad=10)")
            exec("ax" + safe_name + "1d.tick_params(labelsize=ticksize)")
else:
    if square:
        fig = plt.figure(figsize=(width, width))
    else:
        fig = plt.figure(figsize=(width, height))
    for j, name in enumerate(names):
        safe_name = safe_names[j]
        exec(
            "ax"
            + safe_name
            + "1d = fig.add_subplot(dim,dim,i_"
            + safe_name
            + "*dim-(dim-i_"
            + safe_name
            + "))"
        )
        if len(xlimits[j]):
            exec("ax" + safe_name + "1d.set_xlim(xlimits[j])")
        if len(ticklist[j]):
            exec("ax" + safe_name + "1d.set_xticks(ticklist[j])")
        for i, Jn in enumerate(JOBnames):
            Jn = Jn.split("/")[-1]
            exec(
                safe_name
                + "_"
                + Jn
                + "_1d_plot=ax"
                + safe_name
                + "1d.plot(h_"
                + safe_name
                + "_"
                + Jn
                + "[0],h_"
                + safe_name
                + "_"
                + Jn
                + "[1],style1d[i])[0]"
            )
            if line_handles[i] is None:
                line_handles[i] = eval(safe_name + "_" + Jn + "_1d_plot")
                
            if j == len(names) - 1:
                exec("ax" + safe_name + "1d.get_yaxis().set_visible(False)")
                exec(
                    "ax"
                    + safe_name
                    + "1d.set_xlabel(labelnames[j],labelpad=10, rotation=45)"
                )
                exec("ax" + safe_name + "1d.tick_params(labelsize=ticksize)")
            else:
                exec("ax" + safe_name + "1d.get_yaxis().set_visible(False)")
                exec("ax" + safe_name + "1d.tick_params(labelbottom=False)")


if not only_1D:
    for dex, Jn in enumerate(JOBnames):
        for i in range(0, dim):
            for j in range(0, dim):
                if i > j:
                    if dex == 0:
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + f"2d = fig.add_subplot({dim:.0f},{dim:.0f},i_"
                            + safe_names[i]
                            + f"*{dim:.0f}-({dim:.0f}-i_"
                            + safe_names[j]
                            + "))"
                        )
                    # if path.exists(
                    #     "chains/"
                    #     + Jn
                    #     + "/plots/"
                    #     + Jn.split("/")[-1]
                    #     + "_2d_"
                    #     + names[i]
                    #     + "-"
                    #     + names[j]
                    #     + ".dat"
                    # ):
                    #     ii = i
                    #     jj = j
                    #     switch = False
                    # else:
                    #     ii = j
                    #     jj = i
                    #     switch = True
                    # with open(
                    #     "chains/"
                    #     + Jn
                    #     + "/plots/"
                    #     + Jn.split("/")[-1]
                    #     + "_2d_"
                    #     + names[ii]
                    #     + "-"
                    #     + names[jj]
                    #     + ".dat"
                    # ) as f:
                    #     x = []
                    #     y = []
                    #     x95_list = []
                    #     y95_list = []
                    #     x68_list = []
                    #     y68_list = []
                    #     cnt95 = 0
                    #     cnt68 = -1
                    #     for line in f:
                    #         if line != "\n" and cnt68 == -1 and line[0] != "#":
                    #             x.append(float(line.split("\t")[0]))
                    #             y.append(float(line.split("\t")[1].split("\n")[0]))
                    #         elif line == "\n" and cnt68 == -1:
                    #             cnt95 += 1
                    #             x95_list.append(x)
                    #             y95_list.append(y)
                    #             x = []
                    #             y = []
                    #         elif line[0] == "#" and cnt95 > 0:
                    #             cnt68 += 1
                    #         elif line != "\n" and cnt68 >= 0 and line[0] != "#":
                    #             x.append(float(line.split("\t")[0]))
                    #             y.append(float(line.split("\t")[1].split("\n")[0]))
                    #         elif line == "\n" and cnt68 >= 0:
                    #             cnt68 += 1
                    #             x68_list.append(x)
                    #             y68_list.append(y)
                    #             x = []
                    #             y = []
                    # if switch:
                    #     x95 = y95_list
                    #     y95 = x95_list
                    #     x68 = y68_list
                    #     y68 = x68_list
                    # else:
                    #     x95 = x95_list
                    #     y95 = y95_list
                    #     x68 = x68_list
                    #     y68 = y68_list
                    
                    
                    # NEW begin: resolve per-job aliases for the 2D file
                    p_i = selected_name[dex][i]   # alias for axis i in this job (or None)
                    p_j = selected_name[dex][j]   # alias for axis j in this job (or None)
                    if not p_i or not p_j:
                        continue

                    # try both orders
                    path_ij = _dat2d_path(Jn, p_i, p_j)
                    path_ji = _dat2d_path(Jn, p_j, p_i)
                    if os.path.exists(path_ij):
                        dat_path = path_ij
                        switch = False
                    elif os.path.exists(path_ji):
                        dat_path = path_ji
                        switch = True
                    else:
                        continue

                    with open(dat_path) as f:
                        x = []; y = []
                        x95_list = []; y95_list = []
                        x68_list = []; y68_list = []
                        cnt95 = 0
                        cnt68 = -1
                        for line in f:
                            if line != "\n" and cnt68 == -1 and line[0] != "#":
                                x.append(float(line.split("\t")[0]))
                                y.append(float(line.split("\t")[1].split("\n")[0]))
                            elif line == "\n" and cnt68 == -1:
                                cnt95 += 1
                                x95_list.append(x); y95_list.append(y)
                                x = []; y = []
                            elif line[0] == "#" and cnt95 > 0:
                                cnt68 += 1
                            elif line != "\n" and cnt68 >= 0 and line[0] != "#":
                                x.append(float(line.split("\t")[0]))
                                y.append(float(line.split("\t")[1].split("\n")[0]))
                            elif line == "\n" and cnt68 >= 0:
                                cnt68 += 1
                                x68_list.append(x); y68_list.append(y)
                                x = []; y = []

                    if switch:
                        x95, y95 = y95_list, x95_list
                        x68, y68 = y68_list, x68_list
                    else:
                        x95, y95 = x95_list, y95_list
                        x68, y68 = x68_list, y68_list
                    # NEW end

                    
                    
                    for cont in range(cnt95 - 2):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.fill(x95[cont],y95[cont],color=color95[dex],alpha=alpha95,lw=0)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + '2d.plot(x95[cont],y95[cont],"-",color=color95[dex],lw=1.5,alpha=alpha95+0.2)'
                        )
                    for cont in range(cnt68 - 2):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.fill(x68[cont],y68[cont],color=color68[dex],alpha=alpha68,lw=0)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + '2d.plot(x68[cont],y68[cont],"-",color=color68[dex],lw=1.5,alpha=alpha68+0.2)'
                        )
                    if len(xlimits[i]):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_ylim(xlimits[i])"
                        )
                    if len(xlimits[j]):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_xlim(xlimits[j])"
                        )
                    if len(ticklist[i]):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_yticks(ticklist[i])"
                        )
                    if len(ticklist[j]):
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_xticks(ticklist[j])"
                        )
                    exec(
                        "ax"
                        + safe_names[i]
                        + safe_names[j]
                        + "2d.tick_params(labelleft=False)"
                    )
                    exec(
                        "ax"
                        + safe_names[i]
                        + safe_names[j]
                        + "2d.tick_params(labelbottom=False)"
                    )
                    if i == dim - 1:
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.tick_params(labelbottom=True)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_xlabel(labelnames[j],labelpad=10, rotation=45)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.tick_params(labelsize=ticksize)"
                        )
                    if j == 0:
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.tick_params(labelleft=True)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.set_ylabel(labelnames[i],labelpad=10, rotation=0)"
                        )
                        exec(
                            "ax"
                            + safe_names[i]
                            + safe_names[j]
                            + "2d.tick_params(labelsize=ticksize)"
                        )


fig.align_ylabels()
lines = []
labels = []
# for i, Jn in enumerate(JOBnames):
#     Jn = Jn.split("/")[-1]
#     lines.append(eval(safe_names[0] + "_" + Jn + "_1d_plot"))
#     labels.append(legend_labels[i])
for i, Jn in enumerate(JOBnames):
    if line_handles[i] is not None:
        lines.append(line_handles[i])
        labels.append(legend_labels[i])

if square and not only_1D:
    fig.legend(lines, labels, bbox_to_anchor=(0.75, 0.72, 0.2, 0.2))
elif only_1D:
    fig.legend(lines, labels, loc="center left")
else:
    fig.legend(lines, labels, bbox_to_anchor=(0.7, 0.72, 0.2, 0.2))
if square and not only_1D:
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
elif only_1D:
    plt.subplots_adjust(left=0.13, bottom=0.25, right=0.95, top=0.95)
else:
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.97)

plt.subplots_adjust(wspace=dist, hspace=dist)
PATH = output_path
for i, name in enumerate(JOBnames):
    name = name.split("/")[-1]
    PATH += name
    if i != len(JOBnames) - 1:
        PATH += "-"
PATH += ".pdf"
plt.savefig(PATH)
plt.show()
