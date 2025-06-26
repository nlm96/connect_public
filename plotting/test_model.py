"""
This plotting script can be used to test a models performance on the
stored test data. The syntax is 'python test_model.py <path to model>',
and up to three models can be given. Only the first model will be used
to produce cmb spectra, but all models given will be included in the
error plot. The error plot will be saved within the first model specified
as '<path to model>/plots/error.pdf'.

Author: Andreas Nygaard (2022)

"""

"""
New features:
The script now supports a combined plot of CMB power spectra and errors.
It supports custom labels for the models and an specified output directory for the plots.

Usage:
python test_model.py path_to_model1 path_to_model2 path_to_model3 --labels label1 label2 label3 --output path_to_output_directory

#Important Note: The labels must be provided in the same order as the models are provided.
#If no labels are provided, the script will use the last part of the path as the label for each model.
#If no output directory is provided, the script will save the plots in the first model's "plots" directory.
#Also make sure the the keys appear in the above order with --labels before --output.

"""

import os
import sys
import warnings
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib

import PlanckLogLinearScale

# index of test data to use for cmb spectra
n=150

# color of axes and all text in the error plot
color_of_axis_and_text = 'k'

# use latex?
latex = True

fontsize = 11/1.2*1.5

if latex:
    latex_preamble = r' \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{mathtools}'

    #latex_preamble=r'\usepackage{siunitx} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{mathtools} \usepackage{bm} \usepackage{mathrsfs} \parindent = 0pt'
    matplotlib.rcParams.update({
        'text.usetex'        : True,
        'font.family'        : 'serif',
        'font.serif'         : 'cmr10',
        'font.size'          : fontsize,
        'mathtext.fontset'   : 'cm',
        'text.latex.preamble': latex_preamble,
    })



"""
model_paths = []
model_names = []
for arg in sys.argv[1:]:
    model_paths.append(arg)
    model_names.append(arg.split('/')[-1])





model_paths = []
model_names = []

# Check if custom labels are provided with "--labels" flag
if "--labels" in sys.argv:
    labels_index = sys.argv.index("--labels")
    model_paths = sys.argv[1:labels_index]  # Paths before "--labels"
    model_names = sys.argv[labels_index + 1:]  # Labels after "--labels"
    if len(model_paths) != len(model_names):
        raise ValueError("Number of provided model paths and labels must match.")
else:
    for arg in sys.argv[1:]:
        model_paths.append(arg)
        model_names.append(arg.split('/')[-1])  # Default: last part of path

# Check if an output directory is specified with "--output"
if "--output" in sys.argv:
    output_index = sys.argv.index("--output")
    if output_index + 1 < len(sys.argv):
        output_dir = sys.argv[output_index + 1]
        del sys.argv[output_index:output_index + 2]  # Remove --output and the path from sys.argv
    else:
        raise ValueError("No output directory provided after --output flag.")
else:
    output_dir = os.path.join(model_paths[0], "plots")  # Default to the first model's "plots" directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
"""


import os
import sys

model_paths = []
model_names = []
output_dir = None  # Default to None until parsed

# Process command-line arguments manually to handle arbitrary order
args = sys.argv[1:]  # Ignore the script name

if "--labels" in args:
    labels_index = args.index("--labels")
    model_paths = args[:labels_index]  # Paths before "--labels"
    model_names = args[labels_index + 1:]  # Labels after "--labels"
    
    # Ensure the number of labels matches the number of models
    if "--output" in model_names:  # Prevent incorrect parsing
        output_index = model_names.index("--output")
        output_dir = model_names[output_index + 1]
        model_names = model_names[:output_index]  # Remove --output from labels

    if len(model_paths) != len(model_names):
        raise ValueError("Number of provided model paths and labels must match.")

else:
    # No --labels flag, assume all are model paths
    model_paths = args
    model_names = [path.split('/')[-1] for path in model_paths]  # Default: last part of path

# Check if an output directory is specified separately
if "--output" in args:
    output_index = args.index("--output")
    if output_index + 1 < len(args):
        output_dir = args[output_index + 1]
    else:
        raise ValueError("No output directory provided after --output flag.")

# Default output directory (if not provided)
if output_dir is None:
    output_dir = os.path.join(model_paths[0], "plots")  # Default to first model's "plots" directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)



percentiles = [0.682,0.954]
alpha_list  = [0.2,    0.1,   0.05]



if len(model_names) == 3:
    #c_list      = ['green','red','blue']
        c_list      = ['blue','red','green'][::-1] # reverse the order
elif len(model_names) == 2:
    #c_list      = ['red','blue','green']
        c_list      = ['blue','red','green']
else:
    c_list      = ['blue','red','green']
fc_array    = [['cyan',   'blue', 'navy'],
               ['orange', 'red',  'crimson'],
               ['lightgreen','forestgreen','darkgreen']]

def rms(x):
    return np.sqrt(np.mean(x**2))

################################################################

name = sys.argv[1]
model = tf.keras.models.load_model(name, compile=False)
with open(name+'/test_data.pkl', 'rb') as f:
    test_data = pkl.load(f)

import os

# Define the plots directory
plots_dir = os.path.join(name, "plots")
# Ensure the 'plots' directory exists (creates it if missing)
os.makedirs(plots_dir, exist_ok=True)



try:
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])
except:
    test_data = tuple(zip(*test_data))
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])

try:    
    with open(name+'/output_info.pkl', 'rb') as f:
        output_info = pkl.load(f)
    pickle_file = True
    warnings.warn("You are using CONNECT models from an old version (before v23.6.0). Support for this is deprecated and will be removed in a later update.")
    
except:
    pickle_file = False
    output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

if pickle_file:
    try:
        if output_info['normalize']['method'] == 'standardization':
            normalize = 'standardization'
        elif output_info['normalize']['method'] == 'log':
            normalize = 'log'
        elif output_info['normalize']['method'] == 'min-max':
            normalize = 'min-max'
        elif output_info['normalize']['method'] == 'factor':
            normalize = 'factor'
        else:
            normalize = 'factor'
    except:
        normalize = 'standardization'

output_predict  = model.predict(model_params, verbose=0)


    
if pickle_file and normalize == 'standardization':
    mean = output_info['normalize']['mean']
    var  = output_info['normalize']['variance']
    output_predict = output_predict * np.sqrt(var) + mean
    output_data = output_data * np.sqrt(var) + mean
elif pickle_file and normalize == 'min-max':
    x_min = np.array(output_info['normalize']['x_min'])
    x_max = np.array(output_info['normalize']['x_max'])
    output_predict = output_predict * (x_max - x_min) + x_min
    output_data = output_data * (x_max - x_min) + x_min




out_predict = {}
out_data    = {}
for output in output_info['output_Cl']:
    lim0 = output_info['interval']['Cl'][output][0]
    lim1 = output_info['interval']['Cl'][output][1]
    out_data[output]    = output_data[n][lim0:lim1]
    out_predict[output] = output_predict[n][lim0:lim1]
    if pickle_file and normalize == 'log':
        for offset in list(reversed(output_info['normalize']['Cl'][output])):
            out_predict[output]=np.exp(out_predict[output]) - offset
            out_data[output]=np.exp(out_data[output]) - offset


if 'output_derived' in output_info.keys():
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            idx = output_info['interval']['derived'][output]
            out_data[output]    = output_data[n][idx]
            out_predict[output] = output_predict[n][idx]
            if pickle_file and normalize == 'log':
                for offset in list(reversed(output_info['normalize']['derived']['100*theta_s'])):
                    out_predict[output]=np.exp(out_predict[output]) - offset
                    out_data[output]=np.exp(out_data[output]) - offset

for output in output_info['output_Cl']:
    if pickle_file and normalize == 'factor':
        normalize_factor = output_info['normalize']['Cl'][output]
    plt.figure(figsize=(10,7))
    ell        = output_info['ell']
    ll         = np.linspace(2,max(ell)+7,int(max(ell)-1+7))
    Cl_predict = out_predict[output]
    Cl_data    = out_data[output]
    Cl_pre_sp  = CubicSpline(ell,Cl_predict, bc_type = 'natural', extrapolate=True)
    Cl_dat_sp  = CubicSpline(ell,Cl_data,  bc_type = 'natural', extrapolate=True)
    
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_dat_sp(ll)/normalize_factor,'k-',lw=3 )
    else:
        plt.plot(ll, Cl_dat_sp(ll),'k-',lw=3)
    
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_pre_sp(ll)/normalize_factor,'r--',lw=3)
    else:
        plt.plot(ll, Cl_pre_sp(ll),'--', color=c_list[0],lw=3)
        
    
        
    plt.legend([r'CLASS', model_names[0]])
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_{\ell} = C_{\ell}\times\ell(\ell+1)/2\pi$')

    plt.title(output)
    # Save the CMB power spectra plot
    #plt.savefig(name + f'/plots/{output}_spectra.pdf')
    plt.savefig(os.path.join(output_dir, f"{output}_spectra.pdf"), bbox_inches='tight')
    plt.close()
    #print(f"CMB power spectra plot saved to: {name}/plots/{output}_spectra.pdf")
    print(f"CMB power spectra plot saved to: {os.path.join(output_dir, f'{output}_spectra.pdf')}")
    
    


if 'output_derived' in output_info.keys(): 
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            if pickle_file and normalize == 'factor':
                normalize_factor = output_info['normalize']['derived'][output]
            print(output)
            if pickle_file and normalize == 'factor':
                print('CLASS:',out_data[output]/normalize_factor)
            else:
                print('CLASS:',out_data[output])
            if pickle_file and normalize == 'factor':
                print('CONNECT:',out_predict[output]/normalize_factor)
            else:
                print('CONNECT:',out_predict[output])






    
l = np.linspace(2,2500,2499)


def get_error(path,spectrum):
    model = tf.keras.models.load_model(path, compile=False)

    with open(path + '/test_data.pkl', 'rb') as f:
        test_data = pkl.load(f)
    try:
        with open(path + '/output_info.pkl', 'rb') as f:
            output_info = pkl.load(f)
        pickle_file = True
    except:
        pickle_file = False
        output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

    ell = output_info['ell']
    if pickle_file:
        try:
            if output_info['normalize']['method'] == 'standardization':
                normalize = 'standardization'
            elif output_info['normalize']['method'] == 'log':
                normalize = 'log'
            elif output_info['normalize']['method'] == 'min-max':
                normalize = 'min-max'
            elif output_info['normalize']['method'] == 'factor':
                normalize = 'factor'
            else:
                normalize = 'factor'
        except:
            normalize = 'standardization'
            
    try:
        model_params = test_data[0]
        Cls_data     = test_data[1]
    except:
        test_data = tuple(zip(*test_data))
        model_params = np.array(test_data[0])
        Cls_data     = np.array(test_data[1])


    v = tf.constant(model_params)
    Cls_predict = model(v).numpy()

    if pickle_file and normalize == 'standardization':
        mean = output_info['normalize']['mean']
        var  = output_info['normalize']['variance']
        Cls_predict = Cls_predict * np.sqrt(var) + mean
        Cls_data = Cls_data * np.sqrt(var) + mean
    elif pickle_file and normalize == 'min-max':
        x_min = np.array(output_info['normalize']['x_min'])
        x_max = np.array(output_info['normalize']['x_max'])
        Cls_predict = Cls_predict * (x_max - x_min) + x_min
        Cls_data = Cls_data * (x_max - x_min) + x_min
    
    lim0 = output_info['interval']['Cl'][spectrum][0]
    lim1 = output_info['interval']['Cl'][spectrum][1]

    errors = []
    for j, (cls_d, cls_p) in enumerate(zip(Cls_data, Cls_predict)):
        if pickle_file and normalize == 'factor':
            err = ((np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/output_info['normalize']['Cl'][spectrum])/rms(np.array(cls_d[lim0:lim1]))
        elif pickle_file and normalize == 'log':
            clsp = cls_p[lim0:lim1]
            clsd = cls_d[lim0:lim1]
            for offset in list(reversed(output_info['normalize']['Cl'][spectrum])):
                clsp=np.exp(clsp) - offset
                clsd=np.exp(clsd) - offset
            err = (np.array(clsd)-np.array(clsp))/rms(np.array(clsd))
        else:
            err = (np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/rms(np.array(cls_d[lim0:lim1]))
        errors.append(abs(err))

    errors = np.array(errors).T
    return errors, ell





height = 2.2
width = 6.0874173228

change=200

PlanckLogLinearScale.new_change(change)

fig, axs = plt.subplots(2,len(output_info['output_Cl']),
                        figsize=(width*1.5, height*1.5), #(width, height) 
                        gridspec_kw={'height_ratios':[1,4]})
fig.subplots_adjust(wspace=0)
for i in range(len(output_info['output_Cl'])):
    axs[0,i].axis('off')



for k, spectrum in enumerate(output_info['output_Cl']):
    y_max = 0
    y_min = 0
    for i, path in enumerate(model_paths): 
        model_name = model_names[i]
        errors, l_red =  get_error(path, spectrum)

        max_error=[]
        err_lower_array = []
        err_upper_array = []
        for errs in errors:
            err_lower_list = []
            err_upper_list = []
            for p in percentiles:
                err_upper_list.append(np.percentile(errs, 100*p))

            err_upper_array.append(err_upper_list)
            max_error.append(max(errs))
        if max(np.array(err_upper_array).flatten()) > y_max:
            y_max = max(np.array(err_upper_array).flatten())
            
        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(l_red,np.array(err_upper_array).T[j])
            
            axs[1,k].fill(np.array([l[0]]+list(l[:change-1])+[l[change-2]]),
                     np.array([0]+list(abs(err_u(l[:change-1])))+[0]),
                     fc=c_list[i],
                     alpha=alpha_list[j],
                     zorder=i)
            axs[1,k].fill(np.array([l[change-2]]+list(l[change-2:])+[l[-1]]),
                     np.array([0]+list(abs(err_u(l[change-2:])))+[0]),
                     fc=c_list[i],
                     alpha=alpha_list[j],
                     zorder=i)

            axs[1,k].axvline(x=change, linestyle="-", color="lightgrey", zorder=-3)
            if j==len(percentiles)-1 and k==0:
                axs[1,k].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1.25, zorder=i, alpha=list(reversed(percentiles))[j])
            else:
                axs[1,k].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1.25, zorder=i, alpha=list(reversed(percentiles))[j])

    if k==0:
        custom_lines = []
        for i in range(len(model_names)):
            # i=0 => c_list[0] for model 0, i=1 => c_list[1], etc.
            line = matplotlib.lines.Line2D([], [], c=c_list[i], lw=2)
            custom_lines.append(line)

        axs[1,k].legend(custom_lines,model_names,bbox_to_anchor=(2.64+0.3,1.4),ncol=len(model_names),fontsize=fontsize/1.5)
        #Center the legend to the full figure, not just a subplot axis
        #fig.legend(custom_lines,model_names,bbox_to_anchor=(0.5,1.15),ncol=len(model_names),fontsize=fontsize/1.5)

    axs[1,k].set_yscale('log')
    axs[1,k].set_xscale('planck')

    axs[1,k].set_xlim([2,2500])
    if k==1:
        axs[1,k].set_xlabel(r'$\ell$')
    if k == 0:
        if latex:
            axs[1,k].set_ylabel(r'$\dfrac{\left\vert \mathcal{D}_{\ell}^{\textsc{CONNECT}}-\mathcal{D}_{\ell}^{\textsc{CLASS}}\right\vert}{{\rm rms}\left(\mathcal{D}_{\ell}^{\textsc{CLASS}}\right)}$', fontsize=fontsize/1.5)
        else:
            axs[1,k].set_ylabel(r'$|D_{\ell}^{\rm connect}-D_{\ell}^{\rm class}| /{\rm rms}(D_{\ell}^{\rm class})$', fontsize=fontsize/1.5)
    else:
        axs[1,k].yaxis.set_ticklabels([])
    axs[1,k].set_ylim([5e-6,5e-1])
    axs[1,k].set_title(f'{spectrum.upper()}', color=color_of_axis_and_text)
    axs[1,k].tick_params(axis='both', which='both', direction='in', labelsize=fontsize/1.5)
    xticks = [1e+1, 1e+2, 1e+3, 2e+3]
    xticks_minor = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400,500,600,700,800,900,1100,1200,1300,1400,1500,1600,1700,1800,1900,2100,2200,2300,2400,2500] 
    yticklabels=[r'$10^{-1}$',r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$',r'$10^{-5}$']
    xticklabels=[r'$10^{1}$',r'$10^{2}$','1000','2000']
    axs[1,k].set_xticks(xticks)
    axs[1,k].set_xticks(xticks_minor, minor=True)
    axs[1,k].xaxis.set_ticklabels(xticklabels, color=color_of_axis_and_text)
    axs[1,k].set_yticks([1e-1,1e-2,1e-3,1e-4,1e-5])
    if k==0:
        axs[1,k].yaxis.set_ticklabels(yticklabels, color=color_of_axis_and_text)

factor=1.09
factor_y=0.91
offset=axs[1,0].get_position().x0*(factor - 1) - 0.03
offset_y=axs[0,0].get_position().y1*(factor_y - 1) - 0.06
for ax in axs.flatten():
    box=ax.get_position()
    box.x0 *= factor
    box.x0 += -offset-0.005
    box.x1 *= factor
    box.x1 += -offset-0.005
    box.y0 *= factor_y
    box.y0 += -offset_y + 0.0
    box.y1 *= factor_y
    box.y1 += -offset_y + 0.0
    ax.set_position(box)
    

    ax.xaxis.label.set_color(color_of_axis_and_text)           #setting up X-axis label color
    ax.yaxis.label.set_color(color_of_axis_and_text)           #setting up Y-axis label color

    ax.set_facecolor('w')

#fig.patch.set_alpha(0)
    
if not os.path.isdir(model_paths[0]+'/plots'):
    os.mkdir(model_paths[0]+'/plots')

#plt.savefig(model_paths[0]+f'/plots/error.pdf', facecolor=fig.get_facecolor())
plt.savefig(os.path.join(output_dir, "error.pdf"), facecolor=fig.get_facecolor(), bbox_inches='tight')

plt.close()
#print("Error plot saved to:", os.path.join(model_paths[0], "plots/error.pdf"))
print("Error plot saved to:", os.path.join(output_dir, "error.pdf"))


# Set up a multipanel plot: 2 rows, 3 columns (top row for spectra, bottom row for errors)
# Set up a multipanel plot: 2 rows, 3 columns (top row for spectra, bottom row for errors)
# Set up a multipanel plot: 2 rows, N columns (top row for spectra, bottom row for errors).
fig, axs = plt.subplots(2, len(output_info['output_Cl']),
                        figsize=(15, 8), #(width, height)
                        gridspec_kw={'height_ratios': [1.5,1]})
fig.subplots_adjust(wspace=0, hspace=0)


# Loop over each spectrum type, e.g. TT, TE, EE
for k, output in enumerate(output_info['output_Cl']):

    if pickle_file and normalize == 'factor':
        normalize_factor = output_info['normalize']['Cl'][output]

    ell = output_info['ell']
    ll = np.linspace(2, max(ell) + 7, int(max(ell) - 1 + 7))

    # -- Top row: Plot the first model's power spectra (CLASS vs CONNECT).
    Cl_predict = out_predict[output]
    Cl_data    = out_data[output]

    # Interpolation using cubic splines
    Cl_pre_sp = CubicSpline(ell, Cl_predict, bc_type='natural', extrapolate=True)
    Cl_dat_sp = CubicSpline(ell, Cl_data,    bc_type='natural', extrapolate=True)

    if pickle_file and normalize == 'factor':
        axs[0, k].plot(ll, Cl_dat_sp(ll)/normalize_factor, 'k-', lw=3, label='CLASS')
        axs[0, k].plot(ll, Cl_pre_sp(ll)/normalize_factor, 'r--', lw=3, label=model_names[0])
    else:
        axs[0, k].plot(ll, Cl_dat_sp(ll), 'k-', lw=3, label='CLASS')
        axs[0, k].plot(ll, Cl_pre_sp(ll), '--', color=c_list[0], lw=3, label=model_names[0])

    axs[0, k].axvline(x=change, linestyle="-", color="lightgrey", zorder=-3)
    axs[0, k].set_xscale('planck')
    axs[0, k].set_xlim([2, 2500])
    

    
    # Remove set_title; we add a small text label inside the top subplot instead:
    # This positions the text near the top-right corner of the axes:
    axs[0, k].text(
        0.85, 0.85,
        output.upper(),
        transform=axs[0, k].transAxes,
        fontsize=27,
        ha='center',
        va='center',
        color=color_of_axis_and_text
    )

    # Only put a Y-axis label on the leftmost top panel:
    if k == 0:
        axs[0, k].set_ylabel(r'$D_{\ell} = C_{\ell} \times \ell(\ell+1)/2\pi$')
    # Optionally only put the X-axis label on the bottom row, so for the top row you might do:
    # axs[0, k].set_xlabel('')
    # but let's keep an x-label on the top if you prefer.

    # -- Bottom row: Plot errors from all models
    for i, path in enumerate(model_paths):
        model_name = model_names[i]
        errors, l_red = get_error(path, output)  # Must pass output, not spectrum

        # Compute percentiles
        err_upper_array = []
        for errs in errors:
            up_list = [np.percentile(errs, 100*p) for p in percentiles]
            err_upper_array.append(up_list)
        err_upper_array = np.array(err_upper_array).T  # shape: [len(percentiles), n_ell]

        # Fill under + boundary lines
        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(l_red, err_upper_array[j],
                                bc_type='natural', extrapolate=True)

            # fill for log-lin
            axs[1, k].fill(
                np.array([l[0]] + list(l[:change-1]) + [l[change-2]]),
                np.array([0] + list(abs(err_u(l[:change-1]))) + [0]),
                fc=c_list[i],
                alpha=alpha_list[j],
                zorder=i
            )
            axs[1, k].fill(
                np.array([l[change-2]] + list(l[change-2:]) + [l[-1]]),
                np.array([0] + list(abs(err_u(l[change-2:]))) + [0]),
                fc=c_list[i],
                alpha=alpha_list[j],
                zorder=i
            )
            axs[1, k].axvline(x=change, linestyle="-", color="lightgrey", zorder=-3)

            # Plot boundary line
            if j == len(percentiles) - 1 and k == 0:
                axs[1, k].plot(l, abs(err_u(l)),
                               c=c_list[i], lw=2, zorder=i,
                               alpha=list(reversed(percentiles))[j])
            else:
                axs[1, k].plot(l, abs(err_u(l)),
                               c=c_list[i], lw=2, zorder=i,
                               alpha=list(reversed(percentiles))[j])

    # Some cosmetics for bottom row
    axs[1, k].set_yscale('log')
    axs[1, k].set_xscale('planck')
    axs[1, k].set_xlim([2, 2500])
    axs[1, k].set_ylim([5e-6, 5e-1])
    

    # Add x/y labels
    if k == 0:
        if latex:
            axs[1, k].set_ylabel(
                r'$\dfrac{|D_{\ell}^{\textsc{CONNECT}} - D_{\ell}^{\textsc{CLASS}}|}'
                r'{\mathrm{rms}(D_{\ell}^{\textsc{CLASS}})}$'
            )
        else:
            axs[1, k].set_ylabel(
                r'$|D_{\ell}^{\mathrm{connect}} - D_{\ell}^{\mathrm{class}}|'
                r'/\mathrm{rms}(D_{\ell}^{\mathrm{class}})$'
            )
    else:
        # Hide y‐axis labels on other bottom subplots
        axs[1, k].yaxis.set_ticklabels([])

    # You can put the x-axis label only on the bottom row
    axs[1, k].set_xlabel(r'$\ell$')
    
    # We do NOT set a title for the bottom row, so remove:
    # axs[1, k].set_title(...)
    
    # Ticks
    axs[1, k].tick_params(axis='both', which='both', direction='in')
    xticks       = [1e+1, 1e+2, 1e+3, 2e+3]
    xticks_minor = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,
                    200,300,400,500,600,700,800,900,1100,1200,
                    1300,1400,1500,1600,1700,1800,1900,2100,2200,
                    2300,2400,2500]
    yticklabels  = [r'$10^{-1}$',r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$']
    xticklabels  = [r'$10^{1}$', r'$10^{2}$', '1000', '2000']
    axs[1, k].set_xticks(xticks)
    axs[1, k].set_xticks(xticks_minor, minor=True)
    axs[1, k].xaxis.set_ticklabels(xticklabels, color=color_of_axis_and_text)
    axs[1, k].set_yticks([1e-1,1e-2,1e-3,1e-4,1e-5])
    if k == 0:
        axs[1, k].yaxis.set_ticklabels(yticklabels, color=color_of_axis_and_text)

# Build a single super‐legend for the models + colors
    if k==0:
        custom_lines = []
        for i in range(len(model_names)):
            # i=0 => c_list[0] for model 0, i=1 => c_list[1], etc.
            line = matplotlib.lines.Line2D([], [], c=c_list[i], lw=2)
            custom_lines.append(line)

left = axs[1, 0].get_position().x0
right = axs[1, -1].get_position().x1
center_x = (left + right) / 2



# Place the super legend above the figure
fig.legend(custom_lines, model_names,
           loc='upper center',
           bbox_to_anchor=(center_x+0.07, 1.03),
           ncol=len(model_names),
           fontsize=15)

# Adjust positions, etc.
factor    = 1.09
factor_y  = 0.91
offset    = axs[1,0].get_position().x0*(factor - 1) - 0.03
offset_y  = axs[0,0].get_position().y1*(factor_y - 1) - 0.06

for ax in axs.flatten():
    box=ax.get_position()
    box.x0 *= factor
    box.x0 += -offset - 0.005
    box.x1 *= factor
    box.x1 += -offset - 0.005
    box.y0 *= factor_y
    box.y0 += -offset_y
    box.y1 *= factor_y
    box.y1 += -offset_y
    ax.set_position(box)
    ax.xaxis.label.set_color(color_of_axis_and_text)
    ax.yaxis.label.set_color(color_of_axis_and_text)
    ax.set_facecolor('w')

if not os.path.isdir(model_paths[0] + '/plots'):
    os.mkdir(model_paths[0] + '/plots')

#plt.savefig(model_paths[0] + '/plots/combined_spectra_and_errors.pdf', bbox_inches='tight')

plt.savefig(os.path.join(output_dir, "combined_spectra_and_errors.pdf"), bbox_inches='tight')
plt.close()

#print("Combined spectra and errors plot saved to:", os.path.join(model_paths[0], "plots/combined_spectra_and_errors.pdf"))
print("Combined spectra and errors plot saved to:", os.path.join(output_dir, "combined_spectra_and_errors.pdf"))


# Validation Loss Plot for All Models
plt.figure(figsize=(8, 4))  # (width, height)

has_valid_loss = False  # Track if any valid loss file was found

for i, path in enumerate(model_paths):
    val_loss_file = os.path.join(path, 'val_loss.pkl')

    if os.path.exists(val_loss_file):
        with open(val_loss_file, 'rb') as f:
            val_loss = pkl.load(f)

        epochs = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, label=model_names[i], color=c_list[i])  # Use corresponding color

        has_valid_loss = True

# Only adjust y-limits if at least one valid file was found
if has_valid_loss:
    val_min = min(min(pkl.load(open(os.path.join(path, 'val_loss.pkl'), 'rb'))) for path in model_paths if os.path.exists(os.path.join(path, 'val_loss.pkl')))
    val_max = max(max(pkl.load(open(os.path.join(path, 'val_loss.pkl'), 'rb'))) for path in model_paths if os.path.exists(os.path.join(path, 'val_loss.pkl')))

    # Find the nearest lower power of 10
    y_min_tick = 10 ** np.floor(np.log10(val_min))

    # Set y-limits
    y_min = y_min_tick
    y_max = val_max * 1.1  # Slightly above the max value

    plt.ylim([y_min, y_max])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # Log-scale for better visualization
    plt.title("Validation Loss History")
    plt.legend(fontsize=fontsize/1.5)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the validation loss plot
#    plt.savefig(os.path.join(plots_dir, "val_loss.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "val_loss.pdf"), bbox_inches='tight')
    plt.close()
    #print("Validation loss plot saved to:", os.path.join(plots_dir, "val_loss.pdf"))
    print("Validation loss plot saved to:", os.path.join(output_dir, "val_loss.pdf"))

else:
    print("Warning: No validation loss files found for any model. Skipping validation loss plot.")
    


def rms(x):
    return np.sqrt(np.mean(x**2))



import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import tensorflow as tf

# Just for completeness. You presumably already have these.
#c_list = ['red','blue','orange','green','purple','brown']  # etc.
#alpha_list = [0.15, 0.3, 0.45, 0.6]  # for filling different percentiles
#percentiles = [0.68, 0.95]          # example
#model_paths = [...]                 # your list of model directories
#model_names = [...]                 # matching names
#fontsize = 14                       # example
#output_dir = "./plots"              # where we save figures
#os.makedirs(output_dir, exist_ok=True)

#############################################################################
# 1) Extract slices for matter power spectra & un‐normalize if needed
#############################################################################
out_data_pk    = {}
out_predict_pk = {}

# Suppose you already have the following variables defined:
#    output_info     = ...  # from your code
#    output_data     = ...  # from your code
#    output_predict  = ...  # from your code
#    n, pickle_file, normalize etc. are all set as in your code

if "output_Pk" in output_info:
    # E.g. ["pk_cb", "pk"] or similar
    pk_outputs = output_info["output_Pk"]
    k_grid     = np.array(output_info["k_grid"])  # The k grid

    for pk_output in pk_outputs:
        # Each pk_output can have multiple redshifts, stored in interval dict
        pk_z_dict = output_info["interval"]["Pk"][pk_output]
        # Loop over all redshifts for this pk_output
        for z_key, (lim0, lim1) in pk_z_dict.items():
            # Slices for this pk_output, z_key
            data_slice = output_data[n][lim0:lim1]    # "CLASS" reference
            pred_slice = output_predict[n][lim0:lim1] # "CONNECT" prediction

            # If the model used "log" or "factor" for Pk, apply per-output un‐normalization.
            if pickle_file and normalize == "log":
                offsets = output_info["normalize"]["Pk"].get(pk_output, [])
                for offset in reversed(offsets):
                    pred_slice = np.exp(pred_slice) - offset
                    data_slice = np.exp(data_slice) - offset

            elif pickle_file and normalize == "factor":
                factor = output_info["normalize"]["Pk"].get(pk_output, 1.0)
                pred_slice = pred_slice / factor
                data_slice = data_slice / factor

            # Store final un‐normalized slices
            store_key = f"{pk_output}_z{z_key}"
            out_data_pk[store_key]    = data_slice
            out_predict_pk[store_key] = pred_slice


#############################################################################
# 2) (Modified) Plot *all* matter power spectra (CLASS vs CONNECT) in a single
#    figure, one column per redshift, FOR EACH pk_output.
#############################################################################

if "output_Pk" in output_info:
    pk_outputs = output_info["output_Pk"]
    k_grid     = np.array(output_info["k_grid"])

    for pk_output in pk_outputs:

        pk_z_dict = output_info["interval"]["Pk"][pk_output]
        # Sort z_keys by numeric value if you want them in ascending order
        z_keys = sorted(pk_z_dict.keys(), key=lambda x: float(x))
        ncols  = len(z_keys)

        # Create a multi-column figure for *spectra*
        fig_spectra, axs_spectra = plt.subplots(
            1, ncols, figsize=(5.0*ncols, 4.5),
            sharey=True,  # or True if you want the same y‐range
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )

        # If there's only one z‐value, axs_spectra will not be a list, so handle that:
        if ncols == 1:
            axs_spectra = [axs_spectra]

        # Generate a dense k grid for splines
        k_dense = np.logspace(np.log10(k_grid.min()), np.log10(k_grid.max()), 1000)

        for i, z_key in enumerate(z_keys):
            ax = axs_spectra[i]
            store_key = f"{pk_output}_z{z_key}"

            # Retrieve final un‐normalized slices
            pk_data_slice = out_data_pk[store_key]
            pk_pred_slice = out_predict_pk[store_key]

            # Build splines
            cs_data = CubicSpline(k_grid, pk_data_slice, bc_type='natural') # or 'natural'
            cs_pred = CubicSpline(k_grid, pk_pred_slice, bc_type='natural')

            # Plot
            ax.loglog(k_dense, cs_data(k_dense), 'k-', label='CLASS', lw=3)
            ax.loglog(k_dense, cs_pred(k_dense), '--', color=c_list[0],
                      label=model_names[0], lw=3, dashes=(4, 3))
            
            #Write the type of power spectrum in the right corner of the plot
            if pk_output == 'pk_cb':
                ax.text(0.95, 0.95, '$P_{cb}(k)$', transform=ax.transAxes, fontsize=fontsize*1.2,
                        verticalalignment='top', horizontalalignment='right')
            elif pk_output == 'pk':
                ax.text(0.95, 0.95, '$P(k)$', transform=ax.transAxes, fontsize=fontsize*1.2,
                        verticalalignment='top', horizontalalignment='right')
            else:
                ax.text(0.95, 0.95, f'{pk_output}', transform=ax.transAxes, fontsize=fontsize*1.2,
                        verticalalignment='top', horizontalalignment='right')
            
            # Add legend
            if i == 0:
                ax.legend(fontsize=fontsize/1.2)
                

            # Cosmetics
            ax.set_xlabel(r"$k \,[h/\mathrm{Mpc}]$")
            if i == 0:
                ax.set_ylabel(r"$P(k)\,[ (\mathrm{Mpc}/h)^3 ]$")

            ax.set_title(fr"$z={z_key}$")
            
            # Set x‐limits
            ax.set_xlim([k_grid.min(), k_grid.max()])
            # Add minor ticks on both axes
            ax.minorticks_on()
            

        # Make a single legend for the entire figure
        handles, labels = axs_spectra[0].get_legend_handles_labels()
        fig_spectra.tight_layout()

        outname = f"Pk_{pk_output}_spectra_zall.pdf"
        fig_spectra.savefig(os.path.join(output_dir, outname), bbox_inches='tight')
        plt.close(fig_spectra)
        print(f"Matter power spectra multi‐z plot saved: {outname}")


#############################################################################
# 3) Define a helper function for the error envelopes across the entire test set
#############################################################################

def rms(x):
    return np.sqrt(np.mean(x**2))

def get_error_pk(path, pk_output, z_key):
    """
    Returns array of shape [n_k, n_test_samples] giving
    abs( (data - pred) / rms(data) ) for each test sample
    """
    # Load model + test data
    mod = tf.keras.models.load_model(path, compile=False)
    with open(os.path.join(path,'test_data.pkl'),'rb') as f:
        tdata = pkl.load(f)
    model_params_all = np.array(tdata[0])
    out_all          = np.array(tdata[1])

    # Load output_info
    try:
        with open(os.path.join(path,'output_info.pkl'),'rb') as f:
            out_inf = pkl.load(f)
        pfile = True
    except:
        pfile = False
        # For older models, get_raw_info
        out_inf = eval(mod.get_raw_info().numpy().decode('utf-8'))

    # Which normalization?
    try:
        norm_method = out_inf['normalize']['method']
    except:
        norm_method = 'standardisation'

    # Predict
    pred_all = mod.predict(model_params_all, verbose=0)

    # Undo global normalization if needed
    if pfile:
        if norm_method == 'standardisation':
            mn  = out_inf['normalize']['mean']
            var = out_inf['normalize']['variance']
            out_all  = out_all  * np.sqrt(var) + mn
            pred_all = pred_all * np.sqrt(var) + mn
        elif norm_method == 'min-max':
            x_min = np.array(out_inf['normalize']['x_min'])
            x_max = np.array(out_inf['normalize']['x_max'])
            out_all  = out_all  * (x_max - x_min) + x_min
            pred_all = pred_all * (x_max - x_min) + x_min

    # Identify slice [lim0:lim1]
    lim0, lim1 = out_inf['interval']['Pk'][pk_output][str(z_key)]

    # For each test sample, slice & apply any log/factor offset for pk_output
    errors_list = []
    for data_row, pred_row in zip(out_all, pred_all):
        d_slice = data_row[lim0:lim1]
        p_slice = pred_row[lim0:lim1]

        if pfile and norm_method == 'log':
            offsets = out_inf['normalize']['Pk'].get(pk_output, [])
            for offset in reversed(offsets):
                p_slice = np.exp(p_slice) - offset
                d_slice = np.exp(d_slice) - offset

        elif pfile and norm_method == 'factor':
            factor = out_inf['normalize']['Pk'].get(pk_output, 1.0)
            p_slice = p_slice / factor
            d_slice = d_slice / factor

        # Compute error for this sample
        err = np.abs((d_slice - p_slice) / rms(d_slice))
        errors_list.append(err)

    # Convert to [n_k, n_test_samples]
    errors_arr = np.array(errors_list).T
    return errors_arr


#############################################################################
# 4) (Modified) Plot *all* error envelopes in a single figure, one column
#    per redshift, FOR EACH pk_output.
#############################################################################

if "output_Pk" in output_info:
    pk_outputs = output_info["output_Pk"]
    k_grid     = np.array(output_info["k_grid"])

    for pk_output in pk_outputs:

        pk_z_dict = output_info["interval"]["Pk"][pk_output]
        z_keys = sorted(pk_z_dict.keys(), key=lambda x: float(x))
        ncols  = len(z_keys)

        # Create a multi‐column figure for *errors*
        fig_error, axs_error = plt.subplots(
            1, ncols, figsize=(5.0*ncols, 3.5),
            sharey=True, 
            gridspec_kw={'hspace': 0, 'wspace': 0}
        )
        if ncols == 1:
            axs_error = [axs_error]

        # A dense k-axis in logspace
        k_dense = np.logspace(np.log10(k_grid.min()), np.log10(k_grid.max()), 1000)

        for i, z_key in enumerate(z_keys):
            ax = axs_error[i]

            # For each model, compute error array => percentiles => fill
            for m_i, mp in enumerate(model_paths):
                errors_arr = get_error_pk(mp, pk_output, z_key)  # shape [n_k, n_test_samples]

                # Take percentiles
                percentile_arrays = []
                for p_ in sorted(percentiles):
                    pval = np.percentile(errors_arr, 100*p_, axis=1)  # shape [n_k]
                    percentile_arrays.append(pval)
                percentile_arrays = np.array(percentile_arrays)  # shape [len(percentiles), n_k]

                # Fill from 0 up to the percentile, in descending order
                for j, p_ in reversed(list(enumerate(sorted(percentiles)))):
                    spl       = CubicSpline(k_grid, percentile_arrays[j], bc_type='natural')
                    envelope  = spl(k_dense)
                    ax.fill_between(k_dense, 0, envelope,
                                    color=c_list[m_i],
                                    alpha=alpha_list[j],
                                    zorder=m_i)
                    ax.plot(k_dense, envelope,
                            color=c_list[m_i], lw=1.5, zorder=m_i)

            # Cosmetics
            #Set x-range to match the dense k-grid: [k_min, k_max]. Do it this way: ax.set_xlim([k_min, k_max])
            ax.set_xlim([k_grid.min(), k_grid.max()])         
            # Add minor ticks automatically. Do it this way: 
            ax.set_xscale("log")
            #Set minor ticks on the x-axis. 
            ax.minorticks_on()
            ax.set_yscale("log")
            ax.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$", fontsize=fontsize*1.1)
            if i == 0:
                ax.set_ylabel(
                    r"$\frac{|P(k)^{\mathrm{CONNECT}} - P(k)^{\mathrm{CLASS}}|}"
                    r"{\mathrm{rms}(P(k)^{\mathrm{CLASS}})}$", fontsize=fontsize*1.4
                )
            ax.set_title(f"z={z_key}")

            # Write P(k) or P(k)_cb in the right corner:
            if "pk_cb" in pk_output:
                ax.text(0.95, 0.95, r"$P_{\rm cb}(k)$", fontsize=fontsize*1.1,
                        ha='right', va='top', transform=ax.transAxes)
            elif "pk" in pk_output:
                ax.text(0.95, 0.95, r"$P(k)$", fontsize=fontsize*1.1,
                        ha='right', va='top', transform=ax.transAxes)
            else:
                ax.text(0.95, 0.95, pk_output, fontsize=fontsize*1.1,
                        ha='right', va='top', transform=ax.transAxes)

        # Make legend for all models
        custom_lines = [Line2D([], [], color=c_list[ii], lw=2)
                        for ii in range(len(model_paths))]
        fig_error.legend(custom_lines, model_names,
                         loc='upper right', fontsize=fontsize*0.9,
                         ncol=len(model_paths),
                         bbox_to_anchor=(0.84, 1.11))



        fname = f"Pk_{pk_output}_error_zall.pdf"
        fig_error.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
        plt.close(fig_error)
        print(f"Saved multi‐z error plot => {fname}")


#############################################################################
# 5) (NEW) Combined figure with 2 rows: top row (spectra), bottom row (errors)
#    for each pk_output, with columns for each redshift.
#############################################################################

if "output_Pk" in output_info:
    pk_outputs = output_info["output_Pk"]
    k_grid     = np.array(output_info["k_grid"])

    for pk_output in pk_outputs:

        pk_z_dict = output_info["interval"]["Pk"][pk_output]
        z_keys = sorted(pk_z_dict.keys(), key=lambda x: float(x))
        ncols  = len(z_keys)

        # Create the combined figure: 2 rows, ncols columns
        fig_comb, axs_comb = plt.subplots(
            2, ncols, figsize=(6*ncols, 8),
            sharex=True,
            #Share the y-axis between the top and bottom rows, respectively.
            sharey='row',
            #Set the height ratio between the top and bottom rows.
            gridspec_kw={'height_ratios': [1.4, 1]}
        )
        #Set no whitespace between the subplots.
        fig_comb.subplots_adjust(wspace=0, hspace=0)
        
        if ncols == 1:
            # If there's only 1 column, make them lists for indexing
            axs_comb[0] = [axs_comb[0]]
            axs_comb[1] = [axs_comb[1]]

        # Prepare a dense k grid
        k_dense = np.logspace(np.log10(k_grid.min()), np.log10(k_grid.max()), 1000)

        # --- Top row: P(k) for the first model vs CLASS
        for i, z_key in enumerate(z_keys):
            ax_top = axs_comb[0][i]
            store_key = f"{pk_output}_z{z_key}"

            pk_data_slice = out_data_pk[store_key]
            pk_pred_slice = out_predict_pk[store_key]

            cs_data = CubicSpline(k_grid, pk_data_slice, bc_type='natural')
            cs_pred = CubicSpline(k_grid, pk_pred_slice, bc_type='natural')

            ax_top.loglog(k_dense, cs_data(k_dense), 'k-', label='CLASS', lw=3)
            ax_top.loglog(k_dense, cs_pred(k_dense), '--', color=c_list[0],
                          label=model_names[0], dashes=(4, 3), lw=3)

            if i == 0:
                ax_top.set_ylabel(r"$P(k)\,[ (\mathrm{Mpc}/h)^3 ]$",
                                  fontsize=fontsize*1.3)
            ax_top.set_xlabel(r"$k \,[h/\mathrm{Mpc}]$", fontsize=fontsize*1.2)
            ax_top.set_title(rf"$z={z_key}$")

            # Set x‐limits
            ax_top.set_xlim([k_grid.min(), k_grid.max()])
            # Add minor ticks on both axes
            ax_top.minorticks_on()
            
            # Write P(k) or P(k)_cb in the right corner:
            if "pk_cb" in pk_output:
                ax_top.text(0.95, 0.95, r"$P_{\rm cb}(k)$", fontsize=fontsize*1.1,
                            ha='right', va='top', transform=ax_top.transAxes)
            elif "pk" in pk_output:
                ax_top.text(0.95, 0.95, r"$P(k)$", fontsize=fontsize*1.1,
                            ha='right', va='top', transform=ax_top.transAxes)
            else:
                ax_top.text(0.95, 0.95, pk_output, fontsize=fontsize*1.1,
                            ha='right', va='top', transform=ax_top.transAxes)
                


        # --- Bottom row: errors for all models
        for i, z_key in enumerate(z_keys):
            ax_bot = axs_comb[1][i]

            for m_i, mp in enumerate(model_paths):
                errors_arr = get_error_pk(mp, pk_output, z_key)
                percentile_arrays = []
                for p_ in sorted(percentiles):
                    pval = np.percentile(errors_arr, 100*p_, axis=1)
                    percentile_arrays.append(pval)
                percentile_arrays = np.array(percentile_arrays)

                for j, p_ in reversed(list(enumerate(sorted(percentiles)))):
                    spl      = CubicSpline(k_grid, percentile_arrays[j], bc_type='natural')
                    envelope = spl(k_dense)
                    ax_bot.fill_between(k_dense, 0, envelope,
                                        color=c_list[m_i],
                                        alpha=alpha_list[j],
                                        zorder=m_i)
                    ax_bot.plot(k_dense, envelope,
                                color=c_list[m_i], lw=1.5, zorder=m_i)

            ax_bot.set_xscale("log")
            ax_bot.set_yscale("log")
            ax_bot.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$", fontsize=fontsize*1.2)
            if i == 0:
                ax_bot.set_ylabel(
                    r"$\frac{|P(k)^{\mathrm{CONNECT}} - P(k)^{\mathrm{CLASS}}|}"
                    r"{\mathrm{rms}(P(k)^{\mathrm{CLASS}})}$",
                    fontsize=fontsize*1.5
                )
            ax_bot.minorticks_on()
            from matplotlib.ticker import LogLocator
            ax_bot.yaxis.set_minor_locator(LogLocator(subs='all'))
            ax_bot.tick_params(which='both', direction='in', right=True, left=True)
            ax_bot.minorticks_on()

        # Build a legend for the bottom row (all models)
        custom_line_class = Line2D([], [], color='black', lw=2)
        custom_lines = [Line2D([], [], color=c_list[ii], lw=2)
                        for ii in range(len(model_paths))]
        lines = [custom_line_class] + custom_lines
        names = ['CLASS'] + model_names
        fig_comb.legend(lines, names,
                        loc='upper center', bbox_to_anchor=(0.5, 1.01),
                        fontsize=fontsize*1.25, ncol=len(model_paths)+1)
        plt.minorticks_on()

        fname_comb = f"Pk_{pk_output}_combined_zall.pdf"
        fig_comb.savefig(os.path.join(output_dir, fname_comb), bbox_inches='tight')
        plt.close(fig_comb)
        print(f"Saved combined multi‐z figure => {fname_comb}")




