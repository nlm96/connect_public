"""
Script for making triangle plots from analysed Monte Python jobs

 - Andreas Nygaard (2020)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os import path

textwidth = 440  # JCAP uses a textwidth of 440 pts
height = 2
width = textwidth/72.27
fontsize = 11/1.2

latex_preamble = [
    r'\usepackage{lmodern}',
    r'\usepackage{amsmath}',
    r'\usepackage{amsfonts}',
    r'\usepackage{mathtools}',
    r'\usepackage{siunitx}',
]

# Join the list into a single string with newline separators
latex_preamble = '\n'.join(latex_preamble)

matplotlib.rcParams.update({
    'text.usetex'        : True,
    'font.family'        : 'serif',
    'font.serif'         : 'cmr10',
    'font.size'          : fontsize,
    'mathtext.fontset'   : 'cm',
    'text.latex.preamble': latex_preamble,
})




"""______________________ Set options and parameters ______________________"""

# Set output path (end with "/")
output_path = '/home/maanson/Speciale/connect/output/'

# Set names as they appear in the parameter file
#names = ['omega_b', 'omega_cdm', 'H0', 'ln10^{10}A_s', 'n_s', 'tau_reio']

names = [
    'omega_b', 
    'omega_cdm', 
    'H0', 
    'ln10A_s', 
    'n_s', 
    'tau_reio',
    'A_planck',
    'z_reio',
    'Omega_Lambda',
    'YHe',
    'A_s',
    'sigma8',
    '100theta_s'
]


# Set labelnames as you would like them to appear in the figure (supports latex syntax)
#labelnames = [r'$H_0$', r'$\omega_{\mathrm{cdm}}$', r'$100\times\omega_{\mathrm{b}}$']
labelnames = [
    r'$10^{-2}\omega_b$', 
    r'$\omega_{\mathrm{cdm}}$', 
    r'$H_0$', 
    r'$\ln(10^{10}A_s)$', 
    r'$n_s$', 
    r'$\tau_{\mathrm{reio}}$',
    r'$A_{\mathrm{planck}}$', 
    r'$z_{\mathrm{reio}}$', 
    r'$\Omega_{\Lambda}$', 
    r'$Y_{\mathrm{He}}$', 
    r'$10^{-9}A_s$', 
    r'$\sigma_8$', 
    r'$100\theta_s$'
]


# Set x-limits and ticks for each parameter. If a rows is left blanck, python will
# automatically find limits and ticks.
#xlimits = [[65, 70], [0.115, 0.126], [2.16, 2.32]]
#ticklist = [[66, 67.5, 69], [0.117, 0.121, 0.125], [2.2 ,2.25 ,2.3]]

xlimits = [[], [], [], [], [], [], [], [], [], [], [], [], []]
ticklist = [[], [], [], [], [], [], [], [], [], [], [], [], []]


# Choose which jobs to be used for the figure.
# The jobs must have already been analysed with the '--all' option,
# and their location must be in the data folder.
# If they are in a subfolder, you must write both the subfolder and job name
# e.g. 'subfolder/jobname'
JOBnames = ['lcdm_example/compare_iterations']

# Each job is given a legend label
#legend_labels = [r'$\Lambda$CDM (PLANCK-2018)', r'$\Lambda$CDM (PLANCK-2015)']
legend_labels = [
    r'$\Lambda$CDM (CONNECT Iteration 1)', 
    r'$\Lambda$CDM (CONNECT Iteration 2)', 
    r'$\Lambda$CDM (CONNECT Iteration 3)', 
    r'$\Lambda$CDM (CONNECT Iteration 4)'
]


# The colors and styles of the 1D posteriors will be given to the different jobs
# in this order
style1d = ['r-', 'b--', 'k:', 'y-.']

# The corresponding order for the 95- and 68-level contours
color95 = ['r', 'b', 'k', 'y']
color68 = ['r', 'b', 'k', 'y']

# The transparency of the contours
alpha95 = 0.2
alpha68 = 0.4


## layout options
square = True       # should each subfigure be a square in shape?

ticksize = fontsize # fontsize of ticklabels (default to JCAP fontsize) 

only_1D = False     # plot only the 1D posteriors in a horizontal plot

dist = 0.05         # distance between subplots







"""____________________________ Automatic from here _____________________________"""

for j, name in enumerate(names):
  exec('i_'+name+'=j+1')

dim = len(names)

for dex, Jn in enumerate(JOBnames):
  for name in names:
    with open('data/' + JOBnames[dex] + '/plots/' + JOBnames[dex] + '_' + name + '.hist') as f:
      exec('h_'+name+'_'+Jn+' = np.loadtxt(f, delimiter=",", dtype="float", comments="#", skiprows=1, usecols=None)')

if only_1D:
  fig = plt.figure(figsize=(width, height))
  for j, name in enumerate(names):
    exec('ax'+name+'1d = fig.add_subplot(1,dim+1,j+2)')
    if len(xlimits[j]):
      exec('ax'+name+'1d.set_xlim(xlimits[j])')
    if len(ticklist[j]):
      exec('ax'+name+'1d.set_xticks(ticklist[j])')
    for i,Jn in enumerate(JOBnames):
      exec(name+'_'+Jn+'_1d_plot=ax'+name+'1d.plot(h_'+name+'_'+Jn+'[0],h_'+name+'_'+Jn+'[1],style1d[i])[0]')
      exec('ax'+name+'1d.get_yaxis().set_visible(False)')
      exec('ax'+name+'1d.set_xlabel(labelnames[j],labelpad=10)')
      exec('ax'+name+'1d.tick_params(labelsize=ticksize)')        
else:
  if square:
    fig = plt.figure(figsize=(width, width))
  else:
    fig = plt.figure(figsize=(width, height))
  for j, name in enumerate(names):
    exec('ax'+name+'1d = fig.add_subplot(dim,dim,i_'+name+'*dim-(dim-i_'+name+'))')
    if len(xlimits[j]):
      exec('ax'+name+'1d.set_xlim(xlimits[j])')
    if len(ticklist[j]):
      exec('ax'+name+'1d.set_xticks(ticklist[j])')
    for i, Jn in enumerate(JOBnames):
      exec(name+'_'+Jn+'_1d_plot=ax'+name+'1d.plot(h_'+name+'_'+Jn+'[0],h_'+name+'_'+Jn+'[1],style1d[i])[0]')
      if j == len(names) - 1:
        exec('ax'+name+'1d.get_yaxis().set_visible(False)')
        exec('ax'+name+'1d.set_xlabel(labelnames[j],labelpad=10)')
        exec('ax'+name+'1d.tick_params(labelsize=ticksize)')
      else:
        exec('ax'+name+'1d.get_yaxis().set_visible(False)')
        exec('ax'+name+'1d.tick_params(labelbottom=False)')


if not only_1D:
  for dex, Jn in enumerate(JOBnames):
    for i in range(0,dim):
      for j in range(0,dim):
        if i > j:
          if dex == 0:
            exec('ax'+names[i]+names[j]+f'2d = fig.add_subplot({dim:.0f},{dim:.0f},i_'+names[i]+f'*{dim:.0f}-({dim:.0f}-i_'+names[j]+'))')
          if path.exists('data/'+Jn+'/plots/'+Jn+'_2d_'+names[i]+'-'+names[j]+'.dat'):
            ii = i
            jj = j
            switch = False
          else:
            ii = j
            jj = i
            switch = True
          with open('data/' + Jn + '/plots/' + Jn + '_2d_'+names[ii] + '-' + names[jj] + '.dat') as f:
            x = []
            y = []
            x95_list = []
            y95_list = []
            x68_list = []
            y68_list = []
            cnt95 = 0
            cnt68 = -1
            for line in f:
              if line != '\n' and cnt68 == -1 and line[0] != '#':
                x.append(float(line.split('\t')[0]))
                y.append(float(line.split('\t')[1].split('\n')[0]))
              elif line == '\n' and cnt68 == -1:
                cnt95 += 1
                x95_list.append(x)
                y95_list.append(y)
                x = []
                y = []
              elif line[0] == '#' and cnt95 > 0:
                cnt68 += 1
              elif line != '\n' and cnt68 >= 0 and line[0] != '#':
                x.append(float(line.split('\t')[0]))
                y.append(float(line.split('\t')[1].split('\n')[0]))
              elif line == '\n' and cnt68 >= 0:
                cnt68 += 1
                x68_list.append(x)
                y68_list.append(y)
                x = []
                y = []
          if switch:
            x95 = y95_list
            y95 = x95_list
            x68 = y68_list
            y68 = x68_list
          else:
            x95 = x95_list
            y95 = y95_list
            x68 = x68_list
            y68 = y68_list
          for cont in range(cnt95 - 2):
            exec('ax'+names[i]+names[j]+'2d.fill(x95[cont],y95[cont],color=color95[dex],alpha=alpha95,lw=0)')
            exec('ax'+names[i]+names[j]+'2d.plot(x95[cont],y95[cont],"-",color=color95[dex],lw=1.5,alpha=alpha95+0.2)')
          for cont in range(cnt68 - 2):
            exec('ax'+names[i]+names[j]+'2d.fill(x68[cont],y68[cont],color=color68[dex],alpha=alpha68,lw=0)')
            exec('ax'+names[i]+names[j]+'2d.plot(x68[cont],y68[cont],"-",color=color68[dex],lw=1.5,alpha=alpha68+0.2)')
          if len(xlimits[i]):
            exec('ax'+names[i]+names[j]+'2d.set_ylim(xlimits[i])')
          if len(xlimits[j]):
            exec('ax'+names[i]+names[j]+'2d.set_xlim(xlimits[j])')
          if len(ticklist[i]):
            exec('ax'+names[i]+names[j]+'2d.set_yticks(ticklist[i])')
          if len(ticklist[j]):
            exec('ax'+names[i]+names[j]+'2d.set_xticks(ticklist[j])')
          exec('ax'+names[i]+names[j]+'2d.tick_params(labelleft=False)')
          exec('ax'+names[i]+names[j]+'2d.tick_params(labelbottom=False)')
          if i == dim - 1:
            exec('ax'+names[i]+names[j]+'2d.tick_params(labelbottom=True)')
            exec('ax'+names[i]+names[j]+'2d.set_xlabel(labelnames[j],labelpad=10)')
            exec('ax'+names[i]+names[j]+'2d.tick_params(labelsize=ticksize)')
          if j == 0:
            exec('ax'+names[i]+names[j]+'2d.tick_params(labelleft=True)')
            exec('ax'+names[i]+names[j]+'2d.set_ylabel(labelnames[i],labelpad=10)')
            exec('ax'+names[i]+names[j]+'2d.tick_params(labelsize=ticksize)')


fig.align_ylabels()
lines = []
labels = []
for i, Jn in enumerate(JOBnames):
  lines.append(eval(names[0]+'_'+Jn+'_1d_plot'))
  labels.append(legend_labels[i])
if square and not only_1D:
  fig.legend(lines, labels, bbox_to_anchor=(0.75, 0.72, 0.2, 0.2))
elif only_1D:
  fig.legend(lines, labels, loc='center left')
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
  PATH += name
  if i != len(JOBnames) - 1:
    PATH += '-'
PATH += '.pdf'
plt.savefig(PATH)
plt.show()
