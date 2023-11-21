#%%
'''
!! not maintained !!
modified from a version of figGen.py (can remember which one)
This file exists only for backup purposes
'''
#%%
# ---------------------------------------------------------------------------- #
#                                     start                                    #
# ---------------------------------------------------------------------------- #
import pandas as pd
from Exp import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import matplotlib

# [YC] start
showFig = False
showResults = False
saveFigDir = Path(".")/"figs_YC_new"
saveFigDir.mkdir(parents=True,exist_ok=True)
# [YC] end

import nmslplot.nmslplot as nmslplot

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 1000

pd.options.mode.use_inf_as_na = True
np.__version__
exp = Exp()
FPS = 50
VIDEO_LEN = 30
WINDOW_SIZE=FPS
N_USERS = 16
N_SCENES = 3
SCENE = 'House'
SCENES = ['House', 'Bigroom', 'Smallroom']
DEFAULT_PARAM = {
    'fps': FPS,
    'videoLen': VIDEO_LEN,
    'scene': '.*',
    'runTimeResDs': 4,
    'm': 0.03, 
    'h': 0.15,
    'ffrMask': 'Off',
    'maxNumNodes': 96,
    'depthThres': 0.01,
    'psPolicy': 'uniform',
    'cspPolicy': '1Prob',
    'windowSize': 50,
    'solverPolicy': 'UnB',
    # 'solverPolicy': 'BB',
    'placePolicy': 'average',  
    'a': '1e-05',
}
METRICS = ['PSNR', 'SSIM', 'VMAF']

#%% scene complexity
# ---------------------------------------------------------------------------- #
#                               scene complexity                               #
# ---------------------------------------------------------------------------- #
import open3d as o3d
from pathlib import Path
'''
FurnishedCabin: trianges = 30873, vertices = 28994, obj = 156
ScifiTraceBig/Smallroom: trianges = 207310, vertices = 258432, obj = 554
'''
# for scene in ['FurnishedCabin', 'ScifiTraceBigroom']:
# 	dir = Path('Trace_FPS50_LEN30')/scene
# 	obj = list(dir.glob('*.obj'))[0]
# 	mesh = o3d.io.read_triangle_mesh(str(obj))
# 	print(f'{scene}: trianges = {np.asarray(mesh.triangles).shape[0]}, vertices = {np.asarray(mesh.vertices).shape[0]}')

#%%
# from pathlib import Path
# import shutil
# pattern = 'Trace_FPS50_LEN30_*'
# dst = Path('..')/'Qual'
# dirs = list(Path('.').glob(pattern))
# usedDirs = []
# # print(len(dirs))
# for dir in dirs:
#     if (dir/'qual').is_dir():
#         usedDirs.append(str(dir/'qual'))
#         print(dst/dir/'qual')
#         shutil.copytree(dir/'qual', dst/dir/'qual')
# print(len(usedDirs))

#%%
# copy exp.json
# from pathlib import Path
# import shutil
# pattern = 'Trace_FPS50_LEN30_s*'
# dst = Path('..')/'Qual'
# dirs = list(Path('..').glob(pattern))
# usedDirs = []
# print(len(dirs))
# for dir in dirs:
#     print(dir.parts[-1])
#     shutil.copyfile(dir/'Exp.json', dst/dir.parts[-1]/'Exp.json')
# # print(len(usedDirs))

#%%
# shape = exp.mergeAllQualFiles(FPS * VIDEO_LEN, 'Qual/Trace_FPS50_LEN30_s*', 'qual_merge.csv', 'qual_merge.json')
# print(shape)

#%%
# ---------------------------------------------------------------------------- #
#                                Load json file                                #
# ---------------------------------------------------------------------------- #
expJ = json.load(open('qual_merge.json', 'r'))
expJson = {}
for key in expJ:
    nkey = key
    nkey = nkey.replace('FurnishedCabin', 'House')
    nkey = nkey.replace('ScifiTraceBigroom', 'Bigroom')
    nkey = nkey.replace('ScifiTraceSmallroom', 'Smallroom')
    expJson[Path(nkey).parts[-1]] = expJ[key]
for key in expJson:
    print(key)

#%%
# ---------------------------------------------------------------------------- #
#                                 Read csv file                                #
# ---------------------------------------------------------------------------- #
df = pd.read_csv('qual_merge.csv')
df = df.drop("Unnamed: 17", axis=1)
# for paper
df.rename(columns={
    'float_ssim': 'SSIM',
    'psnr_y': 'PSNR',
    'vmaf': 'VMAF',
    'solverPolicy': 'solver',
    'runTimeResDs': 'probing view downsample'
}, inplace=True)
df = df.assign(VMAF = lambda x: x.VMAF/100)
df.expDir.replace({
    'FurnishedCabin': 'House',
    'ScifiTraceBigroom': 'Bigroom',
    'ScifiTraceSmallroom': 'Smallroom',
}, regex=True, inplace=True)
df.scene.replace(
    {
    'FurnishedCabin': 'House',
    'ScifiTraceBigroom': 'Bigroom',
    'ScifiTraceSmallroom': 'Smallroom',
}, regex=True, inplace=True)
# df.solver.replace(
#     {
#     'BB': 'B&B',
#     'UnB': 'U&B',
# }, regex=True, inplace=True)
df.shape

#%%
# ---------------------------------------------------------------------------- #
#                               plot single user                               #
# ---------------------------------------------------------------------------- #
# https://stackoverflow.com/questions/47591650/second-y-axis-time-series-seaborn
'''
PSNR: mean = 33.80240822666667, std = 2.5817905678739086
SSIM: mean = 0.9776985226666667, std = 0.010113549230356412
VMAF: mean = 0.7899586731066666, std = 0.09045225086433144

argmax: 941
argmin: 400
'''
user = 0
param = dict(DEFAULT_PARAM)
param['scene'] = 'House'
# param['scene'] = 'Bigroom'
# param['scene'] = 'Smallroom'

whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)
selected = df[df.expDir.str.match(whichDir)]
selected = selected[selected.user == user]
selected.reset_index(inplace=True, drop=True)
print(selected.shape)
assert selected.shape[0] == FPS*VIDEO_LEN
ax = selected.plot(x='Frame', y=['SSIM', 'VMAF'], color=['r', 'g'])
ax2  = ax.twinx()
selected.plot(x='Frame', y='PSNR', ax=ax2, legend=False, color='b')
# ax.figure.legend()

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.get_legend().remove()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

ax.set_xlabel('View')
ax.set_title(f'Sample User Results from {param["scene"]}')
ax.set_ylabel('SSIM / VMAF')
ax2.set_ylabel('PSNR')
plt.savefig(f'{saveFigDir}/sample_user_{param["scene"]}.eps', bbox_inches = 'tight')
# [YC] start
# plt.show()
if showFig:
	plt.show()
plt.clf()
# [YC] end
for metric in METRICS:
	print(f'\t{metric}: mean = {selected[metric].mean()}, std = {selected[metric].std()}')
print(selected['PSNR'].argmax(), selected['PSNR'].argmin())

#%%
# ---------------------------------------------------------------------------- #
#                          plot single user aggregated                         #
# ---------------------------------------------------------------------------- #
# https://stackoverflow.com/questions/47591650/second-y-axis-time-series-seaborn
'''
PSNR: mean = 33.80240822666667, std = 2.5817905678739086
SSIM: mean = 0.9776985226666667, std = 0.010113549230356412
VMAF: mean = 0.7899586731066666, std = 0.09045225086433144

argmax: 941
argmin: 400
'''
user = 0
param = dict(DEFAULT_PARAM)
param['scene'] = 'House'
# param['scene'] = 'Bigroom'
# param['scene'] = 'Smallroom'

whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)
selected = df[df.expDir.str.match(whichDir)]
selected = selected[selected.user == user]
selected.reset_index(inplace=True, drop=True)
print(selected.shape)
assert selected.shape[0] == FPS*VIDEO_LEN

# aggregate every FPS frames
selected = selected.assign(Frame = lambda x: (x.Frame//FPS) * FPS)
# print(selected['Frame'].unique())
# print((selected['Frame'] == 0).sum())
# ax = sns.lineplot(data=selected, x='Frame', y='SSIM', legend=False, color='r')
# ax = sns.lineplot(data=selected, x='Frame', y='VMAF', legend=False, color='g')
# ax2  = ax.twinx()
# sns.lineplot(data=selected, x='Frame', y='PSNR', ax=ax2, legend=False, color='b')

ax = sns.lineplot(data=selected, x='Frame', y='SSIM', color='r', label='SSIM', err_style='bars', markers=True, dashes=True)
sns.lineplot(data=selected, x='Frame', y='VMAF', color='g', label='VMAF', ax=ax, err_style='bars', markers=True, dashes=True)
ax2  = ax.twinx()
sns.lineplot(data=selected, x='Frame', y='PSNR', color='b', label='PSNR', ax=ax2, err_style='bars', markers=True, dashes=True)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.get_legend().remove()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

# ax.figure.legend()
# ax2.figure.legend()
ax.set_xlabel('Frame Number')
ax.set_title(f'Sample User Results from {param["scene"]}')
ax.set_ylabel('SSIM / VMAF')
ax2.set_ylabel('PSNR')
plt.savefig(f'{saveFigDir}/sample_user_{param["scene"]}_aggregate.eps', bbox_inches = 'tight')

# [YC] start
# plt.show()
if showFig:
	plt.show()
plt.clf()
# [YC] end

for metric in METRICS:
	print(f'\t{metric}: mean = {selected[metric].mean()}, std = {selected[metric].std()}')
print(selected['PSNR'].argmax(), selected['PSNR'].argmin())
#%%
# ---------------------------------------------------------------------------- #
#                          plot single user aggregated subplots  [[SM]         #
# ---------------------------------------------------------------------------- #
# https://stackoverflow.com/questions/47591650/second-y-axis-time-series-seaborn
'''
PSNR: mean = 33.80240822666667, std = 2.5817905678739086
SSIM: mean = 0.9776985226666667, std = 0.010113549230356412
VMAF: mean = 0.7899586731066666, std = 0.09045225086433144

argmax: 941
argmin: 400
'''
user = 0
param = dict(DEFAULT_PARAM)
param['scene'] = 'House'
# param['scene'] = 'Bigroom'
# param['scene'] = 'Smallroom'

whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)
selected = df[df.expDir.str.match(whichDir)]
selected = selected[selected.user == user]
selected.reset_index(inplace=True, drop=True)
print(selected.shape)
assert selected.shape[0] == FPS*VIDEO_LEN

# aggregate every FPS frames
selected = selected.assign(Frame = lambda x: (x.Frame//FPS) * FPS)
selected.rename(columns={
    'PSNR': 'PSNR(dB)'
}, inplace=True)
# print(selected['Frame'].unique())
# print((selected['Frame'] == 0).sum())
# ax = sns.lineplot(data=selected, x='Frame', y='SSIM', legend=False, color='r')
# ax = sns.lineplot(data=selected, x='Frame', y='VMAF', legend=False, color='g')
# ax2  = ax.twinx()
# sns.lineplot(data=selected, x='Frame', y='PSNR', ax=ax2, legend=False, color='b')

plt.rcParams['figure.dpi'] = 1000
fig, axes = plt.subplots(3, 1, figsize=(20, 5), sharex=True)
sns.lineplot(ax=axes[0], data=selected, x='Frame', y='PSNR(dB)', err_style='bars', markers=True, dashes=True)
# plt.title('PSNR')
plt.subplot(312)
sns.lineplot(ax=axes[1], data=selected, x='Frame', y='SSIM', err_style='bars', markers=True, dashes=True)
# plt.title('SSIM')
plt.subplot(313)
sns.lineplot(ax=axes[2], data=selected, x='Frame', y='VMAF', err_style='bars', markers=True, dashes=True)
# plt.title('VMAF')
# plt.show()
plt.savefig(f'{saveFigDir}/sample_user_{param["scene"]}_aggregate_subplots.svg', bbox_inches = 'tight')
# plt.savefig(f'{saveFigDir}/sample_user_{param["scene"]}_aggregate_subplots.svg')

# # [YC] start
# # plt.show()
# if showFig:
# 	plt.show()
# plt.clf()
# # [YC] end

# for metric in METRICS:
# 	print(f'\t{metric}: mean = {selected[metric].mean()}, std = {selected[metric].std()}')
# print(selected['PSNR'].argmax(), selected['PSNR'].argmin())

#%%
# ---------------------------------------------------------------------------- #
#               plot increaseing source views with pure m variant              #
# ---------------------------------------------------------------------------- #
def scene_m_metric(df):
    for scene in SCENES:
        if showResults:
            print(f'{scene}:')
        for m in [0.01, 0.02, 0.03, 0.04, 0.05]:
            if showResults:
                print(f'm: {m}')
            for metric in METRICS:
                sdf = df[df.scene.str.match(scene)]
                sdf = sdf[sdf.m == m]
                if showResults:
                    print(f'\t{metric}: mean = {sdf[metric].mean()}, std = {sdf[metric].std()}')
selections = []
for m in [0.01, 0.02, 0.03, 0.04, 0.05]:
    param = dict(DEFAULT_PARAM)
    param['m'] = m
    whichDir = generateExpDirName(**param)
    whichDir = str(whichDir)
    selections.append(df[df.expDir.str.match(whichDir)])

selected = pd.concat(selections)
selected.reset_index(inplace=True, drop=True)
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * 5 * N_SCENES

scene_m_metric(selected)
'''
House:
m: 0.01
	PSNR: mean = 29.089999538166673, std = 5.328175932330962
	SSIM: mean = 0.9335951085416667, std = 0.057726123779299025
	VMAF: mean = 0.5678797369783333, std = 0.22447306649760412
m: 0.02
	PSNR: mean = 33.07611550575, std = 4.178163645556539
	SSIM: mean = 0.9709247772916666, std = 0.029013877789046186
	VMAF: mean = 0.7335372898879167, std = 0.1419891013286737
m: 0.03
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
m: 0.04
	PSNR: mean = 35.620096366375, std = 3.7475844603110815
	SSIM: mean = 0.9838468349166666, std = 0.01470287400466308
	VMAF: mean = 0.8044405311195832, std = 0.09905730056683948
m: 0.05
	PSNR: mean = 36.03940690033334, std = 3.720261298657041
	SSIM: mean = 0.9855238760416666, std = 0.013293628261186151
	VMAF: mean = 0.8167703094129166, std = 0.09148636820780993
Bigroom:
m: 0.01
	PSNR: mean = 26.576996197333333, std = 2.7436727146675133
	SSIM: mean = 0.8963819614583334, std = 0.043393054087579506
	VMAF: mean = 0.34919744184874996, std = 0.16223665019884387
m: 0.02
	PSNR: mean = 27.841249648416667, std = 2.2016240567383343
	SSIM: mean = 0.924225327625, std = 0.026152820737713477
	VMAF: mean = 0.45157891930125, std = 0.12154963967825867
m: 0.03
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
m: 0.04
	PSNR: mean = 28.358903911458334, std = 2.155845758173473
	SSIM: mean = 0.9357095459166667, std = 0.02039421709711931
	VMAF: mean = 0.49276264088124994, std = 0.10633846501264721
m: 0.05
	PSNR: mean = 28.417357748041667, std = 2.1647330440826233
	SSIM: mean = 0.9372445787500001, std = 0.02010175327057704
	VMAF: mean = 0.4985744285704166, std = 0.10615470805164595
Smallroom:
m: 0.01
	PSNR: mean = 26.179013113083332, std = 2.9532607539556275
	SSIM: mean = 0.8789099611250001, std = 0.07555433771576267
	VMAF: mean = 0.45624391661333324, std = 0.20763071024912552
m: 0.02
	PSNR: mean = 27.365413485333335, std = 2.7244166889879837
	SSIM: mean = 0.9181181547916667, std = 0.05181776253588877
	VMAF: mean = 0.57217077073, std = 0.1630151191302616
m: 0.03
	PSNR: mean = 27.736233725541663, std = 2.7748232212338655
	SSIM: mean = 0.93164457025, std = 0.04407146910365807
	VMAF: mean = 0.6130295675799999, std = 0.14993997780637322
m: 0.04
	PSNR: mean = 27.8848491635, std = 2.7544584247141244
	SSIM: mean = 0.9367687794583334, std = 0.04062503790994172
	VMAF: mean = 0.6275374509325, std = 0.14411791978883692
m: 0.05
	PSNR: mean = 27.970207567083335, std = 2.7847694847030784
	SSIM: mean = 0.940281063625, std = 0.038127432315984144
	VMAF: mean = 0.6369998686516666, std = 0.1417084324406309
'''

# [YC] start
# ---------------------------------------------------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='m')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/m_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(df=selected, x='scene', y='SSIM', hue='m',
                xlabel='Scene', ylabel='SSIM',
				loc='lower left',
                savePlot=True, showPlot=False,
                saveDir=saveFigDir, saveImgName="m_SSIM")

# ---------------------------------------------------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='m')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/m_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='m', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="m_VMAF")

# ---------------------------------------------------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='m')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/m_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='m', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="m_PSNR")

#%%
# ---------------------------------------------------------------------------- #
#                             plot TMM - All solver                            #
# ---------------------------------------------------------------------------- #
def scene_solver_metric(df):
    for scene in SCENES:
        if showResults:
            print(f'{scene}:')
        for solver in df.solver.unique().tolist():
            if showResults:
                print(f'solver: {solver}')
            for metric in METRICS:
                sdf = df[df.scene.str.match(scene)]
                sdf = sdf[sdf.solver == solver]
                if showResults:
                    print(f'\t{metric}: mean = {sdf[metric].mean()}, std = {sdf[metric].std()}')
param = dict(DEFAULT_PARAM)
param['solverPolicy'] = '.*'
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

hue_order = ['C2I', 'C2G', 'Uni', 'B&B',  'U&B', 'Opt']
selected = df[df.expDir.str.match(whichDir)]
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * len(hue_order) * N_SCENES

# [YC] note: This will cause failed
# selected.solver.replace({
# 	'BB': 'B&B',
# 	'UnB': 'U&B',
# }, inplace=True, regex=True)
# [YC] end

selected.solver.replace({
	'BB': 'B\&B',
	'UnB': 'U\&B',
}, inplace=True, regex=True)
scene_solver_metric(selected)

'''
House:
solver: BB
	PSNR: mean = 34.79610603141666, std = 3.983096361818194
	SSIM: mean = 0.9797471957916667, std = 0.021098097362196432
	VMAF: mean = 0.7832819167933334, std = 0.10949820076878712
solver: C2G
	PSNR: mean = 31.501577340874995, std = 6.2174112982408625
	SSIM: mean = 0.9471711595, std = 0.07349957504892342
	VMAF: mean = 0.6484518507645834, std = 0.2565300356382046
solver: C2I
	PSNR: mean = 26.312460215458337, std = 11.629129991301783
	SSIM: mean = 0.8017475601250001, std = 0.2690232447340758
	VMAF: mean = 0.45914168294916663, std = 0.3909525853472614
solver: Opt
	PSNR: mean = 36.62786869033334, std = 3.8935047586938767
	SSIM: mean = 0.9873186097499999, std = 0.010390595407209659
	VMAF: mean = 0.8361148075545832, std = 0.07508513130041017
solver: UnB
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
solver: Uni
	PSNR: mean = 34.51487282762499, std = 4.573305035972031
	SSIM: mean = 0.9778155882916667, std = 0.022089946400975138
	VMAF: mean = 0.7856168671925, std = 0.11128266700034317
Bigroom:
solver: BB
	PSNR: mean = 28.196301047416664, std = 2.163025660136442
	SSIM: mean = 0.932009376, std = 0.022390579183575503
	VMAF: mean = 0.47969399473333335, std = 0.11272346285084807
solver: C2G
	PSNR: mean = 26.140295569500005, std = 4.678284351746452
	SSIM: mean = 0.877838163625, std = 0.13129100840345814
	VMAF: mean = 0.36743701566416664, std = 0.2063286391333103
solver: C2I
	PSNR: mean = 22.694838331541668, std = 6.5694100043334425
	SSIM: mean = 0.7991435533333333, std = 0.22308046461651934
	VMAF: mean = 0.28002971025208334, std = 0.2493375701728708
solver: Opt
	PSNR: mean = 28.453751984291667, std = 2.2586007115647058
	SSIM: mean = 0.9394676013750001, std = 0.02171911624171557
	VMAF: mean = 0.5102014704416666, std = 0.10658970066476933
solver: UnB
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
solver: Uni
	PSNR: mean = 27.88917209241667, std = 2.5151098572854504
	SSIM: mean = 0.9295511660000001, std = 0.03848066485703486
	VMAF: mean = 0.47915286333083335, std = 0.11547487736375311
Smallroom:
solver: BB
	PSNR: mean = 27.725557522333332, std = 2.7789285852422645
	SSIM: mean = 0.9304924885, std = 0.046154753703812515
	VMAF: mean = 0.6107522512416667, std = 0.153231140189888
solver: C2G
	PSNR: mean = 26.364807529083336, std = 3.6212489368315555
	SSIM: mean = 0.8930151085833333, std = 0.09016262102047248
	VMAF: mean = 0.4983794316229167, std = 0.2354009445971358
solver: C2I
	PSNR: mean = 22.656130177458337, std = 6.624703509218502
	SSIM: mean = 0.7859781427083333, std = 0.23967575412400632
	VMAF: mean = 0.3671631733191667, std = 0.3166989288282265
solver: Opt
	PSNR: mean = 28.00299823625, std = 2.8191895918201872
	SSIM: mean = 0.9471384010416667, std = 0.02640341658048577
	VMAF: mean = 0.6530806474895834, std = 0.13180802661652039
solver: UnB
	PSNR: mean = 27.736233725541663, std = 2.7748232212338655
	SSIM: mean = 0.93164457025, std = 0.04407146910365807
	VMAF: mean = 0.6130295675799999, std = 0.14993997780637322
solver: Uni
	PSNR: mean = 27.437713578416666, std = 2.858476592943197
	SSIM: mean = 0.9338782976666667, std = 0.03565756771129526
	VMAF: mean = 0.6149634176412501, std = 0.1414642463778709
'''
# ----------------------------------- SSIM ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/TMM-slvr_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='solver', 
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="TMM-slvr_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/TMM-slvr_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='solver', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="TMM-slvr_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/TMM-slvr_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='solver', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="TMM-slvr_PSNR")

#%%
# ---------------------------------------------------------------------------- #
#                             plot IXR - All solver                            #
# ---------------------------------------------------------------------------- #
param = dict(DEFAULT_PARAM)
param['solverPolicy'] = '.*'
param['psPolicy'] = 'IXR'
param['placePolicy'] = 'IXR'
param['m'] = 0.01
param['h'] = 0.15
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

hue_order = ['C2I', 'C2G', 'Uni', 'B&B',  'U&B', 'Opt']
selected = df[df.expDir.str.match(whichDir)]
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * len(hue_order) * N_SCENES

selected.solver.replace({
	'BB': 'B\&B',
	'UnB': 'U\&B',
}, inplace=True, regex=True)
scene_solver_metric(selected)
'''
House:
solver: BB
	PSNR: mean = 27.36311856904167, std = 5.355271305737163
	SSIM: mean = 0.9168615489166667, std = 0.061946963318124305
	VMAF: mean = 0.5080882772954167, std = 0.2290910996924739
solver: C2G
	PSNR: mean = 26.242018936041664, std = 6.187726181484558
	SSIM: mean = 0.8899581379166666, std = 0.11989351568939863
	VMAF: mean = 0.46339230333958337, std = 0.2697938796225788
solver: C2I
	PSNR: mean = 22.644297515666665, std = 9.10921921098884
	SSIM: mean = 0.755597533875, std = 0.28915874533690944
	VMAF: mean = 0.3560505985820834, std = 0.3348297655967921
solver: Opt
	PSNR: mean = 30.93959555258333, std = 5.518786935436951
	SSIM: mean = 0.9529111719583334, std = 0.04582966421668191
	VMAF: mean = 0.6779249112325, std = 0.1760305159299288
solver: UnB
	PSNR: mean = 27.443045827375, std = 5.289635400117678
	SSIM: mean = 0.9174117734166666, std = 0.06148458002083469
	VMAF: mean = 0.5091672283170834, std = 0.22788203151571074
solver: Uni
	PSNR: mean = 24.450260159791664, std = 6.9546031048700785
	SSIM: mean = 0.8578665992083333, std = 0.13708415923332287
	VMAF: mean = 0.40343161449291665, std = 0.29478708804026943
Bigroom:
solver: BB
	PSNR: mean = 25.85254987441667, std = 3.1688852198158366
	SSIM: mean = 0.881675289625, std = 0.058738125631290536
	VMAF: mean = 0.3177185013754167, std = 0.1655462423656302
solver: C2G
	PSNR: mean = 24.481416619583335, std = 5.0678679957737245
	SSIM: mean = 0.8408812784583334, std = 0.16085103883067933
	VMAF: mean = 0.28600509393124995, std = 0.19653796292202727
solver: C2I
	PSNR: mean = 20.23115577275, std = 7.472928440035389
	SSIM: mean = 0.6877170047916668, std = 0.3253215724572778
	VMAF: mean = 0.20393830409833333, std = 0.21785820291804373
solver: Opt
	PSNR: mean = 27.070846689208334, std = 2.9059773277912417
	SSIM: mean = 0.9123127202083333, std = 0.045164083657529074
	VMAF: mean = 0.42206715193083333, std = 0.14307199351327196
solver: UnB
	PSNR: mean = 25.831197227625, std = 3.11959713862974
	SSIM: mean = 0.8822067652916666, std = 0.057849454949370105
	VMAF: mean = 0.31825544737958333, std = 0.16321579350519072
solver: Uni
	PSNR: mean = 23.319696800166668, std = 4.935951815663412
	SSIM: mean = 0.832128500625, std = 0.1265705834501292
	VMAF: mean = 0.250081099755, std = 0.19656632846916153
Smallroom:
solver: BB
	PSNR: mean = 25.31736163370833, std = 3.3334312926255825
	SSIM: mean = 0.8631287592083333, std = 0.07558070226366763
	VMAF: mean = 0.41916068486416663, std = 0.20853500565756886
solver: C2G
	PSNR: mean = 24.428443284500002, std = 4.432955921180272
	SSIM: mean = 0.8373692393333334, std = 0.13940875578730452
	VMAF: mean = 0.38424193630166664, std = 0.23741864246658867
solver: C2I
	PSNR: mean = 21.174189367041667, std = 6.716355327206276
	SSIM: mean = 0.7191607053333334, std = 0.2846710420775064
	VMAF: mean = 0.29067365032541664, std = 0.2810251214084191
solver: Opt
	PSNR: mean = 26.455398743958334, std = 3.2296870012142267
	SSIM: mean = 0.9040845729166668, std = 0.055146800447756425
	VMAF: mean = 0.5395850859825, std = 0.17428288703467326
solver: UnB
	PSNR: mean = 25.372635539458336, std = 3.2758860514371446
	SSIM: mean = 0.8633480005833334, std = 0.07555705208984481
	VMAF: mean = 0.42093040662500003, std = 0.20816192041691117
solver: Uni
	PSNR: mean = 23.37957530208333, std = 4.182587445888831
	SSIM: mean = 0.8220355747083332, std = 0.11630117313920384
	VMAF: mean = 0.35789473755375, std = 0.2247721617594679
'''
# ----------------------------------- SSIM ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/IXR-slvr_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='solver', 
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="IXR-slvr_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/IXR-slvr_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='solver', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="IXR-slvr_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='solver', hue_order=hue_order)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/IXR-slvr_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='solver', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="IXR-slvr_PSNR")


#%%
# ---------------------------------------------------------------------------- #
#                                plot foveation                                #
# ---------------------------------------------------------------------------- #
# ! almost no effect
param = dict(DEFAULT_PARAM)
param['ffrMask'] = '.*'
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

selected = df[df.expDir.str.match(whichDir)]
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * 2 * N_SCENES

# ----------------------------------- SSIM ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='ffr')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/ffr_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='ffr', 
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="ffr_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='ffr')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/ffr_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='ffr', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="ffr_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='ffr')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/ffr_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='ffr', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="ffr_PSNR")

# %%
# ---------------------------------------------------------------------------- #
#                               plot runTime Res                               #
# ---------------------------------------------------------------------------- #
# ! almost no effect
def scene_res_metric(df):
    for scene in SCENES:
        print(f'{scene}:')
        for Aux_downsample in df['probing view downsample'].squeeze().unique().tolist():
            print(f'Aux_downsample: {Aux_downsample}')
            for metric in METRICS:
                sdf = df[df.scene.str.match(scene)]
                sdf = sdf.rename(columns={
                    'probing view downsample': 'Aux_downsample'
                })
                sdf = sdf[sdf.Aux_downsample == Aux_downsample]
                print(f'\t{metric}: mean = {sdf[metric].mean()}, std = {sdf[metric].std()}')


param = dict(DEFAULT_PARAM)
param['runTimeResDs'] = '.*'
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

selected = df[df.expDir.str.match(whichDir)]
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * 3 * N_SCENES

# selected = selected.rename(columns={
# 	'Aux downsample'
# })
scene_res_metric(selected)
'''
House:
Aux_downsample: 10
	PSNR: mean = 34.723968836750004, std = 3.9008815561973647
	SSIM: mean = 0.9798017358333333, std = 0.020886266763666048
	VMAF: mean = 0.78198295798375, std = 0.1157144020493767
Aux_downsample: 4
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
Aux_downsample: 5
	PSNR: mean = 34.793053785375, std = 3.9156632082009586
	SSIM: mean = 0.9800839944583333, std = 0.020610803643232315
	VMAF: mean = 0.7841509325991667, std = 0.10817064367418346
Bigroom:
Aux_downsample: 10
	PSNR: mean = 28.19623738220833, std = 2.14480021131607
	SSIM: mean = 0.9322579222083334, std = 0.021630050469902875
	VMAF: mean = 0.48069231673125, std = 0.11305996371086079
Aux_downsample: 4
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
Aux_downsample: 5
	PSNR: mean = 28.222978162416663, std = 2.1573364309432796
	SSIM: mean = 0.9329595614166666, std = 0.021469737596905975
	VMAF: mean = 0.4825618386900001, std = 0.11113460106455914
Smallroom:
Aux_downsample: 10
	PSNR: mean = 27.723691757125003, std = 2.776465201367884
	SSIM: mean = 0.9311620569583333, std = 0.043572661954351666
	VMAF: mean = 0.6098916364804167, std = 0.14990675738927425
Aux_downsample: 4
	PSNR: mean = 27.736233725541663, std = 2.7748232212338655
	SSIM: mean = 0.93164457025, std = 0.04407146910365807
	VMAF: mean = 0.6130295675799999, std = 0.14993997780637322
Aux_downsample: 5
	PSNR: mean = 27.73377974783333, std = 2.7801171780273717
	SSIM: mean = 0.9314034173333333, std = 0.04452119137882721
	VMAF: mean = 0.61279247344125, std = 0.1501926392880735
'''

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='probing view downsample')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/res_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='probing view downsample', 
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="res_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='probing view downsample')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/res_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='probing view downsample', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="res_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='probing view downsample')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/res_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='probing view downsample', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="res_PSNR")


# %%
# ---------------------------------------------------------------------------- #
#                                    plot a                                    #
# ---------------------------------------------------------------------------- #
# ! almost no effect
def scene_a_metric(df):
    for scene in SCENES:
        print(f'{scene}:')
        for aa in df.a.unique().tolist():
            print(f'a: {aa}')
            for metric in METRICS:
                sdf = df[df.scene.str.match(scene)]
                sdf = sdf[sdf.a == aa]
                print(f'\t{metric}: mean = {sdf[metric].mean()}, std = {sdf[metric].std()}')
                
param = dict(DEFAULT_PARAM)
param['a'] = '.*'
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

selected = df[df.expDir.str.match(whichDir)]
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * 3 * N_SCENES

scene_a_metric(selected)
'''
House:
a: 0.0001
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
a: 1e-05
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
a: 1e-06
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
Bigroom:
a: 0.0001
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
a: 1e-05
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
a: 1e-06
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
Smallroom:
a: 0.0001
	PSNR: mean = 27.736746099375, std = 2.7739752742870025
	SSIM: mean = 0.9316569558750001, std = 0.04406096264254564
	VMAF: mean = 0.61320507911375, std = 0.14990434821403595
a: 1e-05
	PSNR: mean = 27.736233725541663, std = 2.7748232212338655
	SSIM: mean = 0.93164457025, std = 0.04407146910365807
	VMAF: mean = 0.6130295675799999, std = 0.14993997780637322
a: 1e-06
	PSNR: mean = 27.73829115404167, std = 2.7760811342323732
	SSIM: mean = 0.9317872975, std = 0.04397567938889232
	VMAF: mean = 0.6133869095216667, std = 0.14989478364514056
'''

# ----------------------------------- SSIM ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='a')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/a_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='a', 
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="a_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='a')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/a_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='a', 
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="a_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='a')
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/a_PSNR.eps', bbox_inches = 'tight')
nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='a', 
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="a_PSNR")

#%%
# ---------------------------------------------------------------------------- #
#                               plot computation                               #
# ---------------------------------------------------------------------------- #
# ! almost no effect
def scene_maxNodes_metric(df):
    for scene in SCENES:
        print(f'{scene}:')
        for maxNodes in df.maxNodes.unique().tolist():
            print(f'maxNodes: {maxNodes}')
            for metric in METRICS:
                sdf = df[df.scene.str.match(scene)]
                sdf = sdf[sdf.maxNodes == maxNodes]
                print(f'\t{metric}: mean = {sdf[metric].mean()}, std = {sdf[metric].std()}')

param = dict(DEFAULT_PARAM)
param['maxNumNodes'] = '.*'
whichDir = generateExpDirName(**param)
whichDir = str(whichDir)
print(whichDir)

selected = df[df.expDir.str.match(whichDir)]
selected.reset_index(inplace=True, drop=True)
print(selected.shape)
assert selected.shape[0] == N_USERS * FPS * VIDEO_LEN * 3 * N_SCENES

scene_maxNodes_metric(selected)
'''
House:
maxNodes: 192
	PSNR: mean = 34.851729156416674, std = 3.923895326173498
	SSIM: mean = 0.9803037614583333, std = 0.02041938435394248
	VMAF: mean = 0.7869639885958333, std = 0.10406341879606283
maxNodes: 48
	PSNR: mean = 34.862379408500004, std = 3.8910387011798826
	SSIM: mean = 0.9804776825833332, std = 0.019896502577468164
	VMAF: mean = 0.7870482284425, std = 0.10438678363500051
maxNodes: 96
	PSNR: mean = 34.873247947416665, std = 3.9117174141151776
	SSIM: mean = 0.9804486762083334, std = 0.020027656918030916
	VMAF: mean = 0.7869431801725, std = 0.10406495880106374
Bigroom:
maxNodes: 192
	PSNR: mean = 28.224603377875, std = 2.159031391170039
	SSIM: mean = 0.9330041269166667, std = 0.0213498823276449
	VMAF: mean = 0.48300246376125, std = 0.11082631945107117
maxNodes: 48
	PSNR: mean = 28.221464453708332, std = 2.158552413357085
	SSIM: mean = 0.9329621832083334, std = 0.021352667494182225
	VMAF: mean = 0.4826222612891667, std = 0.11095050145910448
maxNodes: 96
	PSNR: mean = 28.22489536004167, std = 2.1590167006582104
	SSIM: mean = 0.9330116679583335, std = 0.021352361386409287
	VMAF: mean = 0.4830219034225, std = 0.11079778329723075
Smallroom:
maxNodes: 192
	PSNR: mean = 27.734762677208337, std = 2.7773795118952513
	SSIM: mean = 0.9314296500833334, std = 0.044126421671261254
	VMAF: mean = 0.6125941090091667, std = 0.15013330928001764
maxNodes: 48
	PSNR: mean = 27.73067512495833, std = 2.7766297942308205
	SSIM: mean = 0.931408775375, std = 0.044294877762605565
	VMAF: mean = 0.6123075482466667, std = 0.15061462266999484
maxNodes: 96
	PSNR: mean = 27.736233725541663, std = 2.7748232212338655
	SSIM: mean = 0.93164457025, std = 0.04407146910365807
	VMAF: mean = 0.6130295675799999, std = 0.14993997780637322
'''
# ----------------------------------- SSIM ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='SSIM', hue='maxNodes')
# plt.ylim(0.7, 1.0)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/comp_SSIM.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='SSIM', hue='maxNodes',
    ylim=(0.7,1.0),
    xlabel='Scene', ylabel='SSIM',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="comp_SSIM")

# ----------------------------------- VMAF ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='VMAF', hue='maxNodes')
# # plt.ylim(0.5, 1.0)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/comp_VMAF.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='VMAF', hue='maxNodes',
    xlabel='Scene', ylabel='VMAF',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="comp_VMAF")

# ----------------------------------- PSNR ----------------------------------- #
# ax = sns.barplot(data=selected, x='scene', y='PSNR', hue='maxNodes')
# plt.ylim(10, 40)
# sns.move_legend(ax, 'lower left')
# plt.savefig(f'{saveFigDir}/comp_PSNR.eps', bbox_inches = 'tight')

nmslplot.nmslBarPlot(
    df=selected, x='scene', y='PSNR', hue='maxNodes',
    ylim=(10,40),
    xlabel='Scene', ylabel='PSNR (dB)',
    loc='lower left', 
    savePlot=True, showPlot=False,
    saveDir=saveFigDir, saveImgName="comp_PSNR")

exit()
#%%
# traffic reduction
# data = pd.read_csv('encode_merge_separate.csv')
# data['group'] = data['group'].astype(int)
# data['viewId'] = data['viewId'].astype(int)
# data['bytes'] = data['bytes'].astype(int)
# data = data.sort_values(by=list(data.columns))
# for non-IXR cdd
# param = dict(DEFAULT_PARAM)
# param['m'] = '.*'
# param['h'] = '.*'
# print(param['placePolicy'])
# whichDir = generateCandidateDirName(**param)
# whichDir = str(whichDir)
# print(whichDir)
# print(f'data = {data.shape}')
# stat = {}
# for scene, iscene in zip(['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom'], ['House', 'Bigroom', 'Smallroom']):
#     d = {}    
#     sceneSelected = data[data.scene.str.match(scene)]
#     tv = sceneSelected[sceneSelected.type == 'tv']
#     policySelected = sceneSelected[sceneSelected.placePolicy == param['placePolicy']]
#     tex = policySelected[policySelected.type == 'texture']
#     dep = policySelected[policySelected.type == 'depth']
#     aux = policySelected[policySelected.type == 'aux']
#     d['0'] = {}
#     d['0']['tv'] = int(tv['bytes'].sum())
#     d['0']['tex'] = 0
#     d['0']['dep'] = 0
#     d['0']['aux'] = 0
#     for m in [0.01, 0.02, 0.03, 0.04, 0.05]:
#         mtex = tex[tex.m == m]
#         mdep = dep[dep.m == m]
#         maux = aux[aux.m == m]
#         print(f'm = {m}')
#         d[m] = {}
#         d[m]['tv'] = 0
#         d[m]['tex'] = 0
#         d[m]['dep'] = 0
#         d[m]['aux'] = 0
#         # print(tex[tex.m == m].shape[0], dep[dep.m == m].shape[0], aux[aux.m == m].shape[0])
#         pp = dict(DEFAULT_PARAM)
#         pp['scene'] = scene
#         pp['m'] = m
#         expDir = generateExpDirName(**pp)
#         with open(str(Path('..')/expDir/'Exp.json'), 'r') as f:
#             expJ = json.load(f)
#         sols = np.array(expJ['sols'])
#         nGroups = sols.shape[0]
#         for g in range(nGroups):
#             # accumulate tex, dep, and aux
#             nz = np.nonzero(sols[g])[0]
#             mask = np.zeros((sols[g].shape[0]))
#             gtex = mtex[mtex.group == g]
#             gdep = mdep[mdep.group == g]
#             gaux = maux[maux.group == g]
#             for i in nz:
#                 # find those views
#                 # !
#                 vtex = gtex[gtex.viewId == i]
#                 vdep = gdep[gdep.viewId == i]
#                 vaux = gaux[gaux.viewId == i]
#                 assert vtex.shape[0] == 1
#                 assert vdep.shape[0] == 1
#                 assert vaux.shape[0] == 1
#                 d[m]['tex'] += vtex.bytes.sum()
#                 d[m]['dep'] += vdep.bytes.sum()
#                 d[m]['aux'] += vaux.bytes.sum()
#         d[m]['tex'] = int(d[m]['tex'])
#         d[m]['dep'] = int(d[m]['dep'])
#         d[m]['aux'] = int(d[m]['aux'])
#         assert isinstance(d[m]['tex'], int)
#         assert isinstance(d[m]['dep'], int)
#         assert isinstance(d[m]['aux'], int)
#     # print(d)
#     stat[iscene] = d
# print(stat)
# with open('encode_merge_separate_stat.json', 'w') as f:
#     json.dump(stat, f)
#%%
# stack = tex - dep - aux - tv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# https://gist.github.com/ctokheim/6435202a1a880cfecd71

stat = json.load(open('encode_merge_separate_stat.json'))
# {group}_{stack}
# 6 x 3
scenes = ['House', 'Bigroom', 'Smallroom']
data = {}
ms = stat['House'].keys()
tps = ['tex', 'dep', 'aux', 'tv'] # types
# print(ms)
# print(tps)
for m in ms:
    data[m] = {}
    for tp in tps:
        data[m][tp] = [] # 3 scenes
        for scene in scenes:
            data[m][tp].append(stat[scene][m][tp] / 1e6 / 30 / N_USERS)
print(data)
'''
{'0': {'tex': [0.0, 0.0, 0.0], 'dep': [0.0, 0.0, 0.0], 'aux': [0.0, 0.0, 0.0], 'tv': [12.052356145833333, 13.135758827083333, 14.447326295833333]}, '0.01': {'tex': [0.18341903750000002, 0.18983192083333333, 0.2070913395833333], 'dep': [0.045084197916666666, 0.04825613958333333, 0.046935377083333334], 'aux': [0.004454925, 0.005357395833333333, 0.004947704166666666], 'tv': [0.0, 0.0, 0.0]}, '0.02': {'tex': [0.36526899791666667, 0.3813589125, 0.4155154833333333], 'dep': [0.09017103333333334, 0.09620914166666666, 0.09377902499999999], 'aux': [0.008994906249999999, 0.010677737500000001, 0.00996335625], 'tv': [0.0, 0.0, 0.0]}, '0.03': {'tex': [0.5469851083333332, 0.5731861208333334, 0.62545638125], 'dep': [0.13512209375, 0.14460206458333333, 0.14085742083333336], 'aux': [0.013418983333333334, 0.01615781875, 0.015019389583333332], 'tv': [0.0, 0.0, 0.0]}, '0.04': {'tex': [0.7304166770833334, 0.764081625, 0.8319984041666667], 'dep': [0.18020219583333336, 0.19341894791666667, 0.18779064583333333], 'aux': [0.01794797916666667, 0.021739320833333336, 0.020089091666666666], 'tv': [0.0, 0.0, 0.0]}, '0.05': {'tex': [0.91582698125, 0.9538710520833333, 1.0415600583333333], 'dep': [0.22597580208333332, 0.24216209791666668, 0.23467210416666667], 'aux': [0.02251355, 0.027337354166666668, 0.025120514583333333], 'tv': [0.0, 0.0, 0.0]}}
'''

# across all scenes, average
sums = {}
for m in ms:
    sums[m] = 0
    for tp in tps:
        for scene in scenes:
            sums[m] += stat[scene][m][tp] / 1e6 / 30 / N_USERS
print(sums)
'''
{'0': 39.63544126875, '0.01': 0.7353780374999997, '0.02': 1.47193859375, '0.03': 2.2108053812499997, '0.04': 2.9476848875000004, '0.05': 3.6890395145833335}
'''
for m in ms:
    print(f'{m}: ratio {sums[m] / sums["0"]}')
'''
0: ratio 1.0
0.01: ratio 0.01855354737982438
0.02: ratio 0.03713692964257544
0.03: ratio 0.05577849799273151
0.04: ratio 0.0743699273464141
0.05: ratio 0.09307426375222178
'''
# plt.rcParams['figure.dpi'] = 1000
with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.15
    epsilon = .015
    line_width = 1
    opacity = 0.7
    base_bar_positions = np.arange(len(scenes))
    bar_positions = {}
    for i, m in enumerate(ms):
        bar_positions[m] = base_bar_positions + i * bar_width
    
    colors = ['red', 'blue', 'green', 'gray']
    labels = ['Color View', 'Depth View', 'probing View', 'Target View']
    legendSet = False
    for m in ms:
        bottom = np.zeros((3,))
        for i, tp in enumerate(tps):
            if legendSet == False:
                plt.bar(bar_positions[m], data[m][tp], bar_width,
                    color=colors[i], label=labels[i], bottom=bottom)
                if i == len(tps) - 1:
                    legendSet = True
            else:
                plt.bar(bar_positions[m], data[m][tp], bar_width,
                    color=colors[i], bottom=bottom)
            bottom += data[m][tp]
    plt.xticks(bar_positions['0.03'], scenes)
    plt.ylabel('Average Traffic Per User (MB/s)')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.legend(bbox_to_anchor=(0.7, 1.0))
    sns.despine()
plt.savefig(f'{saveFigDir}/traffic.eps', bbox_inches = 'tight')
#%%
# plt.rcParams['figure.dpi'] = 800
tps = ['tex', 'dep', 'aux'] # types
with sns.axes_style("white"):
	sns.set_style("ticks")
	sns.set_context("talk")

	# plot details
	bar_width = 0.15
	epsilon = .015
	line_width = 1
	opacity = 0.7
	base_bar_positions = np.arange(len(scenes))
	bar_positions = {}
	for i, m in enumerate(ms):
		bar_positions[m] = base_bar_positions + i * bar_width

	colors = ['red', 'blue', 'green']
	labels = ['Color View', 'Depth View', 'probing View']
	legendSet = False
	for m in ms:
		bottom = np.zeros((3,))
		for i, tp in enumerate(tps):
			if legendSet == False:
				plt.bar(bar_positions[m], data[m][tp], bar_width,
					color=colors[i], label=labels[i], bottom=bottom)
				if i == len(tps) - 1:
					legendSet = True
			else:
				plt.bar(bar_positions[m], data[m][tp], bar_width,
					color=colors[i], bottom=bottom)
			bottom += data[m][tp]
	plt.xticks(bar_positions['0.03'], scenes)
	plt.ylabel('Average Traffic Per User (MB/s)')
	# plt.legend(loc='upper center', ncol=3, columnspacing=0.2)
	# plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
	plt.legend(bbox_to_anchor=(1.1, 1.2), ncol=3, columnspacing=0.2)
	# plt.tight_layout()
	sns.despine()
plt.savefig(f'{saveFigDir}/traffic_noTargetView.eps', bbox_inches = 'tight')
# plt.savefig(f'{saveFigDir}/traffic_noTargetView.eps')
# %%
metrics = {'opts': [], 'PSNR': [], 'SSIM': [], 'VMAF': []}
# for key in expJson:
#     if ('C2G' in key) or ('C2I' in key):
#         continue
#     print(key)
#     expData = df[df.expDir.str.match(key)]
#     for g in range(len(expJson[key]['opts'])):
#         # group g
#         groupData = expData.query(f'Frame >= {g*FPS} and Frame < {(g+1)*FPS}')
#         assert groupData.shape[0] == N_USERS * FPS
#         for m in metrics:
#             if m == 'opts':
#                 metrics[m].append(expJson[key][m][g])
#             else:
#                 metrics[m].append(groupData[m].mean())
# %%
# with open('obj_metrics.json', 'w') as f:
#     json.dump(metrics, f)
# %%
with open('obj_metrics.json', 'r') as f:
    obj_metricsJ = json.load(f)
for m in metrics:
    print(f'{m} = {len(obj_metricsJ[m])}')
# %%
# obj_metrics_Df = pd.DataFrame(data=obj_metricsJ)
# print(obj_metrics_Df['opts'].shape, obj_metrics_Df['SSIM'].shape)
# colors = ['r'] * obj_metrics_Df.shape[0] + ['g'] * obj_metrics_Df.shape[0]
# print(colors)
# ax = obj_metrics_Df.plot.scatter(x=['opts', 'opts'], y=['SSIM', 'VMAF'], color=colors, legend=False, label=['SSIM', 'VMAF'])
# # obj_metrics_Df.plot.scatter(x='opts', y='VMAF', c='g', label='VMAF', ax=ax)
# ax2  = ax.twinx()
# obj_metrics_Df.plot.scatter(x='opts', y='PSNR', color='b', ax=ax2, legend=False)
# ax.figure.legend()
# ax.set_title(f'Correlation')
# plt.show()
# %%
import scipy
'''
metric: opts (pearson, spearman)
 Pearson = 0.9999999999999994, Speareman = 1.0
metric: PSNR (pearson, spearman)
 Pearson = 0.41129588060822925, Speareman = 0.27068908856268353
metric: SSIM (pearson, spearman)
 Pearson = 0.6579910029237542, Speareman = 0.3964477516339741
metric: VMAF (pearson, spearman)
 Pearson = 0.5067454660355135, Speareman = 0.4772676584082712
'''
fig, ax = plt.subplots()
sz = 5
lns1 = ax.scatter(obj_metricsJ['opts'], obj_metricsJ['SSIM'], label='SSIM', color='r', s=sz)
lns2 = ax.scatter(obj_metricsJ['opts'], obj_metricsJ['VMAF'], label='VMAF', color='g', s=sz)
ax2 = ax.twinx()
lns3 = ax2.scatter(obj_metricsJ['opts'], obj_metricsJ['PSNR'], label='PSNR', color='b', s=sz)
lns = [lns1, lns2, lns3]
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='best')
ax.set_xlabel('g value')
ax.set_ylabel('SSIM / VMAF')
ax2.set_ylabel('PSNR')
for m in metrics:
    pearsonr = scipy.stats.pearsonr(obj_metricsJ['opts'], obj_metricsJ[m])
    spearmanr = scipy.stats.spearmanr(obj_metricsJ['opts'], obj_metricsJ[m])
    print(f'metric: {m} (pearson, spearman) \n Pearson = {pearsonr.statistic}, Speareman = {spearmanr.correlation}')
plt.title('Quality Metrics - g Value')
plt.savefig(f'{saveFigDir}/correlation.eps', bbox_inches = 'tight')
# %%
expJ = json.load(open('qual_merge.json', 'r'))
RAW_SCENES = ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSamllroom']
SCENE_MAPPING = {
    'FurnishedCabin': 'House',
    'ScifiTraceBigroom': 'Bigroom',
    'ScifiTraceSmallroom': 'Smallroom'
}
MS = ['0.01', '0.02', '0.03', '0.04', '0.05']
param = dict(DEFAULT_PARAM)
param['h'] = '0.15'
d = {}
for m in MS:
    d[m] = {'House': {}}
    for raw in SCENE_MAPPING:
        param['m'] = m
        param['scene'] = raw
        whichDir = generateExpDirName(**param)
        whichDir = str(whichDir)
        assert whichDir in expJ
        d[m][SCENE_MAPPING[raw]] = {'Candidate Generator': [], 'Coverage Estimator': [], 'Solver': []}
        d[m][SCENE_MAPPING[raw]]['Candidate Generator'] = (np.array(expJ[whichDir]['timeSplit']) + np.array(expJ[whichDir]['timeCdd'])).tolist()
        d[m][SCENE_MAPPING[raw]]['Coverage Estimator'] = expJ[whichDir]['timeEsted']
        d[m][SCENE_MAPPING[raw]]['Solver'] = expJ[whichDir]['timePlace']
with open('stat_time.json', 'w') as f:
    json.dump(d, f)
# %%
# stack = cdd time, cvg time, solver
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
stat = json.load(open('stat_time.json', 'r'))
scenes = ['House', 'Bigroom', 'Smallroom']
labels = ['Candidate Generator', 'Coverage Estimator', 'Solver']

data = {}
for m in MS:
    data[m] = {}
    for label in labels:
        data[m][label] = [] # 3 scenes
        for scene in scenes:
            data[m][label].append(np.mean(stat[m][scene][label]))
print(data)
'''
{'0.01': {'Candidate Generator': [0.007291666666666667, 0.0067708333333333336, 0.004166666666666667], 'Coverage Estimator': [12.6734375, 12.388020833333334, 11.847395833333334], 'Solver': [16.296875, 18.053645833333334, 17.3890625]}, '0.02': {'Candidate Generator': [0.0020833333333333333, 0.0020833333333333333, 0.0026041666666666665], 'Coverage Estimator': [12.753645833333334, 12.4421875, 11.7671875], 'Solver': [15.968229166666667, 17.467708333333334, 16.64375]}, '0.03': {'Candidate Generator': [0.004166666666666667, 0.008333333333333333, 0.005729166666666666], 'Coverage Estimator': [30.442708333333332, 30.4203125, 28.7671875], 'Solver': [37.97708333333333, 43.11302083333333, 41.81614583333333]}, '0.04': {'Candidate Generator': [0.0078125, 0.011458333333333333, 0.00625], 'Coverage Estimator': [61.63489583333333, 60.609375, 58.8734375], 'Solver': [76.97708333333334, 78.20729166666666, 82.95416666666667]}, '0.05': {'Candidate Generator': [0.008854166666666666, 0.0067708333333333336, 0.009895833333333333], 'Coverage Estimator': [122.51979166666666, 120.5359375, 116.96041666666666], 'Solver': [150.33020833333333, 149.99895833333332, 148.03385416666666]}}
'''
# plt.rcParams['figure.dpi'] = 1000
plt.figure(figsize=(10, 6))
with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.15
    epsilon = .015
    line_width = 1
    opacity = 0.7
    base_bar_positions = np.arange(len(scenes))
    bar_positions = {}
    for i, m in enumerate(MS):
        bar_positions[m] = base_bar_positions + i * bar_width
    
    colors = ['red', 'blue', 'green']
    legendSet = False
    for m in MS:
        bottom = np.zeros((3,))
        for i, tp in enumerate(labels):
            if legendSet == False:
                plt.bar(bar_positions[m], data[m][tp], bar_width,
                    color=colors[i], label=labels[i], bottom=bottom)
                if i == len(labels) - 1:
                    legendSet = True
            else:
                plt.bar(bar_positions[m], data[m][tp], bar_width,
                    color=colors[i], bottom=bottom)
            bottom += data[m][tp]
    plt.xticks(bar_positions['0.03'], scenes)
    plt.ylabel('Runtime for Single Update (sec)')
    plt.legend(bbox_to_anchor=(1.0, 1.15), ncol=3, columnspacing=0.2)
    # plt.legend(loc='upper left')
    # sns.despine()
plt.savefig(f'{saveFigDir}/time_comp.eps', bbox_inches = 'tight')
# %%
def draw_line_chart2(bar_names_list, values_dict, error_bar_dict, legend_loc, y_label_name, x_label_name, output_name, my_format):
    plt.figure(figsize=my_figsize, dpi=100, linewidth=1)
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rc('figure', facecolor='w')


    for idx, bar in enumerate(bar_names_list):
        x = np.arange(len(values_dict[bar]))
        # plt.plot(x, values_dict[bar], color[idx], label=bar)
        # print(error_bar_dict[bar]["bottom"])
        # print(error_bar_dict[bar]["top"])
        plt.errorbar(x=[1, 2, 4, 8, 16], y=values_dict[bar], fmt=color_dict[bar], label=method_translate[bar], yerr=[error_bar_dict[bar]["bottom"], error_bar_dict[bar]["top"]],
         linewidth=line_width, linestyle=linestyle_dict[bar], marker=marker_dict[bar], markersize=40)
        # plt.errorbar(x=[1, 2, 4, 8, 16], y=values_dict[bar], fmt=color_dict[bar], label=bar, 
        #  linewidth=line_width, linestyle=linestyle_dict[bar], marker=marker_dict[bar], markersize=50)

    plt.xlabel(x_label_name, fontsize=my_fontsize)
    plt.ylabel(y_label_name, fontsize=my_fontsize)

    plt.xticks(fontsize=my_fontsize * tick_coe)
    plt.yticks(fontsize=my_fontsize * tick_coe)
    
    ax = plt.gca()
    ax.set_ylim(ymin=0)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=6)

    plt.legend(loc=legend_loc, fancybox=False, labelspacing=0.05, handletextpad=0.5, ncol=4,
             title="", framealpha=1, columnspacing=0.2, fontsize=my_fontsize)
    plt.tight_layout()

    plt.savefig(f"{output_name}.{my_format}",
                format=my_format)
    plt.close()