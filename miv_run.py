'''
Example file for running tmiv
'''
import os

TMIV_RENDERER_PATH = '/home/shengming/test_miv/tmiv_install/bin/Renderer'
numSamplesPerWindow = 47
sv_file_path = 'SourceView'
DEFAULT_SYNTHESIZER = 'ViewWeightingSynthesizer_NoInpainter'
output_file_path = 'output'
miv_pose_file_path = 'TargetView'
filename = 'pose0'
log_file_path = 'log'

os.system(
    f"{TMIV_RENDERER_PATH} \
    -n 1 -N {numSamplesPerWindow} -s {sv_file_path} -f 0 -r rec_0 -P p01 \
    -c ./config/TMIV_{DEFAULT_SYNTHESIZER}_renderer_config.json \
    -p inputDirectory ./ \
    -p outputDirectory {output_file_path} \
    -p configDirectory ./config \
    -p inputSequenceConfigPathFmt {sv_file_path}/miv.json \
    -p inputViewportParamsPathFmt ../{sv_file_path}/miv.json \
    -p inputPoseTracePathFmt ../{miv_pose_file_path}/miv_pose0.csv \
    -p outputViewportGeometryPathFmt {filename}_depth_1920x1080_yuv420p16le.yuv \
    -p outputViewportTexturePathFmt {filename}_texture_1920x1080_yuv420p10le.yuv \
    > {log_file_path}/{filename}.log 2>&1"
)