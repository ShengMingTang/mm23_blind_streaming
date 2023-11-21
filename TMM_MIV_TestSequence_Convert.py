from miv_util import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='test sequence directory')
    args = parser.parse_args()

    f2d = fDepthPlannarFactory(1000)
    root = Path(args.dir)
    generateTMIVInputsStaticSceneFromDir(root/'SourceView', 'helloworld', f2d)
    generateTMIVUsersStaticSceneFromDir(root/'TargetView')