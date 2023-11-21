# A Blind Streaming System for Multi-client Online 6-DoF View Touring
---
## Requirements
* **Windows** is recommended though porting is possible
---
## Technical Detail
* Frame system
    * Common description
        * a (7,) pose always represent (x, y, z, qx, qy, qz, qw)
        * ```obj.summary()``` is used to generate a json file describing its behavior
    * Unity frame: (-1, 7)
        * Recorded directly from MIV_XROrigin, loaded by PoseFeeder
    * O3d frame = Open3D frame:
        * Anchored at Open3D coordinate system
    * Conversion between frames are listed in Common.py
* Blind streaming
    * CamPlace.py: generate candidates
    * diffCalc.m: calculate derivatives given the equation for optimal number of candidates
    * Exp.py: core Exp class, implement most of the details
    * ExpMain.py: switch script for setting different type of works/experiment for Exp class
    * Yuv.py: Yuv class for fast data retrieval
    * Exp_miv_util.py: inherit from old Exp_miv_util file
        * contains some utility functions for trans-coding/pose truncation
    * Exp_unitTest.py: for testing each function, code not organized
    * Exp_unitTest_Runner.py: running a specific unitTest function
    * Exp_unitTest_unitTest_ComplextAngleMean.py: not developement, for complext angle mean (replaced by average quaternion)
* Experiment scripts
    * Exp_AllCadidndates.py: Cmd line tool to generate/prepare(convert to YUV)/prune(Reduce depth file size) candidates
    * Exp_ByWorks_IXR.py: Run experiments for IXR candidate generators and all solvers
    * Exp_ByWorks.py: Run experiments for MM candidate generator and all solvers
    * Exp_Gen.py: Default parameters
    * Exp_filterPureExp.py: move pure exp directory to another directory
    * Exp_truncatePose.py: truncate poses
    * Exp_encodeSourceView.py: encode source view in different ways
    * Exp_encodeSummary.py: collect encoded data to statistic files
        * encode_merge.csv:
            * not used in paper
        * encode_merge_separate.csv:
            * ```,scene,cddDir,type,group,viewId,bytes,path,cspPolicy,windowSize,psPolicy,m,h,placePolicy```
    * vmaf.exe: for running vmaf and objective metrics
    * RVS.exe: complied binary for RVS synthesizer
* Paper
    * Paper_number_check.py: check paper numbers are correct
    * Paper_disocclusion_gen.py: visualize disocclusion and generate its figures
    * Paper_figGen.py: generate paper figures
        * encode_merge_separate_stat.json:
        * example_config_file_rvs.json: example configuration file for RVS synthesizer
        * obj_metrics.json: j[quality metrics][group ount] for ploting correlation between optimization values and metrics
        * stat_time.json: statistic for component runtime
* TMM unit test
    * TMM_MIV_TestSequence_Gen.py: generate coarse pose for MIV test
    * TMM_MIV_TestSequence_Convert.py: generate source views for MIV testb
    * TMM_MIV_TestSequence_DepthVerify.py: visualize depth from unity
* Others are self-explanatory

---

## Directory Structure
* Unity project
    * TMM_URP
* Paper related
    * disocculusion_gen_dir: generate disocclusion example figure
    * figs: figures used in paper
    * nmslplot: code by YC for modified plots
    * obj: 3D meshes exported from Unity
    * Reconstruction: reconstructed scene after Structure-from-Motion
* Code related
    * Trace_Raw: complete trace collected from 16 subjects (truncate them before use)
    * \*.py: see the document inside each of them
---
## Program Setup
1. ```$ svn co svn://snoopy.cs.nthu.edu.tw/ext/vr/doc_TMM22_Code```
2. Install conda (suggest cuda 11.7)
    * Download the [installer](https://developer.nvidia.com/cuda-11-7-0-download-archive)
3. Create conda environment in *Anaconda Prompt*
    1. ```$ conda env create -f windows_env.yml```
    2. **Remove torch-associated packages** after dump conda environment to a file
    3. ```conda activate tmm``` before running any code
    4. Install [pytorch](https://pytorch.org/) (suggest 1.13.1) manually (choose correct platform and cuda version)
4. Download [VMAF](https://github.com/Netflix/vmaf/releases) releases
5. Install **RVS**
    1. Clone from SM's [fork](https://github.com/ShengMingTang/RVS.git) instead of [original one](https://gitlab.com/mpeg-i-visual/rvs)
        * Remove duplicated file I/O for number of input frame = 1
    1. Install OpenCV
        1. Download [OpenCV](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/) for Windows
        3. Let **{opencv}** be the directory containing **OpenCVConfig.cmake**, add **{opencv}/build** to **PATH**
    2. Install Glm
        1. Download [glm](https://sourceforge.net/projects/glm.mirror/), and extract it to a directory **{dirGlm}**
        2. add ***{dirGlm}*/cmake** to **PATH**'
    3. Install Visual Studio (VS)
        1. Install Visual studio installer, remember to check **C++related** packages
    3. Change to the cloned RVS github directory, and open a **Developer Command Prompt for Windows** terminal
    4. ```mkdir build ; cd build```
    5. ```cmake -DWITH_OPENGL=On ..```
    6. Check output from ```$ cmake -DWITH_OPENGL=On ..```
        * Runtime library RVS is going to use F(from output):
            * ```-- OpenCV ARCH: XXX ; -- OpenCV RUNTIME: vcXX ;```
        * If Cmake complains that **OpenCV does not match your computer architecture**
            1. Download OpenCV [source code](https://github.com/opencv/opencv.git), called **{OpenCVGit}**
            2. ```cd {OpenCVGit} ; mkdir build ; cd build ; cmake ..```
            3. Open OpenCV.sln and build
            4. Add **{OpenCVGit}/build** to **PATH**
    6. Open **build/RVS.sln** and click build in VS
    7. Run **{buildType}/RVSUnitTest.exe** to unit test
        * If it complains *XXX.dll not found*, check if the associated file is in **PATH**
    7. Copy **{buildType}/RVS.exe** to the TMM project folder
1. Run any *.py in a **Anaconda Prompt** 
1. Truncate pose trajectories to a directory
1. Prepare for ground truth novel view
---
## Trajectory Collection Setup
1. Set Up XR/VR Environment
    > Do not create an empty project and make it a VR one. Follow the template to save time for other settings
    1. [Download XR Template](https://docs.unity3d.com/Manual/xr-template-vr.html) 
    2. [Setup XR project (ep. 1)](https://www.youtube.com/watch?v=gGYtahQjmWQ)
        * [How to Turn on Preview Package Access](https://medium.com/@jeffreymlynch/where-are-the-missing-preview-packages-in-unity-2020-3ad0935e4193#:~:text=Once%20there%2C%20check%20Enable%20Preview%20Packages)
            * This is mandotory for using XR / XR Interaction Toolkit
        * [How to Install a Unity Registry](https://docs.unity3d.com/2022.1/Documentation/Manual/upm-ui-install.html)
    5.  [Install XR Interaction Toolkit](https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@2.0/manual/installation.html)
    6. [Install SteamVR plugin](https://valvesoftware.github.io/steamvr_unity_plugin/articles/Quickstart.html)
        * Otherwise Unity will not be triggered in SteamVR
    1. Drag **TMM_URP/Asset/Prefabs/MIV_XROrigin** to capture the hmd motion
        * Set **outputCsv** to where the output should be written
    2. Drag **TMM_URP/Asset/Prefabs/MIV_XROriginVerify** to verify the hmd motion
        * Set **inputCsv** to where the **outputCsv**
---
## Unity Setup
1. Download assets from [Unity Asset Store](https://assetstore.unity.com/)
    * Free Sci-Fi Office Pack, Furnished Cabin
2. [Change display resolution](https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/PerceptionCamera.md#output-resolution)
    * Default 960x540
3. Drag **TMM_URP/Asset/Prefabs/MIV_Main** into the scene to capture the source/novel views
    * Set **LoadTracePrefix** to input directory that has **{cameraNamePrefix}_{count}.csv**
    * Check **isCapDepth** if depth is needed
        * Depth is in raw format (HxWx4) floating points, mapping [0, 1000] to [0, 1]
    * Set **outputDir** to where the rgb and depth files are going to be stored
4. [Obj Exporter](https://assetstore.unity.com/packages/tools/utilities/scene-obj-exporter-22250)
    * If the scene has any **prefabs or prefab variants**, right click > prefab > **unpack completely** before exporting
---

## Scenes
* ScifiOfficeLite (Test pass)
    * Two small rooms
    * Demo
        * position = (2.35, 1, 35) (center of the bigger room)
        * position = (2.35, 1, 45) (center of the smaller room)
* FurnishedCabin (Test pass)
    * Living room, bedroom, toilet (simple texture)
    * Demo
        * position = (-1, 1, -2)
---
## Sturcture from Motion (SfM)
* [github](https://github.com/colmap/colmap)
    * Follow the instruction in github
* Reconstruct_CollectSV.py: collect source views for reconstruction
* Reconstruct_Trans.py: transform reconstructed point cloud to align with ground truth
---
## Appendix (Backup)
* [Using XR to record motion (ep. 4)](https://www.youtube.com/watch?v=5NRTT8Tbmoc)
    * Replace XRRig with XROrigin (newer SDK)
* [XROrigin](https://docs.unity3d.com/Packages/com.unity.xr.core-utils@2.0/manual/xr-origin.html)
* [Target framerate](https://docs.unity3d.com/ScriptReference/Application-targetFrameRate.html)
* Use **Exp.makeTargetViews()** to convert pngs to yuv


