# Workout App

Workout app with pose estimation

Use ```pip install -r mpipe_requirements.txt``` to install all required packages

Examples of usage:

* test camera
```shell
python mp_workout.py --test
```

* workout **DUMBBELL LATERAL RAISE**
  - visualize with symmetry mode and full skeleton

```shell
python mp_workout.py --mode symmetry --skeleton 1 --config dumbbell_lateral_raise
```

* workout **DUMBBELL SHOULDER PRESS**
  - visualize with gravity-center mode without skeleton
```shell
python mp_workout.py --mode gravity-center --skeleton 0
```

* *by default* - workout **DUMBBELL SHOULDER PRESS**
  - visualize with headless skeleton
```shell
python mp_workout.py 
```
### Example of different visualization parameters:
1. ```--mode symmetry --skeleton 1```
   
2. ```--mode gravity_center --skeleton 2```

4. ```--mode angles --skeleton 0```

5. ```--mode angles --skeleton 2```

![img](images/visual_modes.jpg)

### Options
The program runs as a command-line script. Below you can see the list of available options. You can always go back to them using the --help flag.
```bash
  -h, --help            show this help message and exit
  
  # GENERAL PARAMETERS
  --save_video          Save video file (result.mp4 - by default).
  --path                Path to video file.
  
  # VISUALISATION SETTINGS
  --skeleton            If set to 0 shows only keypoints, 1 - default, 2 - headless.
  --mode                Mode: [keypoints_names, angles, symmetry].
  --scale               Set scale parameter.
  
  # DEBUG OPTIONS
  --test                Test camera
  --debug               Set logging mode to "debug".

```
