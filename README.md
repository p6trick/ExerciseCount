# ExerciseCount


### Dataset Structure
```bash
dataset
├── csv
│   ├── left
│   │   ├── BB1.csv
│   │   ├── BB2.csv
│   │   ├── BB4.csv
│   │   ├── BB6.csv
│   │   ├── BS3.csv
│   │   ├── BS4.csv
│   │   ├── BS5.csv
│   │   ├── BS6.csv
│   │   └── BS7.csv
│   └── right
│       ├── BB1.csv
│       ├── BB2.csv
│       ├── BB4.csv
│       ├── BB6.csv
│       ├── BS3.csv
│       ├── BS4.csv
│       ├── BS5.csv
│       ├── BS6.csv
│       └── BS7.csv
└── data
    ├── left
    │   └── not_patient
    │       ├── BB1
    │       │   ├── BB1.yaml
    │       │   └── video
    │       │       └── BB1_excercise_videos.mp4
    │       ├── BB2
    │       │   ├── BB2.yaml
    │       │   └── video
    │       │       └── BB2_excercise_videos.mp4
    │       ├── BB4
    │       │   ├── BB4.yaml
    │       │   └── video
    │       │       └── BB4_excercise_videos.mp4
    │       ├── BB6
    │       │   ├── BB6.yaml
    │       │   └── video
    │       │       └── BB6_excercise_videos.mp4
    │       ├── BS3
    │       │   ├── BS3.yaml
    │       │   └── video
    │       │       └── BS3_excercise_videos.mp4
    │       ├── BS4
    │       │   ├── BS4.yaml
    │       │   └── video
    │       │       └── BS4_excercise_videos.mp4
    │       ├── BS5
    │       │   ├── BS5.yaml
    │       │   └── video
    │       │       └── BS5_excercise_videos.mp4
    │       ├── BS6
    │       │   ├── BS6.yaml
    │       │   └── video
    │       │       └── BS6_excercise_videos.mp4
    │       └── BS7
    │           ├── BS7.yaml
    │           └── video
    │                └── BS7_excercise_videos.mp4
    └── right
        └── not_patient
            ├── BB1
            │   ├── BB1.yaml
            │   └── video
            │       └── BB1_excercise_videos.mp4
            ├── BB2
            │   ├── BB2.yaml
            │   └── video
            │       └── BB2_excercise_videos.mp4
            ├── BB4
            │   ├── BB4.yaml
            │   └── video
            │       └── BB4_excercise_videos.mp4
            ├── BB6
            │   ├── BB6.yaml
            │   └── video
            │       └── BB6_excercise_videos.mp4
            ├── BS3
            │   ├── BS-3.yaml
            │   └── video
            │       └── BS3_excercise_videos.mp4
            ├── BS4
            │   ├── BS-4.yaml
            │   └── video
            │       └── BS4_excercise_videos.mp4
            ├── BS5
            │   ├── BS-5.yaml
            │   └── video
            │       └── BS5_excercise_videos.mp4
            ├── BS6
            │   ├── BS-6.yaml
            │   └── video
            │       └── BS6_excercise_videos.mp4
            └── BS7
                ├── BS-7.yaml
                └── video
                    └── BS7_excercise_videos.mp4
```
### install
```bash
virtualenv "envname" --python=python3.8
source "envname"/bin/activate

pip install matplotlib
pip install Pillow==8.1.0
pip install torch --> cuda version check!
pip install tqdm
pip install numpy==1.23.0
pip install mediapipe
pip install requests
pip install pandas
pip install pyyaml
```
### Testing all activity
```bash
python video_test_dir.py --data_dir "dataset_path" --type "patient_or_not_patient" --direction "left_or_right" \\
--result "result_folder_name" --en_th "enter_threshold_float" --ex_th "exit_threshold_float" \\
--max_dis "max_distance_int" --mean_dis "mean_distance_int" --csv_ver "version"

```
### Example
- type: <span style="color:red"> patient
- direction: <span style="color:red">right
- enter_threshold: <span style="color:red">9.9
- exit_threshold: <span style="color:red">0.1
- max_distance: <span style="color:red">20
- mean_distance: <span style="color:red">10
- csv_ver: <span style="color:red">0
```bash
python video_test_dir.py --data_dir ./dataset --type patient --direction right --result ./result --en_th 9.9 --ex_th 0.1 --max_dis 20 --mean_dis 10 --csv_ver 0
```
### Testing each activity
```bash
python video_test_dir.py --data_dir "dataset_path" --type "patient_or_not_patient" --direction "left_or_right" \\
--result "result_folder_name" --en_th "enter_threshold_float" --ex_th "exit_threshold_float" \\
--max_dis "max_distance_int" --mean_dis "mean_distance_int" --csv_ver "version" --act_name "activity name"

```
### Example
- type: <span style="color:red"> patient
- direction: <span style="color:red">right
- enter_threshold: <span style="color:red">9.9
- exit_threshold: <span style="color:red">0.1
- max_distance: <span style="color:red">20
- mean_distance: <span style="color:red">10
- csv_ver: <span style="color:red">0
- act_name: <span style="color:red">BB1
```bash
python video_test_dir.py --data_dir ./dataset --type patient --direction right --result ./result --en_th 9.9 --ex_th 0.1 --max_dis 20 --mean_dis 10 --csv_ver 0 --act_name BB1
```

