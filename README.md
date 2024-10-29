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
            │   ├── BS-4 팔 앞으로 올리기 스트레칭.yaml
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
```
virtualenv "envname" --python=python3.8
source "envname"/bin/activate

pip install matplotlib
pip install Pillow==8.1.0
pip install torch --> cuda version check!
pip install tqdm
pip install numpy==1.23.0
pip install mediapipe
```

