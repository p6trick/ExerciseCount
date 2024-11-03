import cv2
import argparse
import os
from mediapipe.python.solutions import pose as mp_pose
from pose_utils import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing, RepetitionCounter, PoseClassificationVisualizer
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import pandas as pd
from glob import glob
import yaml
import json

def init_dict(args):
    result_dict = {}
    result_dict['Information'] = {
        "direction":args.direction,
        "type":args.type,
        "Enther threshold": args.en_th,
        "Exit threshold": args.ex_th,
        "Max distance": args.max_dis,
        "Mean distance": args.mean_dis
    }
    total_result = {}
    return result_dict, total_result

def log_start(cat_name, args):
    print("="*60)
    print(f'Activity name: {cat_name.split("/")[-1]}')
    print(f'Enter threshold: {args.en_th} || Exit threshold: {args.ex_th}')
    print(f'Max distance: {args.max_dis} || Mean distance: {args.mean_dis}')
    print("-"*60)

def make_csv(args, activity_name):
    if not os.path.isdir(os.path.join(f'./{args.result}', f'{args.type}_{args.direction}')): 
        os.mkdir(os.path.join(f'./{args.result}', f'{args.type}_{args.direction}'))
    csv_path = os.path.join(f'./{args.result}', f'{args.type}_{args.direction}') # ./result/patient_right

    df = pd.read_csv(os.path.join(args.data_dir,'csv',args.csv_ver,args.direction,f'{activity_name}.csv'), header=None) # ./csv/right/BB1.csv
    csv_A = df.loc[df[1] == 'A'].loc[:,df.loc[df[1] == 'A'].columns != 1]
    csv_B = df.loc[df[1] == 'B'].loc[:,df.loc[df[1] == 'B'].columns != 1]

    if not os.path.isdir(os.path.join(csv_path, activity_name)): # patient_right/BB1/A.csv B.csv
        os.mkdir(os.path.join(csv_path, activity_name))
    csv_A.to_csv(os.path.join(csv_path,activity_name,"A.csv"), header=None, index=False, index_label=False)
    csv_B.to_csv(os.path.join(csv_path,activity_name,"B.csv"), header=None, index=False, index_label=False)

    return csv_path


def count(result_dict, activity_name, video_path, label_idx):
    all_count = 0
    all_label = 0
    result_dict['Activity'][activity_name] = []
    for vid,la in zip(video_path,label_idx): ## BB1/abc.mp4, BB1.yaml
        ######################################################################################
        A_repetitions_count = None
        B_repetitions_count = None
        pred_count = 0
        full_count = 0
        
        A_cnt = False
        B_cnt = False
        cnt = 0
        txt = ""

        a_flag = 0
        a_chk = 0

        b_flag = 0
        b_chk = 0
        ######################################################################################
        full_count = label[la]
        
        print(f'Video name: {vid.split("/")[-1]}')

        video_cap = cv2.VideoCapture(vid)

        # Get some video parameters to generate output video with classificaiton.
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_samples_folder = os.path.join(csv_path, activity_name)

        # Initialize tracker.
        pose_tracker = mp_pose.Pose()
        
        # Initialize embedder.
        pose_embedder = FullBodyPoseEmbedder()

        pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=args.max_dis,
        top_n_by_mean_distance=args.mean_dis)

        # Initialize EMA smoothing.
        pose_classification_filter = EMADictSmoothing(
            window_size=10,
            alpha=0.2)

        ######################################################################################
        # Initialize counter.
        A_repetition_counter = RepetitionCounter(
            class_name='A',
            enter_threshold=args.en_th,
            exit_threshold=args.ex_th)

        # Initialize renderer.
        A_pose_classification_visualizer = PoseClassificationVisualizer(
            class_name='A',
            plot_x_max=video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10)
        ######################################################################################
        # Initialize counter.
        B_repetition_counter = RepetitionCounter(
            class_name='B',
            enter_threshold=args.en_th,
            exit_threshold=args.ex_th)

        # Initialize renderer.
        B_pose_classification_visualizer = PoseClassificationVisualizer(
            class_name='B',
            plot_x_max=video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10)
        ######################################################################################

        # Open output video.
        vid_save_path = os.path.join(csv_path,activity_name,'result_video')
        if not os.path.isdir(vid_save_path):
            os.mkdir(vid_save_path)
        out_video = cv2.VideoWriter(os.path.join(vid_save_path,f'{vid.split("/")[-1]}'), cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

        frame_idx = 0
        output_frame = None

        with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
            while True:
                # Get next frame of the video.
                success, input_frame = video_cap.read()
                if not success:
                    break

                # Run pose tracker.
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                result = pose_tracker.process(image=input_frame)
                pose_landmarks = result.pose_landmarks

                # Draw pose prediction.
                output_frame = input_frame.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=output_frame,
                        landmark_list=pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)
                if A_cnt == False and B_cnt == False:
                    if pose_landmarks is not None:
                    # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                    for lmk in pose_landmarks.landmark], dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                        # Classify the pose on the current frame.
                        pose_classification = pose_classifier(pose_landmarks)
                        
                        # Smooth classification using EMA.
                        pose_classification_filtered = pose_classification_filter(pose_classification)

                        # Count repetitions.
                        A_repetitions_count = A_repetition_counter(pose_classification_filtered)
                        a_flag = A_repetitions_count

                        if a_flag != a_chk:
                            a_chk = a_flag
                            A_cnt = True
                            txt += "(A-"
                
                    else:
                        # No pose => no classification on current frame.
                        pose_classification = None

                        # Still add empty classification to the filter to maintaing correct
                        # smoothing for future frames.
                        pose_classification_filtered = pose_classification_filter(dict())
                        pose_classification_filtered = None

                        # Don't update the counter presuming that person is 'frozen'. Just
                        # take the latest repetitions count.
                        A_repetitions_count = A_repetition_counter.n_repeats

                elif A_cnt == True and B_cnt == False:
                    if pose_landmarks is not None:
                    # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                    for lmk in pose_landmarks.landmark], dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                        # Classify the pose on the current frame.
                        pose_classification = pose_classifier(pose_landmarks)
                        
                        # Smooth classification using EMA.
                        pose_classification_filtered = pose_classification_filter(pose_classification)

                        # Count repetitions.
                        B_repetitions_count = B_repetition_counter(pose_classification_filtered)
                        b_flag = B_repetitions_count
                        # print(f'B: {B_repetitions_count}')
                        # print(f'b flag: {b_flag}')
                        if b_flag != b_chk:
                            b_chk = b_flag
                            B_cnt = True
                            txt += "B)"
            
                    else:
                        # No pose => no classification on current frame.
                        pose_classification = None

                        # Still add empty classification to the filter to maintaing correct
                        # smoothing for future frames.
                        pose_classification_filtered = pose_classification_filter(dict())
                        pose_classification_filtered = None

                        # Don't update the counter presuming that person is 'frozen'. Just
                        # take the latest repetitions count.
                        B_repetitions_count = B_repetition_counter.n_repeats
                    # if B_repetitions_count:
                    #     B_cnt = True
                    #     txt += "B)"
                
                elif (A_cnt == True) and (B_cnt == True):
                    # print(A_cnt, B_cnt)
                    cnt += 1
                    A_cnt = False
                    B_cnt = False
        
                # Draw classification plot and repetition counter.
                output_frame = A_pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=pose_classification_filtered,
                    repetitions_count=A_repetitions_count)

                # Save the output frame.
                out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                # Show intermediate frames of the video to track progress.
                # if frame_idx % 50 == 0:
                #   show_image(output_frame)

                frame_idx += 1
                pbar.update()        

        # Close output video.
        out_video.release()

        # Release MediaPipe resources.
        pose_tracker.close()

        
        vid_name = vid.split("/")[-1]
        result_dict['Activity'][activity_name].append({
                    'video_name': vid_name,
                    'full_count': full_count,
                    'pred_count': cnt,
                    'count_log': txt ,
                    'Acc': (cnt/full_count)*100,
                    
                })
        all_count += cnt
        all_label += full_count
        
        print(f'Total counts: {cnt}')
        print(f'A counts: {A_repetitions_count}')
        print(f'B counts: {B_repetitions_count}')
        print(f'count log: {txt}')
        print(f'Acc: {(cnt/full_count)*100:.2f}%')
        print("-"*60)
    return result_dict, all_count, all_label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to dataset dir", default="./dataset")
    parser.add_argument("--direction", type=str, help="right or left", default="right")
    parser.add_argument("--type", type=str, help="patient or not_patient", default="patient")
    parser.add_argument("--result", type=str, help="Save folder name", default="./result")
    parser.add_argument("--en_th", type=float, help="Enter threshold", default=5.0)
    parser.add_argument("--ex_th", type=float, help="Exit threshold", default=2.0)
    parser.add_argument("--max_dis", type=int, help="Max distance", default=20)
    parser.add_argument("--mean_dis", type=int, help="Mean distance", default=10)
    parser.add_argument("--act_name", type=str, help="Activity name", default='')
    parser.add_argument("--csv_ver", type=str, help="Csv version", default='0')


    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), 'dataset 폴더 경로가 잘못되었습니다.'
    assert os.path.isdir(os.path.join(args.data_dir, 'data',args.direction)), '운동 방향 폴더 경로가 잘못되었습니다.'
    assert os.path.isdir(os.path.join(args.data_dir,'data',args.direction, args.type)), '환자 데이터 경로가 잘못되었습니다.'
    assert os.path.isdir(os.path.join(args.data_dir, 'csv')), 'csv 폴더 경로가 잘못되었습니다.'

    if not os.path.isdir(f'./{args.result}'):
        os.mkdir(f'./{args.result}')
    
    category = sorted(glob(os.path.join(args.data_dir,'data',args.direction, args.type, '*'))) # ./dataset/data/right/patient/*

    result_dict, total_result = init_dict(args)
    
    for cat in category: # ./dataset/data/right/patient/BB1
        
        if args.act_name:
            if cat.split('/')[-1] in args.act_name:
                print("="*60)
                print(f'Each Activity: {cat}')
                print(f"Version: {args.csv_ver}")
                print(f"Dataset Path: {args.data_dir}")
                log_start(cat, args)

                result_dict['Activity'] = {}
                
                activity_name = cat.split('/')[-1] # BB1

                csv_path = make_csv(args, activity_name)

                video_path = os.path.join(cat, 'video','*')
                video_path = glob(video_path)
                video_path = sorted(video_path)
                yaml_path = glob(os.path.join(cat, '*.yaml'))
                if len(yaml_path) == 0:
                    print(f"Empty File {cat.split('/')[-1]}")
                    result_dict['Activity'][activity_name] = []
                    result_dict['Activity'][activity_name].append({
                        'Empty': True
                    })
                    total_result[activity_name] ='Empty'
                    continue
                else:
                    yaml_path = yaml_path[0]

                with open(yaml_path) as f:
                    label = yaml.load(f, Loader=yaml.FullLoader)
                label_idx = sorted(label.keys())

                result_dict, all_count, all_label = count(result_dict, activity_name, video_path, label_idx)
        else:
            print("="*60)   
            print("All Activity")
            print(f"Version: {args.csv_ver}")
            print(f"Dataset Path: {args.data_dir}")
            log_start(cat, args)
            result_dict['Activity'] = {}
            
            activity_name = cat.split('/')[-1] # BB1

            csv_path = make_csv(args, activity_name)

            video_path = os.path.join(cat, 'video','*')
            video_path = glob(video_path)
            video_path = sorted(video_path)
            yaml_path = glob(os.path.join(cat, '*.yaml'))
            if len(yaml_path) == 0:
                print(f"Empty File {cat.split('/')[-1]}")
                result_dict['Activity'][activity_name] = []
                result_dict['Activity'][activity_name].append({
                    'Empty': True
                })
                total_result[activity_name] ='Empty'
                continue
            else:
                yaml_path = yaml_path[0]

            with open(yaml_path) as f:
                label = yaml.load(f, Loader=yaml.FullLoader)
            label_idx = sorted(label.keys())

            result_dict, all_count, all_label = count(result_dict, activity_name, video_path, label_idx)

        result_dict['total'] = {
            'Acc': (all_count/all_label)*100,
            'total_pred_count': all_count,
            'total_label_count': all_label
            
        }

        with open(os.path.join(csv_path, activity_name,'log.json'), 'w', encoding="utf-8") as f:
            json.dump(result_dict, f, indent='\t')   
    
        total_result[f'ver.{args.csv_ver}'][activity_name] = (all_count/all_label)*100
        
        print("="*60)
    with open(os.path.join(csv_path,'total_log.json'), 'w', encoding="utf-8") as f:
            json.dump(total_result, f, indent='\t') 