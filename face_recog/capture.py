import argparse
import glob
import os
import sys

import numpy as np
sys.path.append('../')

from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN
import torch
from PIL import Image
import cv2
import pandas as pd

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
MARGIN = 14
THRES = .75

def process_video(vid_path, df_list, detector, resnet, vid_dir):
    cap = cv2.VideoCapture(vid_path)
    idx = 0
    if not cap.isOpened():
        print(f"Error: Could not open video stream {vid_path}.")
        return df_list
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 3)  # Process one frame per second
    
    while True:
        color = (255, 0, 0)
        ret, BRG_frame = cap.read()
        if not ret:
            break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if frame_id % frame_interval == 0:
            frame = cv2.cvtColor(BRG_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(frame)
            if boxes is not None:
                for box in boxes:
                    box = np.array(box, dtype=int)
                    cv2.rectangle(BRG_frame, (box[0], box[1]), (box[2], box[3]), color, 4)
                    path = f'img/{vid_dir}/capture_image_{idx}.jpg'
                    input_face = detector(frame, save_path=path)
                    img_embeddings = resnet(input_face).detach().numpy()[0]
                    df_list.append([path, str(img_embeddings.tolist())])
                    idx += 1
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        # cv2.imshow('image', BRG_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(df_list, columns=['path', 'vector'])

    return df_list

def run_cam(vid_paths, name):
    df_list = []
    detector = MTCNN(keep_all=True, margin=MARGIN)
    
    for vid_path in vid_paths:
        base = os.path.basename(vid_path)
        df_list = process_video(vid_path, df_list, detector, resnet, base)
        print(f'done {base}')
    
    df = pd.DataFrame(df_list, columns=['path', 'vector'])
    os.makedirs('../df', exist_ok=True)
    df.to_csv(f'../df/{name}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facenet capture face')
    parser.add_argument("--name", default='p2v', type=str, help="dataframe csv save name")
    parser.add_argument("--vid_dir", default='../vid', type=str, help="Video directory")
    # parser.add_argument("--vid_paths", nargs='+', default=[], type=str, help="List of video paths")
    args = parser.parse_args()
    vid_paths = glob.glob(os.path.join(args.vid_dir, '*'))
    run_cam(vid_paths, args.name)
