import sys

import numpy as np
sys.path.append('../')

from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN
import torch
from PIL import Image
import cv2
# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
MARGIN = 14
THRES = .75

def run_cam():
    distance = -1
    img_embeddings = []
    img_taken = False
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    detector = MTCNN(keep_all=True, margin=MARGIN)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        color = (255,0,0)
        ret, BRG_frame = cap.read()
        frame = cv2.cvtColor(BRG_frame, cv2.COLOR_BGR2RGB)
        w, h = frame.shape[0], frame.shape[1]
        if not ret:
            break
        boxes,_ = detector.detect(frame)
        if boxes is not None:
            for box in boxes:
                box = np.array(box, dtype=int)
                face = frame[box[1]:box[3], box[0]:box[2]]
                
                if cv2.waitKey(1) & 0xFF == ord('1'):
                    input_face = detector(frame, save_path = 'img/capture_image.jpg')
                    print('Taken picture')
                    img_embeddings = resnet(input_face).detach()
                    # cv2.imwrite(, face)
                    img_taken = True

                if (img_taken):
                    input_face = detector(frame)
                    embeddings = resnet(input_face).detach()
                    distance = (embeddings - img_embeddings).norm().item()
                    if distance < THRES : color = (0,255,0)
                    else : color = (0,0,255)

                cv2.rectangle(BRG_frame, (box[0], box[1]), (box[2], box[3]), color, 4)
                cv2.putText(BRG_frame, str(distance), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('2'):
            print('Release picture')
            img_taken = False
        if (img_taken):
            cv2.putText(BRG_frame, 'img taken', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == 27:  # esc key
            break
        cv2.imshow('image', BRG_frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam()