import sys

import numpy as np
sys.path.append('../')

from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN
import torch
from PIL import Image
import cv2
# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

def run_cam():
    img_embeddings = []
    img_taken = False
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    detector = MTCNN(keep_all=True)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        color = (255,0,0)
        ret, frame = cap.read()
        w, h = frame.shape[0], frame.shape[1]
        if not ret:
            break
        boxes,_ = detector.detect(frame)
        if boxes is not None:
            for box in boxes:
                box = np.array(box, dtype=int)
                face = frame[box[1]:box[3], box[0]:box[2]]
                input_face = detector(frame)  
                
                if cv2.waitKey(1) & 0xFF == ord('1'):
                    print('Taken picture')
                    img_embeddings = resnet(input_face).detach()
                    cv2.imwrite('img/capture_image.jpg', face)
                    img_taken = True

                if (img_taken):
                    embeddings = resnet(input_face).detach()
                    distance = (embeddings - img_embeddings).norm().item()
                    if distance < 1.0: color = (0,255,0)
                    else: color = (0,0,255)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 4)
                cv2.putText(frame, str(input_face.shape), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('2'):
            print('Delete picture')
            img_taken = False
        if (img_taken):
            cv2.putText(frame, 'img taken', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == 27:  # esc key
            break
        cv2.imshow('image', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam()