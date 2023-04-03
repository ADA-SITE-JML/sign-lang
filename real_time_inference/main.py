import cv2
import torch
import mediapipe as mp
from EncoderRNN import *
from AttnDecoderRNN import *
from lstm_attention_inference import *
import time
from torchvision.transforms import (
    Compose,
    CenterCrop,
    Resize
)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

frames = torch.zeros((0,3,480,640)).to(device)
#frames = []
max_frames = 64
img_size = 224
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
recognize = False
sentence = " - "
text_coord = (450, 450)

def show_frames(list1):
    global frames
    print(list1)
    frames.clear()

def begin_threading(starter):
    global start_thread
    if starter:
        t1.start()
    start_thread=False

def apply_video_transforms(resize_size: int = img_size):
    return Compose([
        CenterCrop(300),
        Resize(size=(resize_size, resize_size))
    ])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        if recognize == False:
            recognize = True

        cv2.circle(img, (300, 30), 30, (0, 0, 255), cv2.FILLED)
        #frames.append(imgRGB)
        frames = torch.cat((frames, torch.unsqueeze(torch.from_numpy(imgRGB).permute(2, 0, 1),dim=0)),0)
    else:
        if recognize == True:
            # Recognize the collected frames
            measure = time.time()

            video_frames = frames #torch.FloatTensor(frames)
            print('List to tensor conversion:',time.time() - measure)
            measure = time.time()

            n,l,w,h = video_frames.shape
            # video_frames = video_frames.permute(0,3,1,2)
            # print('Permutation:',time.time() - measure)
            # measure = time.time()

            video_frames = apply_video_transforms()(video_frames)
            print('Transformation:',time.time() - measure)
            measure = time.time()
            print('Shape:',video_frames.shape)

            compliment_arr = torch.zeros(max(0,max_frames-n),l,img_size,img_size).to(device)
            video_frames = torch.cat((video_frames,compliment_arr),0)

            output_words, attentions = evaluate(video_frames)

            sentence = output_words
            frames = torch.zeros((0,3,480,640)).to(device)

            recognize = False

        cv2.circle(img, (300, 30), 30, (0, 255, 0), cv2.FILLED)

        cv2.putText(img, sentence, text_coord, font, fontScale,
                        color, thickness, cv2.LINE_4, False)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()

print(frames)