import multiprocessing
from time import *
import cv2
from hand_detection import *
from random_word import *
import torch


#frames = torch.zeros(0)

frames=[]




font = cv2.FONT_HERSHEY_SIMPLEX
org = (450, 450)
org2=(450,450)
fontScale = 1
color = (255, 0, 0)
thickness = 2
frames_list = []
wordy = " "

wordy2=" "



start_thread=True

not_visible=False


def hazir_deyil(not_hand):
    if not_hand:
        return "Hele hazir deyil!!!!"

def show_frames(list1):
    global frames
    print(list1)
    frames.clear()

# t1=Thread(target=hazir_deyil,args=[not_visible])
# t1.start()
t1=multiprocessing.Process(target=show_frames,args=(frames))
# t1.start()
def begin_threading(starter):
    global start_thread
    if starter:
        t1.start()
    start_thread=False



cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()


    # print(frames_list)

    img = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        frames.append(imgRGB)
        loopy = True

        cv2.circle(img, (300, 30), 30, (0, 0, 255), cv2.FILLED)
        org = list(org)
        org[0] -= 3
        org = tuple(org)
        wordy += get_word(loopy)
        wordy += " "
        cv2.putText(img, wordy, org, font, fontScale,
                    color, thickness, cv2.LINE_4, False)
        # frames.append(frame)
        # t1.start()
    else:
        begin_threading(start_thread)
        show_frames(frames)
        cv2.circle(img, (300, 30), 30, (0, 255, 0), cv2.FILLED)
        not_visible=True
        org2 = list(org2)
        org2[0] -= 3
        org2 = tuple(org2)
        wordy2+=hazir_deyil(not_visible)
        wordy2 += " "
        cv2.putText(img, wordy2, org2, font, fontScale,
                    color, thickness, cv2.LINE_4, False)





    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break






#     # After the loop release the cap object
cap.release()




# Destroy all the windows
cv2.destroyAllWindows()



print(frames)


