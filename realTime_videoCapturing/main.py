# import the opencv library
import cv2


from hand_detection import *
from random_word import *




# define a video capture object
cap = cv2.VideoCapture(0)





font=cv2.FONT_HERSHEY_SIMPLEX
org = (450, 450)
fontScale = 1
color = (255, 0, 0)
thickness=2

wordy=" "
while(True):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        loopy=True

        cv2.circle(img,(300,30), 30, (0,0,255),cv2.FILLED)
        org=list(org)
        org[0]-=3

        org=tuple(org)
        wordy+=get_word(loopy)
        wordy+=" "
        cv2.putText(img, wordy, org, font, fontScale,
                  color, thickness, cv2.LINE_4, False)



    else:
        cv2.circle(img,(300,30), 30, (0,255,0),cv2.FILLED)

    # Display the resulting frame


    cv2.imshow('frame', img)



    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

