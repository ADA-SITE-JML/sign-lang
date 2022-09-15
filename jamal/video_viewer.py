# For hand detection used this: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/

import math
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt


# plays a video file
def play_video_format(video_data):
	# Read until video is completed
	while(video_data.isOpened()):
		# Capture frame-by-frame
		ret, frame = video_data.read()

		if ret == True:
			# Display the resulting frame
			cv2.imshow('Frame',frame)
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break

def play_video_arr(video_arr):
	# Read until video is completed
	for frame in video_arr:
		cv2.imshow('Frame',frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
				break

# shows frames as images
def show_frames(video_data,start_frame, frame_count):
	fig = plt.figure(figsize=(10, 7))
	rows = int(math.sqrt(frame_count))
	columns = int(frame_count/rows)+1

	mpHands = mp.solutions.hands
	hands = mpHands.Hands(static_image_mode=False,
	                      max_num_hands=2,
	                      min_detection_confidence=0.5,
	                      min_tracking_confidence=0.5)

	for f in range(frame_count): 
		fig.add_subplot(rows, columns, f+1)

		video_data.set(cv2.CAP_PROP_POS_FRAMES,f*15)
		ret, frame = video_data.read()

		hand_results = hands.process(frame)
		if ret == True:
			plt.imshow(frame)
			plt.axis('off')
			if hand_results.multi_hand_landmarks:
				plt.title('Detected')
			else:
				plt.title('No hands')
	plt.show()

# keeps only informative frames
def keep_frames_with_hands(video_data):
	video_arr = []

	mpHands = mp.solutions.hands
	hands = mpHands.Hands(static_image_mode=False,
	                      max_num_hands=2,
	                      min_detection_confidence=0.5,
	                      min_tracking_confidence=0.5)

	while(video_data.isOpened()):
		ret, frame = video_data.read()

		if ret == True:
			hand_results = hands.process(frame)
			if hand_results.multi_hand_landmarks:
				video_arr.append(frame)
		# Break the loop
		else:
			break

	return video_arr

if __name__ == '__main__':
	video_dir = '../../video/1/'
	fname = '2022-04-19 15-44-38.mp4'

	sample = cv2.VideoCapture(video_dir+fname)

	# Check if camera opened successfully
	if (sample.isOpened() == False):
		print("Error opening video stream or file")

	# Show the whole video
	#play_video_format(sample)

	# Demonstrate some frames with hand detection
	#show_frames(sample,0,22)

	# Show video with hands only
	hands_only = keep_frames_with_hands(sample)
	play_video_arr(hands_only)

	# When everything done, release the video capture object
	sample.release()
	# Closes all the frames
	cv2.destroyAllWindows()
