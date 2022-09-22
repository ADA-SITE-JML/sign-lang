import os
import json
import cv2
import shutil


def video_to_frames(glossStart, glossEnd, video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []
    
    while True:
        res, frame = cap.read()
    
        if res:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames[glossStart: glossEnd+1]

def convert_frames_to_video(frames, path_out, size, fps=25):
    writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame in frames:
        # writing to a image array
        writer.write(frame)
    writer.release()


def create_word_level_data(gloss_df, read_src, write_src):
    for gloss, df in gloss_df.groupby(by='gloss'):
        for i in range(len(df)):
            
            df.reset_index(drop=True, inplace=True)

            video_path = os.path.join(read_src, str(df.loc[i, 'sentenceID']), df.loc[i, 'fileName'])
            
            if os.path.exists(video_path):
                write_path = path_out = os.path.join(write_src, gloss)
                
                if not os.path.exists(write_path):
                    os.mkdir(write_path)

                path_out = os.path.join(write_path, df.loc[i,'fileName'])
                
                frames = video_to_frames(int(df.loc[i, 'glossStart']),
                                            int(df.loc[i, 'glossEnd']),
                                            video_path,
                )

                size = frames[0].shape[:2][::-1]
                convert_frames_to_video(frames, path_out, size, fps=25)
                
                if os.path.exists(path_out):
                    print(f">>> Created: {path_out}")
                else:
                    print("sth wrong")
                    break
