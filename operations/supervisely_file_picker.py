import json
import os
import argparse
import shutil


def get_annotated_video_list(annotations_src: str) -> list:
    uploaded_list = []

    for directory in os.listdir(annotations_src):
        
        ann_path = os.path.join(annotations_src, directory, "ann")
        
        if not os.path.exists(ann_path):
            continue
        
        for filename in os.listdir(ann_path):
            if "json" in filename:
                uploaded_list.append(filename.split('.')[0] + ".mp4")
                
    
    # write contents of the uploaded_list to a txt file
    
    with open(os.path.join(annotations_src, 'uploaded_list.txt'), 'w+') as f:
        for item in uploaded_list:
            f.write(f"{item}\n")
    
    return uploaded_list
    
def get_unannotated_video_copies(video_src: str, annotations_src: str, video_copy_dest: str, verbose: bool = False) -> None:
    
    uploaded_list = get_annotated_video_list(annotations_src)
    
    if not video_copy_dest:
        video_copy_dest = os.path.join(video_src, 'upload_copies')
        
    for directory in os.listdir(video_src):
        
        video_dir = os.path.join(video_src, directory)
        
        if os.path.isdir(video_dir):
            
            for filename in os.listdir(video_dir):
                
            # copy video from one directory to another
                if filename not in uploaded_list:
                    src = os.path.join(video_dir, filename)
                    dst = os.path.join(video_copy_dest, directory, filename)
                    shutil.copy(src, dst)
                    
                    if verbose:
                        print("Copied {} to {}".format(src, dst))
        
            
# command line argument parser for the get_unannotated_video_copies method
    
    
parser = argparse.ArgumentParser(description='Get unannotated video copies')
parser.add_argument('-vs', '--video_src', type=str, required=True,
                    help='path to the directory containing the videos')
parser.add_argument('-as', '--annotations_src', type=str, required=True,
                    help='path to the uploaded list file')
parser.add_argument('--video_copy_dest', type=str, default=None,
                    help='path to the directory where the video copies will be stored')
parser.add_argument('--verbose', default=False, help='verbose mode')
args = parser.parse_args()

get_unannotated_video_copies(args.video_src, args.annotations_src, args.video_copy_dest, args.verbose)