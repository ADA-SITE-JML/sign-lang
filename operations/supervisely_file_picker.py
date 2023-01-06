import json
import os
import argparse



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
    
    with open('uploaded_list.txt', 'w+') as f:
        for item in uploaded_list:
            f.write(f"{item}\n")
    
    return uploaded_list
            
# command line argument parser for the get_annotated_video_list method
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotations_src", required=True, help="path to the annotations directory")
args = vars(parser.parse_args())

get_annotated_video_list(args["annotations_src"])