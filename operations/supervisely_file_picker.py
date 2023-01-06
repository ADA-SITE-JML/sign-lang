import json
import os

annotations_src = "../SLR_sync/Data/annotations"
uploaded_list = []

def get_annotated_video_list(annotations_src: str) -> list:

    for directory in os.listdir(annotations_src):
        print(directory)
        
        ann_path = os.path.join(annotations_src, directory, "ann")
        
        if not os.path.exists(ann_path):
            continue
        
        for filename in os.listdir(ann_path):
            if "json" in filename:
                uploaded_list.append(filename.split('.')[0] + ".mp4")
                
    
    # write contents of the uploaded_list to a txt file
    
    with open('uploaded_list.txt', 'w') as f:
        for item in uploaded_list:
            f.write("%s" % item)
    
    return uploaded_list
            
print(uploaded_list)

    
    