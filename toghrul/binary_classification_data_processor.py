import os
import shutil

src_parent = "../data"
dst_parent = "../data/binary-data/"
partitions = ["train", "val", "test"]
glosses = ["MƏN", "VAR"]


for folder_name in os.listdir(src_parent):
    if folder_name in partitions:        
        for gloss in glosses:
            src_path = os.path.join(src_parent, folder_name, gloss)
            dst_path = os.path.join(dst_parent, folder_name, gloss)
            shutil.copytree(src_path, dst_path)