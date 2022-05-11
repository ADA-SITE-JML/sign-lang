import os
import shutil
import logging
import datetime
import sys
import pandas as pd
import numpy as np

py_parent_dir = os.path.dirname(os.path.abspath('dir_formatter.py'))
# py_parent_dir = 'H:\Other computers\My Computer\SLR'

d = datetime.datetime.now()

log_filename = f"C:\\Users\\togru\\ml-playground\\SLR\\operations\\logs\\dir_formatter\\logfile_{d.strftime('%Y-%m-%d %H-%M-%S')}.log"

logging.basicConfig(filename = log_filename,
                    filemode = "w+",
                    level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info(f"Logging Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")

src_parent = f"{py_parent_dir}\\Data\\Video\\Cam2"
dest_parent = r"C:\Users\togru\ml-playground\SLR\Data\Video\Cam1"


if not os.path.exists(dest_parent):
    os.mkdir(dest_parent)

dest_file_indices= []
for (dir_path, dir_names, file_names) in os.walk(dest_parent):

    for file_name in file_names:
        sentence_no = dir_path.split('\\')[-1]
        path_to_sentence = sentence_no + '\\'
        dest_file_indices.append([sentence_no,file_name, path_to_sentence + file_name])

dest_files = pd.DataFrame(dest_file_indices, columns=['sentence_no', 'file_name', 'full_path'])

logging.info(f"Destination directory file indexed --> {len(dest_file_indices)} files present")

if dest_file_indices:
    dest_file_indices.sort()

for (dir_path, dir_names, file_names) in os.walk(src_parent):
    for file_name in file_names:

        sentence_no = dir_path.split('\\')[-1]
        path_to_sentence = sentence_no + '\\'
        dest_path = dest_parent + '\\' + path_to_sentence
        dest_file_path = dest_path + file_name
        logging.info(f"Looking for {file_name} in {dest_parent}")
        # print('outside if: ', dest_files['full_path'])

        if path_to_sentence + file_name not in dest_files['full_path'].values and file_name in dest_files['file_name'].values:

            org_file_path = dest_parent + '\\' + file_name

            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
                logging.info(f"Destination Path '{dest_path}' created")

            shutil.move(org_file_path, dest_file_path)
            logging.info(f"File '{file_name}' in {dest_parent} moved to to path: {dest_path}")

        elif path_to_sentence + file_name in dest_files['full_path'].values:
            logging.info(f"File '{file_name} already in {dest_path}'")

        elif file_name not in dest_files['file_name'].values:
            logging.info(f"File '{file_name} does not exist in {dest_parent}'")


for (dir_path, dir_names, file_names) in os.walk(dest_parent):
    print(f"{dir_path} {dir_names} {file_names} {len(file_names)}")

    for file_name in file_names:
        file_path = dir_path + '\\' + file_name
        os.remove(file_path)
        logging.info(f"File '{file_name}' removed from path: {dir_path}")
    break


logging.info(f"Logging Session Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
