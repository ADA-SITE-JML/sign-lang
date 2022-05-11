import os
import shutil
import logging
import datetime
import sys


# py_parent_dir = os.path.dirname(os.path.abspath('temp_video_handler.py'))
# os.startfile(py_parent_dir)
py_parent_dir = 'H:\Other computers\My Computer\SLR'

d = datetime.datetime.now()

log_filename = f"C:\\Users\\togru\\ml-playground\\SLR\\operations\\logs\\video_handler\\logfile_{d.strftime('%Y-%m-%d %H-%M-%S')}.log"

logging.basicConfig(filename = log_filename,
                    filemode = "w+",
                    level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info(f"Logging Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")

src_parent = f"{py_parent_dir}\\Data\\Video\\Cam2"
dest_parent = f"C:\\Users\\togru\\ml-playground\\SLR\\Data\\Video\\temp"

if not os.path.exists(dest_parent):
    os.mkdir(dest_parent)

dest_file_indices, src_file_indices = [], []

for (dirpath, dirnames, filenames) in os.walk(dest_parent):
    # print("filename: ", dirnames)
    for file_name in filenames:
        dest_file_indices.append(file_name)

# print("destination file indices: ", dest_file_indices)
if dest_file_indices:
    dest_file_indices.sort()

for (dirpath, dirnames, filenames) in os.walk(src_parent):
    for file_name in filenames:
        # print(file_name)
        src = dirpath + '\\' + file_name
        dest_path = dest_parent + '\\' + dirpath.split('\\')[-1]
        dest = dest_path + '\\' + file_name
        logging.info(f"Looking for {file_name} in {dest_path}")

        if not dest_file_indices or file_name > dest_file_indices[-1]:

            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
                logging.info(f"Destination Path '{dest_path}' created")

            shutil.copyfile(src,dest)
            logging.info(f"File '{file_name}' copied to path: {dest_path}")

        elif file_name <= dest_file_indices[-1]:
            os.remove(dest)
            logging.info(f"File '{file_name}' removed from path: {dest_path}")

logging.info(f"Logging Session Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
