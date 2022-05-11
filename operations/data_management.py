import os
import shutil


src_parent = r"D:\SLR\Data\Video\Cam2"
dest_parent = r"D:\SLR\Data\Video\temp"

if not os.path.exists(dest_parent):
    os.mkdir(dest_parent)

dest_file_indices, src_file_indices = [], []

for (dirpath, dirnames, filenames) in os.walk(dest_parent):
    for file_name in filenames:
        dest_file_indices.append(file_name)

if dest_file_indices:
    dest_file_indices.sort()

import datetime

d = datetime.datetime.now()
print(f"{d%%:.2f}")
print(f"{getattr(d, 'year'):g}")

for (dirpath, dirnames, filenames) in os.walk(src_parent):
    for file_name in filenames:
        print(dirpath)

        src = dirpath + '\\' + file_name
        dest_path = dest_parent + '\\' + dirpath.split('\\')[-1]
        dest = dest_path + '\\' + file_name

        if not dest_file_indices or file_name > dest_file_indices[-1]:

            if not os.path.exists(dest_path):
                os.mkdir(dest_path)

            shutil.copyfile(src,dest)

        elif file_name <= dest_file_indices[-1]:
            os.remove(dest)
