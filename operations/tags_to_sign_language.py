import pandas as pd
import json
import os

cwd = os.path.dirname(os.path.abspath("tags_to_sign_language.py"))

parent_dir = cwd + "\\Data\\annotations\\"

# df = pd.read_json(json_path)
tags_list = []

for (dirpath, dirnames, filenames) in os.walk(parent_dir):
    if dirpath.split("\\")[-1] == 'ann':
        for filename in filenames:
            json_path = dirpath + '/' + filename
            sentence_idx = dirpath.split("\\")[-2]
            # print(dirpath.split("\\"))

            with open(json_path) as json_file:
                json_dict = json.load(json_file)
                tags = json_dict['tags']

                for i in range(len(tags)):
                    tags[i]['fileName'] = '.'.join(filename.split('.')[:-1])
                    tags[i]['sentenceIndex'] = sentence_idx
                tags_list += tags

data = pd.DataFrame(tags_list).drop(['labelerLogin', 'updatedAt', 'createdAt', 'key'], axis=1)
data.to_csv(cwd + '\\Data\\processed_gloss.csv')
