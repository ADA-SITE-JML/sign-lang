import pandas as pd
import json
import os

cwd = os.path.dirname(os.path.abspath("tags_to_sign_language.py"))
print(cwd)
parent_dir = cwd + "/data_validation/data/annotations/"

# df = pd.read_json(json_path)
tags_list = []

for (dirpath, dirnames, filenames) in os.walk(parent_dir):
    if dirpath.split("/")[-1] == 'ann':
        for filename in filenames:
            json_path = dirpath + '/' + filename
            sentence_idx = dirpath.split("/")[-2]
            # print(dirpath.split("\\"))

            with open(json_path) as json_file:
                json_dict = json.load(json_file)
                tags = json_dict['tags']
                for i in range(len(tags)):
                    try:
                        tags[i]['glossStart'] = tags[i]['frameRange'][0]
                        tags[i]['glossEnd'] = tags[i]['frameRange'][1]
                        # print(type(tags[i]['frameRange']))
                    except KeyError as err:
                        print(f">>> Empty label error: {err}")
                    tags[i]['fileName'] = '.'.join(filename.split('.')[:-1])
                    tags[i]['sentenceID'] = sentence_idx
                    
                tags_list += tags
            
# print(">>>", tags_list)
data = pd.DataFrame(tags_list).drop(['frameRange', 'labelerLogin', 'updatedAt', 'createdAt', 'key'], axis=1)
cols = list(data.columns)

cols[0] = 'gloss'
data.columns = cols

data.to_csv(cwd + '/data_validation/processed_gloss.csv', index=False)