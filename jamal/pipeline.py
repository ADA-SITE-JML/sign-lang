import os
import random

import numpy as np
import pandas as pd

class Config:
    csv_path = ''
    seed = 69
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    max_words_in_sentence = 10
    video_folder = '../../video'

    # attn_state_path = 'attn.pth'
    # attn_logs = 'attn.csv'
    
    # train_csv_path = '.csv'
    # test_csv_path = '.csv'

config = Config()


# read cvs file
sentences = pd.read_csv('sentences.csv',header = None)

# unique words
word_set = set()
sentences.iloc[:,2].str.lower().str.split().apply(word_set.update)
sorted_word_set = sorted(word_set)
print('Unique words',sorted_word_set)

# create word encoding
encodings = { k:v for v,k in enumerate(sorted_word_set)}
print('Word encodings',encodings)

# converts a sentence with zero padded encoding list
def get_sentence_encoded(sentence):
    encoded = [encodings[key] for key in sentence.split()]
    return  encoded+ list([0]) * (config.max_words_in_sentence - len(encoded))

print('Padding test 1',get_sentence_encoded('mən hansı sənəd vermək'))
print('Padding test 2',get_sentence_encoded('mən bakı yaşamaq'))

# generate (video file name, encoding list)
# Good recommendation on not to iterate over DFs like this:
# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
# but it's not my case - I have fewer rows and one to many with videos.
df = pd.DataFrame(columns=["id", "video_file","encoding"])

for index, row in sentences.iterrows():
    id = row[0]
    phrase = row[2].lower()
    encoded = get_sentence_encoded(phrase)
    # iterate over video folders
    dir = config.video_folder+'/'+str(id)
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            entry = pd.DataFrame.from_dict({"id": id, "video_file": f, "encoding": [encoded]})
            df = pd.concat([df, entry], ignore_index = True)

print(df.head(5))
