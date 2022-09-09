import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def sign_validator(df_path, gloss, save=False):

    for group_id, group in sample_df.groupby(by='gloss'):

        if group_id == gloss: 
            group.reset_index(inplace=True, drop=True)

            nrows = len(group)
            ncols = 5

            fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=(5*ncols, 4*nrows))

            for i in range(len(group)):
                frames = np.linspace(group.loc[i, 'glossStart'], group.loc[i, 'glossEnd'], num=5, dtype=int)
                # print(frames)
                file_path = "C:\\Users\\togru\\ml-playground\\SLR\\Data\\test\\" + str(group.loc[i, 'sentenceID']) + "\\" + str(group.loc[i, 'fileName'])
                

                for j, frame_no in enumerate(frames):
                    cap = cv2.VideoCapture(file_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
                    res, frame = cap.read()

                    if j == 0:
                        axes[i, j].set_ylabel(group.loc[i, 'fileName'])

                    axes[i, j].imshow(frame)
                    
            if save:
                plt.savefig(f"{gloss}_VAL.png")
            # fig.suptitle(str(gloss), fontsize=16)
            plt.show()
