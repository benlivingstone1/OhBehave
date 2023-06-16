import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import glob
import os
import re

def analysis(df, start):
    fps = 32.318
    # Drop data points before start time
    start_frame = int(start * fps)
    df = df.drop(df.index[:start_frame])
    df.reset_index(drop=True)

    # Drop all data after 10 min (600s)
    end_frame = int(600 * fps)
    df  = df.drop(df.index[end_frame:])

    # Determine the dimensions of the arena by looking at min/max x and y values
    dimensions = [(df['x'].max() - df['x'].min()), (df['y'].max() - df['y'].min())]
    # The arena is 48x48cm, we will assume the avg between x and y dimensions is 48cm
    fourtyeight = np.mean(dimensions)
    print(fourtyeight)
    # 'unit' is the number of pixels / cm
    unit = fourtyeight / 48
    print(unit)

    # df = df.drop(df.index[:625])
    df['distance'] = ((df['x'].diff() ** 2 + df['y'].diff() ** 2) ** 0.5) / unit
    df['rolling average distance'] = df['distance'].rolling(5).mean()

    # Calculate the cumulative distance
    df['cumulative distance'] = df['rolling average distance'].cumsum()

    # Calculate velocity
    df['velocity'] = np.abs(df['rolling average distance'].diff() / (1/fps))

    return df


if __name__ == "__main__":
    # Create dataframe for summary data across all animals in folder
    summary = pd.DataFrame()
    summary['parameters'] = ['Edge time', 'Center time', 'Cumulative distance', 'Average speed']

    # because you didn't crop the videos and had to input start times manually *clown emoji*
    # # OFT exp 1.1:
    # starts = {'10941': 18, '10942': 26, '10943': 30, '10944': 32, '10945': 25, '10946': 36,
    #           '10947': 26, '10948': 26, '10949': 9, '10950': 11, '10951': 17, '10952': 15,
    #           '10953': 17, '10954': 11, '10955': 13, '10956': 12, '10957': 11, '10958': 8,
    #           '10959': 14, '10960': 10}
    # OFT exp 1.2:
    starts = {'10961': 31, '10962': 33, '10963': 21, '10964': 14, '10965': 54, '10966': 54,
              '10967': 20, '10968': 20, '10969': 48, '10970': 48, '10971': 31, '10972': 33,
              '10973': 30, '10974': 32, '10975': 15, '10976': 13, '10977': 30, '10978': 32,
              '10979': 34, '10980': 36}

    for file in glob.glob('/Volumes/Extreme SSD/Behavioural_pilot_videos/calibrated_2OFT/output_csv/*.csv'):
        base_name = os.path.basename(file)
        df = pd.read_csv(file)

        # Get the animal numbers from file name
        pattern = '(\d{5}-\d{2})'
        match = re.search(pattern, file)
        if match:
            section = match.group(1)
            print(section)
        nums = section.split('-')
        large = nums[0][:3]
        left_num = nums[0]
        right_num = large + nums[1]

        # Split the main dataframe in 2 for the animals on the left / right
        left_df = df[df['rectangle'] == 'LEFT'].reset_index(drop=True)
        right_df = df[df['rectangle'] == 'RIGHT'].reset_index(drop=True)

        # Get video data and length of test
        fps = 32.318
        seconds = left_df.shape[0] / fps
        test_range = np.arange(0, seconds, 1/fps)

        # Analyze data for left and right animals
        left_df = analysis(left_df, starts[left_num])
        right_df = analysis(right_df, starts[right_num])

        # Save analysis under animal ID#
        left_df.to_csv(f'/Volumes/Extreme SSD/Behavioural_pilot_videos/calibrated_2OFT/analysed_csv/OFT_{left_num}_analyzed.csv')
        right_df.to_csv(f'/Volumes/Extreme SSD/Behavioural_pilot_videos/calibrated_2OFT/analysed_csv/OFT_{right_num}_analyzed.csv')

        # Calculate edge and center time for summary table 
        # Right/left edge time (ret/let)
        df = left_df[left_df['location'] == 'edge'].reset_index(drop=True)
        let = df.shape[0] / fps
        df = right_df[right_df['location'] == 'edge'].reset_index(drop=True)
        ret = df.shape[0] / fps

        # Right/left center time (rct/lct)
        df = left_df[left_df['location'] == 'center'].reset_index(drop=True)
        lct = df.shape[0] / fps
        df = right_df[right_df['location'] == 'center'].reset_index(drop=True)
        rct = df.shape[0] / fps

        times = [let, ret, lct, rct]
        print(*times)

        summary[f'{left_num}'] = [let, lct, left_df['rolling average distance'].sum(), left_df['velocity'].mean()]
        summary[f'{right_num}'] = [ret, rct, right_df['rolling average distance'].sum(), right_df['velocity'].mean()]

    
    summary.to_csv('/Volumes/Extreme SSD/Behavioural_pilot_videos/calibrated_2OFT/analysed_csv/summary_data.csv')
        # # Small bins
        # plt.hist2d(left_df['x'], left_df['y'], bins=(50, 50), cmap=plt.cm.jet)
        # plt.show()
        

        
    

