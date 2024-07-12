import pandas as pd 
import numpy as np
from make_features_acceleration import butter_lowpass_filter,apply_fft_normalized,get_last_freq
import sys
import os 

# define input data and OUTPUT_DIR

csv_path_druckkopf= r"C:\Users\Vincent\Documents\Masterarbeit\Data\Prusa\final_csv_files\prusa_accelerometer_druckkopf.csv"
OUTPUT_DIR = r"C:\Users\Vincent\Documents\Masterarbeit\Data\Prusa\final_csv_files"

# read in the data

df_druckkopf = pd.read_csv(csv_path_druckkopf)

# divide by the classes 

df_druckkopf_cls= {cls: df_druckkopf[df_druckkopf['Class'] == cls] for cls in df_druckkopf['Class'].unique()}
df_druckkopf_good=df_druckkopf_cls[0]
df_druckkopf_stringing=df_druckkopf_cls[1]
df_druckkopf_underextrusion=df_druckkopf_cls[2]


# define new Dataframes to save the features
features_good = pd.DataFrame()
features_stringing = pd.DataFrame()
features_underextrusion = pd.DataFrame()

# Define the parameters for the feature extraction
segment_length = 2000  # Define the length of each segment
sampling_frequency = 500.0  # sample rate, Hz
cutoff_frequency = sampling_frequency/2.5 # desired cutoff frequency of the filter, Hz

# define the value for fft whwere to get the highest frequency
target_value=0.01

# define the columns of interest
columns_interest=['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

# define the dataframes to read in and to save 
dataframes= [df_druckkopf_good,df_druckkopf_stringing,df_druckkopf_underextrusion]
dataframes_to_save=[features_good,features_stringing,features_underextrusion]
    
# Calculate RMS for each segment in each column
for j,df in enumerate(dataframes):
    for col in columns_interest:
        col_rms = []
        col_peak_frequency = []
        col_min_value = []
        col_max_value = []
        for start in range(0, len(df), segment_length):
            segment = df[col].iloc[start:start + segment_length]
            if len(segment) == segment_length:
                rms_value = np.sqrt(np.mean(segment**2))
                max_value = np.max(segment)
                min_value = np.min(segment)

                segment_filtered= butter_lowpass_filter(segment, cutoff_frequency, sampling_frequency)
                fft,freq=apply_fft_normalized(segment_filtered,sampling_frequency)
                peak_frequency=get_last_freq(fft,freq,target_value)


                col_rms.append(rms_value)
                col_peak_frequency.append(peak_frequency)
                col_min_value.append(min_value)
                col_max_value.append(max_value)

        
        # Add features to new dataframes 
        dataframes_to_save[j][f'{col}_rms'] = col_rms
        dataframes_to_save[j][f'{col}_peak_frequency'] = col_peak_frequency
        dataframes_to_save[j][f'{col}_min_value'] = col_min_value
        dataframes_to_save[j][f'{col}_max_value'] = col_max_value


        dataframes_to_save[j]["Class"]=j
       
# concetanate the new dataframes to the a new one with all classes 
complete_df= pd.concat(dataframes_to_save,axis=0,ignore_index=True)

# save the new dataframe to a csv file 
output_path=os.path.join(OUTPUT_DIR, "features_acceleration.csv")

complete_df.to_csv(output_path, index=False)