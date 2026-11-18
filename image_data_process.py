import os
import cv2 as cv
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, time
from skimage.metrics import structural_similarity

#====================================RENAMING================================
"""
root = Path("/home/priyasha/multi_gt_images")

for folder in root.iterdir():
    if not folder.is_dir():
        continue

    print(f"\nProcessing {folder.name}...")

    for file in folder.glob("*.jpg"):
        parts = file.stem.split("_")

        if len(parts) != 5:
            print(f"Skipped: {file.name}")
            continue

        # Place_Date_001_HH_MM
        date = parts[1]
        hour = parts[3]
        minute = parts[4]

        new_name = f"{date}_{hour}_{minute}.jpg"

        file.rename(file.with_name(new_name))
        print(f"{file.name} -> {new_name}")

print("\nAll folders completed.")

#============REMOVE ORIGINAL IMAGES IF CORRESPONDING DEHAZE IMAGE DOES NOT EXISTS==========
#============BELUR======================
dr=os.listdir("/home/priyasha/multi_gt_images/Belur")
dhr=os.listdir("/home/priyasha/multi_gt_images_dehazed/Belur")

for im in dr:
    d=im[:-4]+'_d.png'
    if d in dhr:
        continue
    else:
        os.remove('/home/priyasha/multi_gt_images/Belur/'+im)
"""
'''
#==============GHUSURI=======================
drg=os.listdir("/home/priyasha/multi_gt_images/Ghusuri")
dhrg=os.listdir("/home/priyasha/multi_gt_images_dehazed/Ghusuri")

for im in drg:
    d=im[:-4]+'_d.png'
    if d in dhrg:
        continue
    else:
        os.remove('/home/priyasha/multi_gt_images/Ghusuri/'+im)
'''
'''
#===================FRB===========================
drf=os.listdir("/home/priyasha/multi_gt_images/FRB")
dhrf=os.listdir("/home/priyasha/multi_gt_images_dehazed/FRB")

for im in drf:
    d=im[:-4]+'_d.png'
    if d in dhrf:
        continue
    else:
        os.remove('/home/priyasha/multi_gt_images/FRB/'+im)
'''
#====================CLEAN DATASET==================================================================

df_belur = pd.read_excel('/home/priyasha/Downloads/site_510120260612093607.xlsx', skiprows=16)
df_ghusuri = pd.read_excel('/home/priyasha/Downloads/site_27420260612094020.xlsx', skiprows=16)
df_frb = pd.read_excel('/home/priyasha/Downloads/site_29620260612094319.xlsx', skiprows=16)

def clean_dataframe(df):
    From = df['From Date'].str.split(' ', n=1, expand=True)
    To = df['To Date'].str.split(' ', n=1, expand=True)

    df['From'] = pd.to_datetime(From[0], format='%d-%m-%Y')
    df['To'] = pd.to_datetime(To[0], format='%d-%m-%Y')

    df['Time_1'] = pd.to_datetime(From[1], format='%H:%M').dt.time
    df['Time_2'] = pd.to_datetime(To[1], format='%H:%M').dt.time

    for col in ['PM2.5', 'RH', 'WS', 'AT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['PM2.5', 'RH', 'WS', 'AT'])

    return df[['From', 'To', 'Time_1', 'Time_2', 'RH', 'WS', 'AT', 'PM2.5']]

df_belur = clean_dataframe(df_belur)
df_ghusuri = clean_dataframe(df_ghusuri)
df_frb = clean_dataframe(df_frb)
    #SAVE AS CSV
#df_belur.to_csv('/home/priyasha/multi_gt_images/csv/belur_data.csv',index=False)
#df_ghusuri.to_csv('/home/priyasha/multi_gt_images/csv/ghusuri_data.csv',index=False)
#df_frb.to_csv('/home/priyasha/multi_gt_images/csv/frb_data.csv',index=False)

#=============================CREATE NEW DATASETS===========================================

from pathlib import Path
from datetime import datetime
import pandas as pd

def create_dataset(image_folder, station_df):
    X = []
    rh_list = []
    ws_list = []
    at_list = []
    pm25_list = []

    image_folder = Path(image_folder)

    for img in image_folder.glob("*.jpg"):

        filename = img.name
        date = datetime.strptime(filename[:11], "%d-%b-%Y")
        t = datetime.strptime(filename[12:17], "%H_%M").time()

        row = station_df[
            (station_df["From"] == date) &
            (station_df["Time_1"] <= t) &
            (station_df["Time_2"] > t)
        ]

        if row.empty:
            continue

        X.append(filename)
        rh_list.append(row.iloc[0]["RH"])
        ws_list.append(row.iloc[0]["WS"])
        at_list.append(row.iloc[0]["AT"])
        pm25_list.append(row.iloc[0]["PM2.5"])

    return pd.DataFrame({
        "ImageFile": X, "RH": rh_list,
        "WS": ws_list,
        "AT": at_list,
        "PM2.5": pm25_list
    })

frb=create_dataset("/home/priyasha/new data/roi_preset_combined_train",df_frb)
#belur_pm = create_dataset("/home/priyasha/multi_gt_images/Belur",df_belur)
#belur_frb_pm = create_dataset("/home/priyasha/multi_gt_images/Belur",df_frb)
#ghusuri_pm = create_dataset("/home/priyasha/multi_gt_images/Ghusuri",df_ghusuri)
#ghusuri_frb_pm = create_dataset("/home/priyasha/multi_gt_images/Ghusuri",df_frb)
#frb_belur_pm=create_dataset("/home/priyasha/multi_gt_images/FRB",df_frb)

frb.to_csv("/home/priyasha/new data/frb(train).csv",index=False)
#belur_pm.to_csv('/home/priyasha/multi_gt_images/csv/final_csv/belur.csv',index=False)
#belur_frb_pm.to_csv('/home/priyasha/multi_gt_images/csv/final_csv/belur_frb.csv',index=False)
#ghusuri_pm.to_csv('/home/priyasha/multi_gt_images/csv/final_csv/ghusuri.csv',index=False)
#ghusuri_frb_pm.to_csv('/home/priyasha/multi_gt_images/csv/final_csv/ghusuri_frb.csv',index=False)
#frb_belur_pm.to_csv('/home/priyasha/multi_gt_images/csv/final_csv/frb.csv',index=False)
