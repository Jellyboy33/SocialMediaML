import os
import pandas as pd
from process_data import process_data
from process_images import process_images

# process training csv file creating the dataframe, adding images to it
def process_csv(data_fp):
    img_dir_path = os.path.join(data_fp, "image")

    df = readProfile(data_fp)
    map_to_string = {0: 'male', 1: 'female'}
    df["gender_label"] = df["gender"].replace(map_to_string)

    age_labels = []
    images = []
    genders = []
    user_ids = []

    for i, row in df.iterrows():
        userid = row['userid']
        age = row['age']
        age_group = age_to_group(age)
        gender = row['gender']
        img_path = os.path.join(img_dir_path, userid + ".jpg")       

        age_labels.append(age_group)
        images.append(img_path)
        genders.append(gender)
        user_ids.append(userid)
        
    df_upd = pd.DataFrame()
    df_upd['user_ids'], df_upd['img_path'], df_upd['gender'], df_upd['age_group'] = user_ids, images, genders, age_labels
    age_group_to_int = {"xx-24": 0, "25-34": 1, "35-49": 2, "50-xx": 3}
    df_upd["age_group_label"] = df_upd["age_group"].replace(age_group_to_int)
    
    data = process_images(img_dir_path,df_upd,1)
    process_data(data)

def readProfile(data_fp):
    with open(os.path.join(data_fp,"profile/profile.csv"),'r') as f:
         userInfo = pd.read_csv(f)
    return userInfo

def age_to_group(age):
    if age <= 24:
        return 'xx-24'
    elif 25 <= age <= 34:
        return '25-34'
    elif 35 <= age <= 49:
        return '35-49'
    else:
        return '50-xx'