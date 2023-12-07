import os
import pandas as pd
import numpy as np
from process_csv import readProfile
from process_images import process_images
from write_users import writeUsers
import keras

# process test csv file, writing xml files for each user
def process_test_csv(data_fp, pred_fp):
    img_dir_path = os.path.join(data_fp, "image")

    df = readProfile(data_fp)

    images = []
    user_ids = []

    for i,row in df.iterrows():
        userid = row['userid']
        img_path = os.path.join(img_dir_path, userid + ".jpg")       

        images.append(img_path)
        user_ids.append(userid)
        
    df_upd = pd.DataFrame()
    df_upd['user_ids'], df_upd['img_path'] = user_ids, images
    
    test_data = process_images(img_dir_path,df_upd,0)

    user_ids = test_data['ids_with_faces']
    imgs = test_data['imgs']
    ids_wo_faces = test_data['ids_wo_faces']
    user_ids = np.array(user_ids)

    userIDs = []

    for i, row in enumerate(user_ids):
        X = imgs[i] / 255
        gender_dict = {0:'male', 1:'female'}
        age_group_dict = {0: "xx-24", 1: "25-34", 2: "35-49", 3: "50-xx"}
        predictions = predict(X.reshape(1, 128, 128, 1))
        predicted_gender = gender_dict[np.argmax(predictions[0][0])]
        predicted_age_group = age_group_dict[np.argmax(predictions[1][0])]
        dict = {'userid': row, 'gender': predicted_gender, 'age_group': predicted_age_group}
        userIDs.append(dict)
        print(row, predicted_gender, predicted_age_group)

    for i, row in enumerate(ids_wo_faces):
        dict = {'userid': row, 'gender': 'female', 'age_group': 'xx-24'}
        userIDs.append(dict)
    
    userIDs = pd.DataFrame(userIDs)
    writeUsers(pred_fp, userIDs)

def predict(img):
    model = keras.models.load_model('./age_gender_model.keras')
    prediction = model.predict(img)
    return prediction