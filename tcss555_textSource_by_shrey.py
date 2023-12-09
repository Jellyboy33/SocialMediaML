import sys
import os
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVR
import tensorflow as tf


# Download VADER lexicon if not already downloaded
import nltk
nltk.download('vader_lexicon')

def main():
    args = sys.argv[1:]
   
    #obtain filepaths for input and output
    test_data_fp = args[1]
    pred_fp = args[3]

    #///CHANGE TO CORRECT TRAINING FILEPATH///
    train_data_fp = 'training/'
    profile_df = readProfile(train_data_fp)
    liwc_df = readLIWC(train_data_fp)
    text_df = readText(train_data_fp)
    test_text_df = readText(test_data_fp)
    model, tfidf_vectorizer = modelTraining(profile_df, text_df)
    target_traits = ['ope', 'con', 'ext', 'agr', 'neu']
    model_personality = modelTrainingPersonality(liwc_df, profile_df, target_traits)
    X, y = preprocessingTrainData(profile_df, liwc_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1501, random_state=42)
    # Create a StandardScaler instance
    scaler = StandardScaler()
    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_logistic_reg = LogisticRegression(max_iter=10000, C=100, penalty='l2', solver='newton-cg')
    best_logistic_reg.fit(X_train_scaled, y_train)

    test_profile_df = readProfile(test_data_fp)
    test_liwc_df = readLIWC(test_data_fp)
    test_liwc_df = test_liwc_df.drop('Seg', axis=1)
    test_liwc_df['userId'] = test_liwc_df['userId'].apply(str)
    test_userids = test_profile_df.iloc[:, 1]
    #print(test_userids)
    writeUsers(pred_fp, test_userids, best_logistic_reg, test_liwc_df, scaler, model, test_text_df, model_personality, target_traits, tfidf_vectorizer)

    
     

def modelTraining(profile_df, text_data):
    merged_data = pd.merge(text_data, profile_df, how='inner', on='userid')

    # Perform sentiment analysis and add the 'compound' column using VADER
    sid = SentimentIntensityAnalyzer()
    merged_data['compound'] = merged_data['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Preprocess text data
    X = merged_data[['text', 'compound']]
    y = merged_data['gender']

    # Label encoding for gender
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=100)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])

    # Combine TF-IDF with sentiment scores
    X_train_combined = hstack([X_train_tfidf, X_train['compound'].values.reshape(-1, 1)])
    X_test_combined = hstack([X_test_tfidf, X_test['compound'].values.reshape(-1, 1)])

    X_train_combined = X_train_combined.toarray()
    X_test_combined = X_test_combined.toarray()

    # Define and compile the improved neural network model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train_combined.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train_combined, y_train, epochs=10, batch_size=64, validation_data=(X_test_combined, y_test), callbacks=[early_stopping])

    return model, tfidf_vectorizer



def readLikes(data_fp):
    with open(os.path.join(data_fp,"relation/relation.csv"),'r') as f:
         userInfo = pd.read_csv(f)
   
    userInfo['like_id'] = userInfo['like_id'].apply(str)
    return userInfo
   

def writeUsers(pred_fp, userIDs, best_logistic_reg, test_liwc_df, scaler, model, test_text_df, model_personality, target_traits, tfidf_vectorizer):
    if (not os.path.exists(pred_fp)):
        os.mkdir(pred_fp)
    for user in userIDs:
        writeUser(pred_fp, user, best_logistic_reg, test_liwc_df, scaler, model, test_text_df, model_personality, target_traits, tfidf_vectorizer)


def writeUser(pred_fp, userID, best_logistic_reg, test_liwc_df, scaler, model, test_text_df, model_personality, target_traits, tfidf_vectorizer):
    userFile = str(userID) + '.xml'
    #change to column name of prediction
    age_group = predict(best_logistic_reg, userID, test_liwc_df, scaler)
    gender = predictUsingNeuralNetwork(model, userID, test_text_df, tfidf_vectorizer)
    open_score = predictUsingSVR(model_personality['ope'], userID, test_liwc_df)
    con_score = predictUsingSVR(model_personality['con'], userID, test_liwc_df)
    ext_score = predictUsingSVR(model_personality['ext'], userID, test_liwc_df)
    agree_score = predictUsingSVR(model_personality['agr'], userID, test_liwc_df)
    neur_score = predictUsingSVR(model_personality['neu'], userID, test_liwc_df)

    output = f'<user id="{userID}"\n' \
             f'age_group="{age_group}"\n' \
             f'gender="{gender}"\n' \
             f'open="{open_score}"\n' \
             f'conscientious="{con_score}"\n' \
             f'extrovert="{ext_score}"\n' \
             f'agreeable="{agree_score}"\n' \
             f'neurotic="{neur_score}"/>'
    
    f = open(os.path.join(pred_fp,userFile),"w")
    f.write(output)
    f.close()

def modelTrainingPersonality(liwc_df, profile_df, target_traits):
    new_profile_df = profile_df.copy()
    new_profile_df.drop(['Unnamed: 0', 'age', 'gender'], axis=1, inplace=True)
    merged_df = pd.merge(liwc_df, new_profile_df, left_on='userId', right_on='userid')
    features = merged_df.drop(['userId', 'userid','Seg', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, merged_df[target_traits], test_size=501, random_state=42)
    models = {}
    for trait in target_traits:
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train[trait])
        models[trait] = model
    return models

def preprocessingTrainData(profile_df, liwc_df):
    new_profile_df = profile_df[['userid', 'age']]
    # Merge the datasets using 'userid' as the common column
    merged_df = pd.merge(liwc_df, new_profile_df, left_on='userId', right_on='userid')
    # Drop 'Seg' column as mentioned
    merged_df = merged_df.drop('Seg', axis=1)
    merged_df = merged_df.drop(['userId', 'userid'], axis=1)
    merged_df['age_group'] = merged_df['age'].apply(age_to_group)
    X = merged_df.iloc[:, 0:81]
    y = merged_df.iloc[:, -1]
    return X, y


def predict(best_logistic_reg, userID, test_liwc_df, scaler):
    #print(test_liwc_df.info())
    row = test_liwc_df[test_liwc_df['userId'] == str(userID)]
    test_X = row.iloc[:, 1:]
    test_X_scaled = scaler.transform(test_X)
    test_y = best_logistic_reg.predict(test_X_scaled)
    return test_y[0]

from nltk.sentiment import SentimentIntensityAnalyzer

def predictUsingNeuralNetwork(model, userID, test_text_df, tfidf_vectorizer):
    row = test_text_df[test_text_df['userid'] == str(userID)].copy()  # Use copy to avoid the warning
    
    # Ensure 'text' column is a string
    row['text'] = row['text'].astype(str)
    
    # Sentiment analysis directly on 'text' column
    sid = SentimentIntensityAnalyzer()
    row['compound'] = row['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Combine TF-IDF with sentiment scores
    text_vectorized = tfidf_vectorizer.transform(row['text'])
    test_X_text_stack = hstack([text_vectorized, row['compound'].values.reshape(-1, 1)])
    test_X_text_stack = test_X_text_stack.toarray()
    # Ensure the shape of test_X_text_stack is (1, 5001)    
    # Make predictions
    y_pred = (model.predict(test_X_text_stack) > 0.5).astype(int)
    return y_pred[0][0]
    
def predictUsingSVR(model, userID, test_liwc_df):
    #print(test_liwc_df.info())
    row = test_liwc_df[test_liwc_df['userId'] == str(userID)]
    test_X = row.iloc[:, 1:]
    test_y = model.predict(test_X)
    return test_y[0]

def age_to_group(age):
    if age <= 24:
        return 'xx-24'
    elif 25 <= age <= 34:
        return '25-34'
    elif 35 <= age <= 49:
        return '35-49'
    else:
        return '50-xx'

def readProfile(data_fp):
    with open(os.path.join(data_fp,"profile/profile.csv"),'r') as f:
         userInfo = pd.read_csv(f)
    return userInfo

def readLIWC(data_fp):
    with open(os.path.join(data_fp,"LIWC/LIWC.csv"),'r') as f:
        userInfo = pd.read_csv(f)
    return userInfo

def readText(data_fp):
    text_folder_path = f'{data_fp}/text'
    profile_df = pd.read_csv(f'{data_fp}/profile/profile.csv')
    def read_text_files(user_ids):
        data = []
        for user_id in user_ids:
            file_path = f'{text_folder_path}/{user_id}.txt'
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    data.append({'userid': user_id, 'text': text})
            except FileNotFoundError:
                pass  # Handle missing files if needed
        return pd.DataFrame(data)
    text_data = read_text_files(profile_df['userid'])
    return text_data


if __name__ == "__main__":
        main()
