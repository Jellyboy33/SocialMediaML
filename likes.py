import sys
import os
import random
import pandas as pd
import numpy as np
import math
import keras
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pandas.api.types import is_float_dtype
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt 



def main():
    args = sys.argv[1:]
   
    #obtain filepaths for input and output
    test_data_fp = args[1]
    pred_fp = args[3]

    #///CHANGE TO CORRECT TRAINING FILEPATH///
    train_data_fp = 'C:/Users/jelly/Documents/School/UWT/TCSS555/test_data/training/'


    profile_df = readProfile(train_data_fp)
    likes_df = readLikes(train_data_fp)
    testprofile_df = readProfile(test_data_fp)
    testlikes_df = readLikes(test_data_fp)


    pred_labels = ['age_group','gender','ope','con','ext','agr','neu']
    
    #EXTERNAL EVALUATION
    #X_train, y_train = preprocessLikes(profile_df,likes_df,pred_labels)
    #X_test, y_test = preprocessLikes(testprofile_df,testlikes_df,pred_labels)
    
    #INTERNAL EVALUATION
    X, y = preprocessLikes(profile_df, likes_df,pred_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1501)

    X_test = X_test.reset_index()
    predictions = X_test.copy()
    #NB Model - Likes
    genderMap = {0.0:'male',1.0:'female'}
    predictions['gender'] = NaiveBayesModel(X_train, X_test, y_train['gender'], y_test['gender'])
    #predictions['gender'] = LogisticRegressionModel(X_train,X_test,y_train['gender'],y_test['gender'])
    #predictions['age_group'] = NaiveBayesModel(X_train, X_test, y_train['age_group'], y_test['age_group'])
    predictions['age_group'] = LogisticRegressionModel(X_train,X_test,y_train['age_group'],y_test['age_group'])
    #predictions['ope'] = LinearRegressionModel(X_train,X_test,y_train['ope'],y_test['ope'])
    #predictions['ope'] = NeuralNetwork(X_train,X_test,y_train['ope'],y_test['ope'])
    predictions['gender'] = predictions['gender'].map(genderMap)


    writeUsers(pred_fp, predictions)


#Support function
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

#MODELS 
    
def NeuralNetwork(X_train,X_test1,y_train,y_test):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



    countVec = CountVectorizer()
    X_train = countVec.fit_transform(X_train['like_id'])
    X_train = convert_sparse_matrix_to_sparse_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)
    
    clf = Sequential()
    clf.add(Dense(84,input_dim=X_train.shape[1],activation='relu'))
    clf.add(Dropout(0.2))
    clf.add(Dense(38,activation='relu'))
    clf.add(Dropout(0.2))
    clf.add(Dense(1,activation='relu'))
    clf.compile(optimizer='adam',loss='mse',metrics=['mse'])
    
    X_train = tf.sparse.reorder(X_train)
    X_test2 = countVec.transform(X_test1['like_id'])
    X_test2 = convert_sparse_matrix_to_sparse_tensor(X_test2)
    X_test2 = tf.sparse.reorder(X_test2)
    y_test2 = tf.convert_to_tensor(y_test)
    
    
    history = clf.fit(X_train,y_train,epochs=50,batch_size=25,validation_data=[X_test2,y_test2])
   
    y_predict = clf.predict(X_test2)
    return y_predict

    #Evaluation
    y_base = pd.DataFrame(y_test)
    #CHANGE TO TARGET FEATURE BASELINE PREDICTION
    y_base['ope'] = str(3.909)

    #Plotting Trainving and Validation Loss Curves
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('NN MSE loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("MSE: %.2f" % math.sqrt(mean_squared_error(y_test,y_predict)))
    print("MSE: %.2f" % math.sqrt(mean_squared_error(y_test,y_base)))

def LinearRegressionModel(X_train,X_test1,y_train,y_test):
    countVec = CountVectorizer()
    X_train = countVec.fit_transform(X_train['like_id'])
    
    clf = LinearRegression(fit_intercept=True)
    clf.fit(X_train, y_train)

    X_test2 = countVec.transform(X_test1['like_id'])
    y_predict = clf.predict(X_test2)
    return y_predict

    #Evaluation
    y_base = pd.DataFrame(y_test)
    #CHANGE TO TARGET FEATURE BASELINE PREDICTION
    y_base['ope'] = str(3.909)

    #Accuracy Checking
    print("MSE: %.2f" % math.sqrt(mean_squared_error(y_test,y_predict)))
    print("MSE: %.2f" % math.sqrt(mean_squared_error(y_test,y_base)))

def LogisticRegressionModel(X_train,X_test1,y_train,y_test):
    countVec = CountVectorizer()
    X_train = countVec.fit_transform(X_train['like_id'])
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    X_test2 = countVec.transform(X_test1['like_id'])
    y_predict = clf.predict(X_test2)
    return y_predict

    #Accuracy Checking

    #Evaluation
    y_base = pd.DataFrame(y_test)
    #CHANGE TO TARGET FEATURE BASELINE PREDICTION
    y_base['gender'] = 1.0
    
    print("Accuracy: %.2f" % accuracy_score(y_test,y_predict))
    print("Baseline Accuracy: %.2f" % accuracy_score(y_test,y_base))
    classes = [0,1]
    #classes = ['xx-24','25-34','35-49','50-xx']
    cnf_matrix = confusion_matrix(y_test,y_predict,labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)


def NaiveBayesModel(X_train,X_test1,y_train,y_test):

    #counting like ids tokens from combined strings
    countVec = CountVectorizer(max_df=0.9,min_df=10)
    X_train = countVec.fit_transform(X_train["like_id"])

    #Training
    clf = MultinomialNB()
    clf.fit(X_train,y_train)

    #Testing
    X_test2 = countVec.transform(X_test1['like_id'])
    y_predict = clf.predict(X_test2)
    return y_predict

    #Evaluation
    y_base = pd.DataFrame(y_test)
    #CHANGE TO TARGET FEATURE BASELINE PREDICTION
    y_base['age_group'] = 'xx-24'
    
    #Accuracy Checking
    print("Accuracy: %.2f" % accuracy_score(y_test,y_predict))
    print("Accuracy: %.2f" % accuracy_score(y_test,y_base))
    #classes = [0,1]
    #classes = ['xx-24','25-34','35-49','50-xx']
    #cnf_matrix = confusion_matrix(y_test,y_predict,labels=classes)
    #print("Confusion matrix:")
    #print(cnf_matrix)


#PREPROCESSING
def preprocessLikes(userData,likeData,pred_label):
    #convert Relation Table to likeID string for count vectorization
    userLikesTemp = likeData.groupby(['userid']).agg({'like_id': ' '.join})
    if(userData.loc[0,'age'] == '-'):
        userData['age'] = 1.0
    bins = [0.0,25.0,35.0,50.0,np.inf]
    names = ['xx-24','25-34','35-49','50-xx']
    userData['age_group'] = pd.cut(userData['age'],bins,labels=names)

    finalSet = pd.merge(userData, userLikesTemp, left_on='userid', right_on='userid')
    dropSet = finalSet.copy()
    dropSet = dropSet.drop(columns=['age'])
    X = dropSet.drop(columns = pred_label)
    y = finalSet[pred_label]
    return X,y


def age_to_group(age):
    if age <= 24:
        return 'xx-24'
    elif 25 <= age <= 34:
        return '25-34'
    elif 35 <= age <= 49:
        return '35-49'
    else:
        return '50-xx'
    

#OS I/O
def readLikes(data_fp):
    with open(os.path.join(data_fp,"relation/relation.csv"),'r') as f:
         userInfo = pd.read_csv(f)
   
    userInfo['like_id'] = userInfo['like_id'].apply(str)
    return userInfo
   

def writeUsers(pred_fp, userIDs): 
    if (not os.path.exists(pred_fp)):
        os.mkdir(pred_fp)
    for index,user in userIDs.iterrows():
        if index > len(userIDs):
            break
        else:
            writeUser(pred_fp,user) 


def writeUser(pred_fp, userIDs):

    userFile = str(userIDs['userid']) + '.xml'
    #change to column name of prediction or baseline results
    age_group = userIDs['age_group']
    userID = userIDs['userid']
    gender = userIDs['gender']
    open_score = str(3.909)
    con_score = str(3.446)
    ext_score = str(3.487)
    agree_score = str(3.584)
    neur_score = str(2.732)

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

def readProfile(data_fp):
    with open(os.path.join(data_fp,"profile/profile.csv"),'r') as f:
         userInfo = pd.read_csv(f)
    return userInfo

if __name__ == "__main__":
        main()

