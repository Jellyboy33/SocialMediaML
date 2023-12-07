import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import conv_model

def process_data(data):
    X = data['imgs']
    y_gender = np.array(data['genders'])
    y_age = np.array(data['age_groups'])    
    X = np.array(X)

    X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(X, y_gender, y_age, test_size=0.1, random_state=42)
    y_gender_train = to_categorical(y_gender_train)
    y_gender_test = to_categorical(y_gender_test)
    y_age_train = to_categorical(y_age_train)
    y_age_test = to_categorical(y_age_test)
    # y_gender = to_categorical(y_gender)
    # y_age = to_categorical(y_age)

    # normalize inputs from 0-255 to 0-1
    # X = X / 255
    X_train = X_train / 255    
    X_test = X_test / 255    

    # build the model
    model = conv_model()
    # save the model
    model.save('./age_gender_model.keras')
    # Fit the model
    model.fit(X_train, y={'gender_out': y_gender_train, 'age_out': y_age_train}, validation_data=(X_test, {'gender_out': y_gender_test, 'age_out': y_age_test}), epochs=10, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, {'gender_out': y_gender_test, 'age_out': y_age_test}, verbose=0)
    print(scores)