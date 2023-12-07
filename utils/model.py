from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

def conv_model():
    input_shape = (128, 128, 1)
    inputs = Input((input_shape))
    conv1 = Conv2D(32, (5,5), activation='relu') (inputs)
    maxpool1 = MaxPooling2D((2,2)) (conv1)
    conv2 = Conv2D(64, (3,3), activation='relu') (maxpool1)
    maxpool2 = MaxPooling2D((2,2)) (conv2)
    conv3 = Conv2D(128, (3,3), activation='relu') (maxpool2)
    maxpool3 = MaxPooling2D((2,2)) (conv3)
    conv4 = Conv2D(256, (3,3), activation='relu') (maxpool3)
    maxpool4 = MaxPooling2D((2,2)) (conv4)
    flatten = Flatten() (maxpool4)
    dense_1 = Dense(256, activation='relu') (flatten)
    dense_2 = Dense(256, activation='relu') (flatten)
    dropout_1 = Dropout(0.2) (dense_1)
    dropout_2 = Dropout(0.1) (dense_2)

    gender_out = Dense(2, activation='sigmoid', name='gender_out') (dropout_1)
    age_out = Dense(4, activation='softmax', name='age_out') (dropout_2)

    model = Model(inputs=inputs,outputs=[gender_out, age_out])

    # Compile model
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])
    return model