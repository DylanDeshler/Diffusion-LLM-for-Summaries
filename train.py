import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # The model building API we'll be using
from tensorflow.keras.layers import Dropout, Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D # The layers we'll need for our first models
# from tensorflow.keras.utils import np_utils # Some utilities for formatting numpy arrays
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kymatio.keras import Scattering2D
import efficientnet.tfkeras as efn

from imgaug import augmenters as iaa
import imgaug as ia

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Some other potentially useful modules
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import os
import sys
import cv2
import glob

tf.random.set_seed(0)
np.random.seed(0)
ia.seed(0)

CROP_PORTION = 0.75
IMG_SIZE = 244
MEAN = np.asarray([0.485, 0.456, 0.406]).astype('float32')
STD = np.asarray([0.229, 0.224, 0.225]).astype('float32')

rand_aug = iaa.RandAugment(n=(3, 7), m=11)

def augment(x):
    x = rand_aug(image=x.astype('uint8')).astype('float32')
    x = np.clip(x, 0, 255)
    # x = x / 255
    # x = (x - MEAN) / STD

    x = (x / 127.5) - 1
    return x

def gaussian_noise(x):
    noise = np.random.normal(0, 0.1, (IMG_SIZE, IMG_SIZE, 3))
    x += noise
    return x

def load_data(split=0.2):
    data_dir = "/home/ddeshler/final_project/Final_Project_Dataset/"

    pics = os.listdir(data_dir)
    pic_labels = [x[:2] for x in pics]
    print(set(pic_labels))

    X = []

    for img in pics:
        img = cv2.imread(os.path.join(data_dir, img),0)
        X.append(np.stack([img, img, img], axis=-1))
        # X.append(img)

    # integer encode
    label_encoder = LabelEncoder()
    pic_int_encoding = label_encoder.fit_transform(pic_labels)
    print(pic_int_encoding[:10])
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    pic_encoded = onehot_encoder.fit_transform(pic_int_encoding.reshape(-1, 1))
    print(pic_encoded[:10])
    y = pic_encoded

    X = np.asarray(X).astype('float32')
    print(X.shape)

    # same split everytime
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    x_train = tf.image.resize(
        tf.image.central_crop(
            x_train.reshape(x_train.shape[0], 75, 75, 3).astype('float32'), 
            CROP_PORTION
        ), 
        (IMG_SIZE, IMG_SIZE)
    )

    x_test = tf.image.resize(
        tf.image.central_crop(
            x_test.reshape(x_test.shape[0], 75, 75, 3).astype('float32'), 
            CROP_PORTION
        ), 
        (IMG_SIZE, IMG_SIZE)
    )

    print(x_train.shape)
    print('Data scale: ', np.min(X), np.mean(X), np.max(X))

    # x_test = x_test / 255
    # x_test = (x_test - MEAN) / STD

    x_test = (x_test / 127.5) - 1

    return x_train, x_test, y_train, y_test

def build_model(input_shape, num_classes):
    model = Sequential()
    # We defined the size of the images earlier, and will use that to define the input_shape
    # We will also define our kernel/filter to be 5x5, or process 25 pixels at a time
    # The strides parameter dictates how the filter passes over our images, in this case 1x and 1y at a time
    model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(75, (3, 3), activation='relu'))
    # Next we add a Pooling layer, it's 2D because we are working a static image
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(150, (3, 3), activation='relu'))
    model.add(Conv2D(150, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # After our set of convolutions, we flatten the image to create a ANN
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    
    # The softmax function will need to know how many classes we've defined
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return(model)

def build_effnet(input_shape, classes, trainable=False):
    # noisy-student pretrained weights are substantially better than imagenet
    backbone = efn.EfficientNetB0(weights='noisy-student', include_top=False, input_shape=input_shape)
    # backbone = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape=input_shape)
    backbone.trainable = trainable

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)    # so we don't affect the bn when full model fine-tuning
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.6)(x)
    # x = Dense(128, activation='swish')(x)
    x = Dropout(0.8)(x) # super high dropout for regularization
    regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1)
    x = Dense(classes)(x)
    model = Model(inputs, x)
    # model = Sequential([
    #     backbone,
    #     GlobalAveragePooling2D(),
    #     Dropout(0.2),
    #     # Dense(64, activation='relu'),
    #     Dense(y_train[0].shape[0], activation='softmax')
    # ])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), 
        metrics=['accuracy']
    )

    return model

def build_wavenet(input_shape, classes):
    inputs = Input(shape=(75, 75))
    x = Scattering2D(J=3, L=8)(inputs)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=['accuracy'])

    return model

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_data()
    input_shape = x_train[0].shape
    batch_size = 64

    # I messed around with ensembling trained efficient nets because they had large variances
    # Didn't work too well, gets an extra 1-2% for an enormous cost
    if sys.argv[1] == 'eval':
        model_weights = glob.glob("/home/ddeshler/final_project/models/model*.h5")
        model = build_effnet(input_shape, y_train[0].shape[0], trainable=True)

        preds = []
        # average over model outputs
        for weights in model_weights:
            model.load_weights(weights)
            preds.append(model.predict(x_test, batch_size=batch_size))
        
        preds = np.stack(preds, axis=-1)
        preds = np.mean(preds, axis=-1)
        print(np.mean(np.equal(np.argmax(preds, axis=-1), np.argmax(y_test, axis=-1)).astype('float32')))
    elif sys.argv[1] == 'train':
        datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=40,
            # width_shift_range=int(0.2*IMG_SIZE),
            # height_shift_range=int(0.2*IMG_SIZE),
            # brightness_range=(0, 0),
            # horizontal_flip=True,
            # vertical_flip=True,
            # validation_split=0.2,
            preprocessing_function=augment,
        )
        datagen.fit(x_train)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3)

        for i in range(10):
            model = build_effnet(input_shape, y_train[0].shape[0])
            # train to convergence with classification layer to prevent large gradients from destroying base weights
            model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),#, subset='training'),
                # validation_data=datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'), 
                epochs=35, 
                batch_size=batch_size, 
                shuffle=True,
                callbacks=[early_stopping, reduce_lr],
            )

            model.evaluate(x_test, y_test, batch_size=batch_size)

            # full model fine-tune
            model.trainable = True
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                metrics=['accuracy']
            )
            model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),#, subset='training'),
                # validation_data=datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'), 
                epochs=80,
                batch_size=batch_size, 
                shuffle=True,
                callbacks=[early_stopping, reduce_lr],
            )
            loss = model.evaluate(x_test, y_test, batch_size=batch_size)
            save_path = f'/home/ddeshler/final_project/models/modelB2_{i}_{loss}.h5'
            model.save_weights(save_path)
    # Trains a single efficient net, I use validation data when iterating on the network, but include that in the final training
    elif sys.argv[1] == 'exp':
        datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=40,
            # width_shift_range=int(0.2*IMG_SIZE),
            # height_shift_range=int(0.2*IMG_SIZE),
            # brightness_range=(0, 0),
            # horizontal_flip=True,
            # vertical_flip=True,
            validation_split=0.2,
            preprocessing_function=augment,
        )
        datagen.fit(x_train)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3)

        model = build_effnet(input_shape, y_train[0].shape[0])
        # train to convergence with classification layer to prevent large gradients from destroying base weights
        model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),#, subset='training'),
            validation_data=datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'), 
            epochs=35, 
            batch_size=batch_size, 
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
        )

        model.evaluate(x_test, y_test, batch_size=batch_size)

        # full model fine-tune
        model.trainable = True
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
            metrics=['accuracy']
        )
        model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),#, subset='training'),
            validation_data=datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'), 
            epochs=80,
            batch_size=batch_size, 
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
        )
        loss = model.evaluate(x_test, y_test, batch_size=batch_size)
        save_path = f'/home/ddeshler/final_project/models/modelB0_{loss}.h5'
        model.save_weights(save_path)
    else:
        raise NotImplemntedError(f'{sys.argv[1]} has not been implemented')


    # model.fit(
    #     x_train, y_train,
    #     validation_split=0.1, 
    #     epochs=100, 
    #     batch_size=batch_size, 
    #     shuffle=True
    # )
    # model.evaluate(x_test, y_test, batch_size=batch_size)