import pandas as pd
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import sklearn
print(tf.__version__)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from PIL import Image
import os


from tensorflow import keras
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.list_physical_devices('GPU')
devices = tf.config.experimental.list_physical_devices('GPU')
print(devices)
for gpu in devices:
  tf.config.experimental.set_memory_growth(gpu, True)


BS = 64
EPOCHS = 300
STEPS_PER_EPOCH = 200
LR = 0.001
WEIGHT_DECAY = 0.0005

csv_df = '/home/contrastive_step_19_01/features/5k_PNG_2905_top_latlons.csv'
images_path = '/home/Landsat8/10k_5k_PNG_all/'
pretrained_weights = '/home/contrastive_step_19_01/weights/projection_weights_15_05.h5'
features_csv1 = '/home/contrastive_step_19_01/features/full_features_5km_27_05_top.csv'


print('Before reading images path')
image_filenames = [f for f in os.listdir(images_path) if f.endswith('.png')]
image_paths = [os.path.join(images_path, filename) for filename in os.listdir(images_path)]
print('Images found')
images = []


print('Before reading csv')
df = pd.read_csv(csv_df)
print(len(df.index), 'images available')
# print(df['pf_pr'].isnull().sum(), 'missing prevalence values')
df.head()



def get_encoder():
    inputs = tf.keras.layers.Input((224, 224, 3), name='Inputs_BaseEncoder')
    alpha = 0.2
    # Block 1 of convolutional layers
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv1_BaseEncoder')(inputs)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder1')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv2_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder2')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='Pool1_BaseEncoder')(x)
    
    # Block 2 of convolutional layers
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv3_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder3')(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv4_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder4')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='Pool2_BaseEncoder')(x)

    # Block 3 of convolutional layers
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv5_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder5')(x)
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv6_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder6')(x)
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv7_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder7')(x)
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv8_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder8')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='Pool3_BaseEncoder')(x)
    # Block 4 of convolutional layers
    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv9_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder9')(x)
    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv10_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder10')(x)
    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv11_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder11')(x)
    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv12_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder12')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='Pool4_BaseEncoder')(x)

    # Block 5 of convolutional layers
    x = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv13_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder13')(x)
    x = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv14_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder14')(x)
    x = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv15_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder15')(x)
    x = tf.keras.layers.Conv2D(1024, (3,3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Conv16_BaseEncoder')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder16')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='Pool5_BaseEncoder')(x)

    # Global average pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling2D(name='GAP_BaseEncoder')(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Dense1_BaseEncoder', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder17')(x)
    x = tf.keras.layers.Dense(2048, activation=tf.keras.layers.LeakyReLU(alpha=alpha), name='Dense2_BaseEncoder', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder18')(x)
    z = tf.keras.layers.Dense(2048, name='Dense3_BaseEncoder', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    z = tf.keras.layers.BatchNormalization(name='BN_BaseEncoder19')(z)

    f = tf.keras.Model(inputs, z, name='BaseEncoder')

    return f


get_encoder().summary()
projection = get_encoder()
projection.load_weights(pretrained_weights)

print(projection.layers[38])

rn50 = tf.keras.Model(projection.input, projection.layers[38].output)
rn50.summary()
print('Model loaded, weights attached')


features = []
for image_name in df['filenames']:
    image = tf.image.decode_png(tf.io.read_file(os.path.join(images_path, image_name)), channels = 3)
    features.append(rn50(image[tf.newaxis, ...]).numpy().flatten())



print(features[1])

print(df.head())



import csv
images_names = df['filenames']
lat = df['lat']
lon = df['lon']

with open(features_csv1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filenames',  'lat', 'lon'] + [f'feature_{i}' for i in range(len(features[0]))])

    for i, image_name in enumerate(images_names):
        writer.writerow([image_name, lat[i], lon[i]] + list(features[i]))     

print('Features Done')
