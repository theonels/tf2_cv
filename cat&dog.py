import tensorflow as tf
import os
# load data
_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip',origin=_url,extract=True)
Path = os.path.join(os.path.dirname(path_to_zip),"cats_and_dogs_filtered")

train_dir = os.path.join(Path,'train')
validation_dir = os.path.join(Path,'validation')
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

num_train = num_cats_tr + num_dogs_tr
num_validation = num_cats_val + num_dogs_val
# print(num_train)
# print(num_validation)

