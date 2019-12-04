from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
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

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_image_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir,
                                                           shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                           class_mode='binary')
validation_image_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_dir,
                                                                     target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                                     class_mode='binary')
# # Visualize training images
# sample_training_images, sample_training_lable = next(train_image_gen)
#
# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# plotImages(sample_training_images[:5])

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# show structure of this model
# model.summary()

history = model.fit_generator(
    train_image_gen,
    steps_per_epoch=num_train,
    epochs=epochs,
    validation_data=validation_image_gen,
    validation_steps=num_validation
)

with open("./history.txt","w") as f:
    for e,index in enumerate(history.epoch):
        f.write(e)
        if index<=len(history.epoch)-1:
            f.write(",")
        else:
            f.write("\n")
    f.write(json.dumps(history.history))

