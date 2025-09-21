import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os

# Checking GPU 
tf.config.list_physical_devices()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')  # use GPU
        print("Train with GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU, Train with CPU")

# load dataset 
train_data ='dataset/train/'
val_data ='dataset/test/'

# image augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )

val_datagen = ImageDataGenerator(rescale=1./255)

# read the image from folder
train_generator = train_datagen.flow_from_directory(train_data,
                                                    color_mode='grayscale',
                                                    target_size=(48, 48),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )

val_generator = val_datagen.flow_from_directory(val_data,
                                                color_mode='grayscale',
                                                target_size=(48, 48),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=True
                                                )

class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()

# model CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

train_path ='dataset/train/'
test_path ='dataset/test/'

num_train = 0
for root, dirs, files in os.walk(train_path):
    num_train += len(files)

num_test = 0
for root, dirs, files in os.walk(train_path):
    num_test += len(files)

print("GPU Available:", tf.config.list_physical_devices('GPU'))

history = model.fit(train_generator,
                    steps_per_epoch=num_train//32,
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=num_test//32)

model.save('model_50Epoch.h5')