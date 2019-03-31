#CNN implementation to classify Cat and Dog images

# Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

#Create Model
model=Sequential()

#Add Convalutional layer to Model
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))

#Add Max Pool layer
model.add(MaxPool2D(pool_size=(2,2)))

#Adding second Convolutional and MaxPool layer
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

#Adding Flatten to the model
model.add(Flatten())

#Add ANN to the pipeline to classify images
model.add(Dense(units=128,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=1,activation='sigmoid',))

#Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Below code is from Keras Help document on Image preprocessing from directory
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=2000)