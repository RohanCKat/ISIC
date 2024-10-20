### !!! NOTICE !!!
### This peice of code cannt work within its own Enviroment. It needs the data/Librarys on our PC's to run.
### It also must be ran locally, due to the concerns of our privacy and our inexperience with port fowarding.
### Sorry for the inconvience - Coders A & B


### Importing TensorFlow And Keras For their API. Importing OS and shutil for the dataset - Coder B
import tensorflow as tf
import keras
import os, shutil

###Getting the images to train the AI on from my personal computer. These are all directories that I make using files on my PC.
### This part of the code was inspiried by Seciton 5.4 in the Book "Deep Learning With Pyton" by François Chollet
###Creating The 3 main branches of the data - Coder A
### Original Dataset
original_dataset_dir = 'F:\InvasiveSpecies'

###Directory for sorted data
base_dir = 'F:\Beta'
os.mkdir(base_dir)

###Training Directory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

###Validation Directory
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

###Test Directory
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

###Making 3 Directories using the Asian Longhorn Beetle
train_AsianLonghornBeetle_dir = os.path.join(train_dir, 'AsianLonghornBeetle')
os.mkdir(train_AsianLonghornBeetle_dir)

validation_AsianLonghornBeetle_dir = os.path.join(validation_dir, 'AsianLonghornBeetle')
os.mkdir(validation_AsianLonghornBeetle_dir)

test_AsianLonghornBeetle_dir = os.path.join(test_dir, 'AsianLonghornBeetle')
os.mkdir(test_AsianLonghornBeetle_dir)

###Getting the first 250 images for the training
fnames = ['AsianLonghornBeetle. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_AsianLonghornBeetle_dir, fname)
    shutil.copyfile(src, dst)


###Getting the proceding 125 images for the Validation
fnames = ['AsianLonghornBeetle. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_AsianLonghornBeetle_dir, fname)
    print(fnames)
    print(fname)
    shutil.copyfile(src, dst)
###Getting the last 125 images for the Testing
fnames = ['AsianLonghornBeetle. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_AsianLonghornBeetle_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for The Asian Tiger Mosquito
train_AsianTigerMosquito_dir = os.path.join(train_dir, 'AsianTigerMosquito')
os.mkdir(train_AsianTigerMosquito_dir)

validation_AsianTigerMosquito_dir = os.path.join(validation_dir, 'AsianTigerMosquito')
os.mkdir(validation_AsianTigerMosquito_dir)

test_AsianTigerMosquito_dir = os.path.join(test_dir, 'AsianTigerMosquito')
os.mkdir(test_AsianTigerMosquito_dir)

###Getting the first 250 images for the training
fnames = ['AsianTigerMosquito. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_AsianTigerMosquito_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['AsianTigerMosquito. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_AsianTigerMosquito_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['AsianTigerMosquito. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_AsianTigerMosquito_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for Canadian Thistle
train_CanadianThistle_dir = os.path.join(train_dir, 'CanadianThistle')
os.mkdir(train_CanadianThistle_dir)

validation_CanadianThistle_dir = os.path.join(validation_dir, 'CanadianThistle')
os.mkdir(validation_CanadianThistle_dir)

test_CanadianThistle_dir = os.path.join(test_dir, 'CanadianThistle')
os.mkdir(test_CanadianThistle_dir)

###Getting the first 250 images for the training
fnames = ['CanadianThistle. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_CanadianThistle_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['CanadianThistle. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_CanadianThistle_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['CanadianThistle. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_CanadianThistle_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for Garlic Mustard
train_GarlicMustard_dir = os.path.join(train_dir, 'GarlicMustard')
os.mkdir(train_GarlicMustard_dir)

validation_GarlicMustard_dir = os.path.join(validation_dir, 'GarlicMustard')
os.mkdir(validation_GarlicMustard_dir)

test_GarlicMustard_dir = os.path.join(test_dir, 'GarlicMustard')
os.mkdir(test_GarlicMustard_dir)

###Getting the first 250 images for the training
fnames = ['GarlicMustard. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_GarlicMustard_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['GarlicMustard. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_GarlicMustard_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['GarlicMustard. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_GarlicMustard_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for Garlic Mustard
train_GypsyMoth_dir = os.path.join(train_dir, 'GypsyMoth')
os.mkdir(train_GypsyMoth_dir)

validation_GypsyMoth_dir = os.path.join(validation_dir, 'GypsyMoth')
os.mkdir(validation_GypsyMoth_dir)

test_GypsyMoth_dir = os.path.join(test_dir, 'GypsyMoth')
os.mkdir(test_GypsyMoth_dir)

###Getting the first 250 images for the training
fnames = ['GypsyMoth. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_GypsyMoth_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['GypsyMoth. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_GypsyMoth_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['GypsyMoth. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_GypsyMoth_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for the Japanese Barberry
train_JapaneseBarberry_dir = os.path.join(train_dir, 'JapaneseBarberry')
os.mkdir(train_JapaneseBarberry_dir)

validation_JapaneseBarberry_dir = os.path.join(validation_dir, 'JapaneseBarberry')
os.mkdir(validation_JapaneseBarberry_dir)

test_JapaneseBarberry_dir = os.path.join(test_dir, 'JapaneseBarberry')
os.mkdir(test_JapaneseBarberry_dir)

###Getting the first 250 images for the training
fnames = ['JapaneseBarberry. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_JapaneseBarberry_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['JapaneseBarberry. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_JapaneseBarberry_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['JapaneseBarberry. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_JapaneseBarberry_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for the Japanese Knotweed
train_JapaneseKnotweed_dir = os.path.join(train_dir, 'JapaneseKnotweed')
os.mkdir(train_JapaneseKnotweed_dir)

validation_JapaneseKnotweed_dir = os.path.join(validation_dir, 'JapaneseKnotweed')
os.mkdir(validation_JapaneseKnotweed_dir)

test_JapaneseKnotweed_dir = os.path.join(test_dir, 'JapaneseKnotweed')
os.mkdir(test_JapaneseKnotweed_dir)

###Getting the first 250 images for the training
fnames = ['JapaneseKnotweed. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_JapaneseKnotweed_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['JapaneseKnotweed. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_JapaneseKnotweed_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['JapaneseKnotweed. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_JapaneseKnotweed_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for the Japanese Honeysuckle
train_JapaneseHoneySuckle_dir = os.path.join(train_dir, 'JapaneseHoneySuckle')
os.mkdir(train_JapaneseHoneySuckle_dir)

validation_JapaneseHoneySuckle_dir = os.path.join(validation_dir, 'JapaneseHoneySuckle')
os.mkdir(validation_JapaneseHoneySuckle_dir)

test_JapaneseHoneySuckle_dir = os.path.join(test_dir, 'JapaneseHoneySuckle')
os.mkdir(test_JapaneseHoneySuckle_dir)

###Getting the first 250 images for the training
fnames = ['JapaneseHoneySuckle. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_JapaneseHoneySuckle_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['JapaneseHoneySuckle. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_JapaneseHoneySuckle_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['JapaneseHoneySuckle. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_JapaneseHoneySuckle_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for the mile a minute
train_MileAMinute_dir = os.path.join(train_dir, 'MileAMinute')
os.mkdir(train_MileAMinute_dir)

validation_MileAMinute_dir = os.path.join(validation_dir, 'MileAMinute')
os.mkdir(validation_MileAMinute_dir)

test_MileAMinute_dir = os.path.join(test_dir, 'MileAMinute')
os.mkdir(test_MileAMinute_dir)

###Getting the first 250 images for the training
fnames = ['MileAMinute. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_MileAMinute_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['MileAMinute. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_MileAMinute_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['MileAMinute. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_MileAMinute_dir, fname)
    shutil.copyfile(src, dst)

###Making Directories for the Multi Flora Rose
train_MultifloraRose_dir = os.path.join(train_dir, 'MultifloraRose')
os.mkdir(train_MultifloraRose_dir)

validation_MultifloraRose_dir = os.path.join(validation_dir, 'MultifloraRose')
os.mkdir(validation_MultifloraRose_dir)

test_MultifloraRose_dir = os.path.join(test_dir, 'MultifloraRose')
os.mkdir(test_MultifloraRose_dir)

###Getting the first 250 images for the training
fnames = ['MultifloraRose. ({}).jpg'.format(i) for i in range(1,250)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_MultifloraRose_dir, fname)
    shutil.copyfile(src, dst)

###Getting the proceding 125 images for the Validation
fnames = ['MultifloraRose. ({}).jpg'.format(i) for i in range(250, 375)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_MultifloraRose_dir, fname)
    shutil.copyfile(src, dst)

###Getting the last 125 images for the Testing
fnames = ['MultifloraRose. ({}).jpg'.format(i) for i in range(375, 500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_MultifloraRose_dir, fname)
    shutil.copyfile(src, dst)

###Changing All of the images into a (150, 150, 3) tensor to fit into the Convnet layers
###Import the PreProcessing instructions from Keras - Coders A & B
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=20,class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=20,class_mode='categorical')


### From Keras Import the layers and the deep learning module - Coders A & B
from keras import layers
from keras import models 

### This is the actual module, and its telling the layers to be sequentials - Coders A & B
model = models.Sequential()

### These are the actual layers. Think of these like a coffee filter, with each layer adding a filter to the data being passed, which are the images that the AI needs to recognize.
### This Specific Layer is using Convnet2D, and MaxPooling2D layers. - Coders A & B
###Heavily Inspiried by Deep Learning With Pyton
model.add(layers.Conv2D(32, (3, 3), activation='relu',
### Note how the input shape has to be a tensor with the shape of (150, 150, 3) - Coders A & B
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
### This layer randomly drops out some data, in this case 50% of it - Coders A & B
model.add(layers.Dropout(0.5))
### This makes the data from the previous steps into a smaller tensor, allowing it through the Dense Layers - Coders A & B
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))

###This turnds the data into a prediction - Coders A & B
model.add(layers.Dense(12, activation='softmax'))

from keras import optimizers
model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)

model.save("F:\InvasiveSpeciesNeuralNet")

## Documentation - Coders A & B
## Tensorflow: https://www.tensorflow.org/
## Keras: https://keras.io/
## Database: https://invasive.org
## Deep Learning With Python: https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438
## Invasive.org: https://www.invasive.org/index.cfm
## SciPy: https://scipy.org/download/
## MatPlotLib: https://scipy.org/download/