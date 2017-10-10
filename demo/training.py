from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
#from keras.applications import imagenet_utils
import matplotlib.pyplot as plt

#Formato de las imagenes de entrada
input_shape=(150, 150, 3)

#Construye un modelo secuencial
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Compila el modelo
model.compile(loss='binary_crossentropy', #binario pues solo hay dos clases
              optimizer='rmsprop',
              metrics=['accuracy'])

#Pocas epocas para que ejecute en un CPU
n_epochs = 32 #se puede incrementar a 100 o mas si se tienen varios CUDA Cores
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

print(validation_generator)

'''
for attr, value in validation_generator.__dict__.iteritems():
        print attr, value
'''

#Entrena el algoritmo
history_fg = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)


acc = history_fg.history['acc']
val_acc = history_fg.history['val_acc']
loss = history_fg.history['loss']
val_loss = history_fg.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r.')
plt.plot(epochs, val_acc, 'r')
plt.title('Training and validation accuracy')
plt.savefig('accuracy.png')

plt.figure()
plt.plot(epochs, loss, 'r.')
plt.plot(epochs, val_loss, 'r-')
plt.title('Training and validation loss')
plt.savefig('loss.png')


#Guarda los pesos
model.save_weights('_first_try.h5')  # always save your weights after training or during training