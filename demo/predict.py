import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.preprocessing import image
from keras.applications import imagenet_utils

#Formato de las imágenes de entrada
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

model.load_weights('first_try.h5')  # always save your weights after training or during training

### Predict ###

#img_path = 'data/validation/bear/1_90.jpg'
img_path = 'data/validation/elephant/5688.jpg'

#carga la imagen
img = image.load_img(img_path, target_size=input_shape)

#convertir en arreglo numpy
x = image.img_to_array(img)
#print(x[:4,:4,:])
#print('x array shape', x.shape)

plt.imshow(x)

#normalizar
x = x/255

#expande dimensiones para el modelo
x = np.expand_dims(x, axis=0)

print('x array expanded dims shape', x.shape)
#print(x[:,:4,:4,:])

#class_indices {'bear': 0, 'elephant': 1}
# Get the prediction.
pred = model.predict(x)
classes = ['bear', 'elephant']
print('predicted value: ',pred)
print('predicted class: ', classes[int(np.round(pred, 0)[0, 0])])

'''
batch_size = 16

predict_datagen = ImageDataGenerator(rescale=1./255)

predict_generator = predict_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

pred2 = model.predict_generator(predict_generator, steps=2000 // batch_size)

print('predict 2: ', pred2)

'''