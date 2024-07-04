"""
## Practical 1 : Implement Feed-forward Neural Network and train the network with different optimizers and compare the results.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the feed-forward neural network model
def create_model():
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Function to train and evaluate the model
def train_and_evaluate(optimizer):
    model = create_model()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Train and evaluate models with different optimizers
optimizers = ['SGD', 'Adam', 'RMSprop']
results = {}
for optimizer in optimizers:
    accuracy = train_and_evaluate(optimizer)
    results[optimizer] = accuracy

# Print the results
for optimizer, accuracy in results.items():
    print(f'Accuracy with {optimizer} optimizer: {accuracy:.4f}')

"""## Practical 2 : Write a Program to implement regularization to prevent the model from overfitting"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the feed-forward neural network model with regularization
def create_regularized_model():
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# Function to train and evaluate the regularized model
def train_and_evaluate_regularized_model():
    model = create_regularized_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Train and evaluate the regularized model
accuracy_regularized = train_and_evaluate_regularized_model()

print(f'Accuracy with regularization: {accuracy_regularized:.4f}')

"""## Practical 3 : Implement deep learning for recognizing classes for datasets like CIFAR-10 images for previously unseen images and assign them to one of the 10 classes."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define the CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Function to train and evaluate the CNN model
def train_and_evaluate_cnn_model():
    model = create_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.1)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(np.argmax(y_test, axis=-1), y_pred)
    return accuracy

# Train and evaluate the CNN model
accuracy_cnn = train_and_evaluate_cnn_model()

print(f'Accuracy of the CNN model on CIFAR-10 test set: {accuracy_cnn:.4f}')

"""## Practical 4 : Implement deep learning for the Prediction of the autoencoder from the test data (e.g. MNIST data set)"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Define the autoencoder model
def create_autoencoder_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(784, activation='sigmoid')
    ])
    return model

# Function to train the autoencoder model
def train_autoencoder_model():
    model = create_autoencoder_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    return model

# Train the autoencoder model
autoencoder_model = train_autoencoder_model()

# Predict outputs using the trained autoencoder model
reconstructed_images = autoencoder_model.predict(X_test)

# Display original and reconstructed images
import matplotlib.pyplot as plt

n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

"""## Practical 5 : Implement Convolutional Neural Network for Digit Recognition on the MNIST Dataset"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the images to add a channel dimension (required for CNN)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# One-hot encode the target labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Function to train and evaluate the CNN model
def train_and_evaluate_cnn_model():
    model = create_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))
    return model

# Train and evaluate the CNN model
cnn_model = train_and_evaluate_cnn_model()

"""## Practical 6 : Write a program to implement Transfer Learning on the suitable dataset."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
# Define paths to the dataset
train_dir = '/kaggle/input/dogs-cats-images/dataset/training_set/'
test_dir = '/kaggle/input/dogs-cats-images/dataset/test_set/'
# Define constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
# Data augmentation for training set
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)
# Normalization for test set
test_datagen = ImageDataGenerator(rescale=1./255)
# Load and prepare data
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(IMAGE_SIZE, IMAGE_SIZE),
batch_size=BATCH_SIZE,
class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(IMAGE_SIZE, IMAGE_SIZE),
batch_size=BATCH_SIZE,
class_mode='binary'
)
# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE,
IMAGE_SIZE, 3))
# Freeze convolutional layers
for layer in vgg_model.layers:
  layer.trainable = False
# Create new model
model = Sequential([
vgg_model,
Flatten(),
Dense(512, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
train_generator,
steps_per_epoch=train_generator.samples // BATCH_SIZE,
epochs=10,
validation_data=test_generator,
validation_steps=test_generator.samples // BATCH_SIZE
)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print('Test accuracy:', test_acc)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Define data directories
train_dir = 'cats_vs_dogs/train'
validation_dir = 'cats_vs_dogs/validation'
test_dir = 'cats_vs_dogs/test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Create ImageDataGenerators for train, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

# Load pre-trained VGG16 model without top (fully connected layers)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the convolutional base
vgg_base.trainable = False

# Create a new model on top of the pre-trained base
model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test Accuracy:", test_accuracy)

"""## Practical 7 : Write a program for the Implementation of a Generative Adversarial Network for generating synthetic shapes (like digits)

"""

# New
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize data
X_train = X_train.astype('float32') / 255.0

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Generator
generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Combined model
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Training
epochs = 20000
batch_size = 128
for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=[batch_size, 100])
    generated_images = generator.predict(noise)
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    X = np.concatenate([real_images, generated_images])
    y_dis = np.zeros(2*batch_size)
    y_dis[:batch_size] = 0.9  # Label smoothing
    discriminator.trainable = True
    d_loss = discriminator.train_on_batch(X, y_dis)
    noise = np.random.normal(0, 1, size=[batch_size, 100])
    y_gen = np.ones(batch_size)
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, y_gen)
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

# Generate synthetic images
noise = np.random.normal(0, 1, size=[10, 100])
generated_images = generator.predict(noise)

# Display generated images
plt.figure(figsize=(10, 10))
for i in range(generated_images.shape[0]):
    plt.subplot(1, 10, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

"""## Practical 8 : Write a program to implement a simple form of a recurrent neural network.
a. E.g. (4-to-1 RNN) to show that the quantity of rain on a certain day also depends on the values of the previous day


b. LSTM for sentiment analysis on datasets like UMICH SI650 for similar

### A
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate synthetic data for demonstration purposes
# Each data point consists of the quantity of rain on a certain day (X) and the quantity of rain on the previous day (y)
X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]])
y = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Define the RNN model
model = Sequential([
    SimpleRNN(32, input_shape=(2, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X.reshape(-1, 2, 1), y, epochs=100)

# Predict the quantity of rain for a new day based on the quantity of rain on the previous day
new_day_rain = np.array([[0.9, 1.0]])  # Quantity of rain on the previous day
predicted_rain = model.predict(new_day_rain.reshape(-1, 2, 1))
print("Predicted quantity of rain for the new day:", predicted_rain[0][0])

"""### B
#### https://www.embedded-robotics.com/sentiment-analysis-using-lstm/
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = "/content/umich-sentiment-train.txt"
with open(data_path, 'r',encoding="utf8") as f:
    lines = f.readlines()

sentences = []
labels = []
for line in lines:
    parts = line.strip().split('\t')
    labels.append(int(parts[0]))
    sentences.append(parts[1])

# Preprocessing
max_features = 10000
maxlen = 100
embedding_size = 128

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test))

# Evaluate
score, acc = model.evaluate(X_test, y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)


#If the code gives error,
# comment out this line : from keras.preprocessing.text import Tokenizer and theis line "import tensorflow as tf"
# replace this line : tokenizer = Tokenizer(num_words=max_features) with this line "tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)"



"""# Practical 9: Write a program for object detection from the image/video

Link: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
"""

#Image Detection
import cv2
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("image.jpg") #Download image from the link

# OpenCV opens images as BRG
# but we want it as RGB and
# we also need a grayscale
# version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Creates the environment
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

# Use minSize because for not
# bothering with extra-small
# dots that would look like STOP signs
stop_data = cv2.CascadeClassifier('stop_data.xml')
found = stop_data.detectMultiScale(img_gray,
								minSize =(20, 20))
# Don't do anything if there's
# no sign
amount_found = len(found)

if amount_found != 0:

	# There may be more than one
	# sign in the image
	for (x, y, width, height) in found:

		# We draw a green rectangle around
		# every recognized sign
		cv2.rectangle(img_rgb, (x, y),
					(x + height, y + width),
					(0, 255, 0), 5)

# Creates the environment of
# the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

"""## Practical 10 : Write a program for object detection using pre-trained models to use object detection."""

