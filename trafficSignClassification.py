

import warnings
warnings.filterwarnings("ignore")


# In[176]:


# import libraries 
import pickle
import seaborn as sns
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random


# In[177]:


# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[178]:


X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# In[179]:


X_train.shape


# In[180]:


y_train.shape


# # STEP 2: IMAGE EXPLORATION

# In[181]:


i = 1001
plt.imshow(X_train[i]) # Show images are not shuffled
y_train[i]


# # STEP 3: DATA PEPARATION

# In[182]:


## Shuffle the dataset 
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


# In[183]:


X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray  = np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray  = np.sum(X_validation/3, axis=3, keepdims=True) 


# In[184]:


X_train_gray_norm = (X_train_gray - 128)/128 
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128


# In[185]:


X_train_gray.shape


# In[186]:


i = 610
plt.imshow(X_train_gray[i].squeeze(), cmap='gray')
plt.figure()
plt.imshow(X_train[i])


# # STEP 4: MODEL TRAINING

# The model consists of the following layers: 
# 
# - STEP 1: THE FIRST CONVOLUTIONAL LAYER #1
#     - Input = 32x32x1
#     - Output = 28x28x6
#     - Output = (Input-filter+1)/Stride* => (32-5+1)/1=28
#     - Used a 5x5 Filter with input depth of 3 and output depth of 6
#     - Apply a RELU Activation function to the output
#     - pooling for input, Input = 28x28x6 and Output = 14x14x6
# 
# 
#     * Stride is the amount by which the kernel is shifted when the kernel is passed over the image.
# 
# - STEP 2: THE SECOND CONVOLUTIONAL LAYER #2
#     - Input = 14x14x6
#     - Output = 10x10x16
#     - Layer 2: Convolutional layer with Output = 10x10x16
#     - Output = (Input-filter+1)/strides => 10 = 14-5+1/1
#     - Apply a RELU Activation function to the output
#     - Pooling with Input = 10x10x16 and Output = 5x5x16
# 
# - STEP 3: FLATTENING THE NETWORK
#     - Flatten the network with Input = 5x5x16 and Output = 400
# 
# - STEP 4: FULLY CONNECTED LAYER
#     - Layer 3: Fully Connected layer with Input = 400 and Output = 120
#     - Apply a RELU Activation function to the output
# 
# - STEP 5: ANOTHER FULLY CONNECTED LAYER
#     - Layer 4: Fully Connected Layer with Input = 120 and Output = 84
#     - Apply a RELU Activation function to the output
# 
# - STEP 6: FULLY CONNECTED LAYER
#     - Layer 5: Fully Connected layer with Input = 84 and Output = 43

# In[187]:


# Import train_test_split from scikit library

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split


# In[188]:


image_shape = X_train_gray[i].shape


# In[189]:


cnn_model = Sequential()

cnn_model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)))
cnn_model.add(AveragePooling2D(pool_size=(2, 2)))

cnn_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
cnn_model.add(AveragePooling2D(pool_size=(2, 2)))

cnn_model.add(Flatten())

cnn_model.add(Dense(units=120, activation='relu'))

cnn_model.add(Dense(units=84, activation='relu'))

cnn_model.add(Dense(units=43, activation = 'softmax'))


# In[190]:


cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


# In[191]:


history = cnn_model.fit(X_train_gray_norm,
                        y_train,
                        batch_size=500,
                        epochs=50,  # Corrected argument name
                        verbose=1,
                        validation_data=(X_validation_gray_norm, y_validation))


# # STEP 5: MODEL EVALUATION

# In[192]:


score = cnn_model.evaluate(X_test_gray_norm, y_test,verbose=0)
print('Test Accuracy : {:.4f}'.format(score[1]))


# In[193]:


history.history.keys()


# In[194]:


accuracy = history.history['accuracy']  # Updated key
val_accuracy = history.history['val_accuracy']  # Updated key
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()


# In[195]:


plt.plot(epochs, loss, 'ro', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[197]:


# Get the predictions for the test data
predictions = cnn_model.predict(X_test_gray_norm)

# Convert predictions to class indices
predicted_classes = predictions.argmax(axis=-1)

# Get the indices to be plotted
y_true = y_test


# In[198]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25,25))
sns.heatmap(cm, annot=True)


# In[199]:


L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Prediction={}\n True={}".format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)

