#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import albumentations as A
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Data Loading and Labeling
data_dir = r'C:\Users\vishn\Desktop\mini project\DogSkinDisease_combined'  # ***ADJUST THIS PATH***
classes = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis'] #updated classes

images = []
labels = []

for i, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        images.append(image_path)
        labels.append(i)  # Numerical label for each class

# Data Splitting
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Data Preprocessing and Augmentation
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_and_preprocess_image(image_path, label):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    transformed_image = transformed['image']
    return transformed_image, label

# Create TensorFlow Datasets
def create_tf_dataset(images, labels):
    image_label_pairs = list(zip(images, labels))
    images_tf = []
    labels_tf = []
    for image_path, label in image_label_pairs:
        image, label_num = load_and_preprocess_image(image_path, label)
        images_tf.append(image)
        labels_tf.append(label_num)
    return tf.data.Dataset.from_tensor_slices((images_tf, labels_tf)).batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset = create_tf_dataset(train_images, train_labels)
val_dataset = create_tf_dataset(val_images, val_labels)
test_dataset = create_tf_dataset(test_images, test_labels)

# Build the Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x) #number of classes is now automatically set.
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze Base Model Layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)  # Adjust epochs as needed

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy}')

# Save the Model
model.save('dog_skin_disease_model.h5')


# In[12]:


pip install tensorflow scikit-learn opencv-python albumentations


# In[ ]:


import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import albumentations as A
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Data Loading and Labeling
data_dir = r'C:\Users\vishn\Desktop\mini project\DogSkinDisease_combined'  # ***ADJUST THIS PATH***
classes = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis'] #updated classes

images = []
labels = []

for i, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        images.append(image_path)
        labels.append(i)  # Numerical label for each class

# Data Splitting
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Data Preprocessing and Augmentation
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_and_preprocess_image(image_path, label):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    transformed_image = transformed['image']
    return transformed_image, label

# Create TensorFlow Datasets
def create_tf_dataset(images, labels):
    image_label_pairs = list(zip(images, labels))
    images_tf = []
    labels_tf = []
    for image_path, label in image_label_pairs:
        image, label_num = load_and_preprocess_image(image_path, label)
        images_tf.append(image)
        labels_tf.append(label_num)
    return tf.data.Dataset.from_tensor_slices((images_tf, labels_tf)).batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset = create_tf_dataset(train_images, train_labels)
val_dataset = create_tf_dataset(val_images, val_labels)
test_dataset = create_tf_dataset(test_images, test_labels)

# Build the Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x) #number of classes is now automatically set.
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze Base Model Layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)  # Adjust epochs as needed

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy}')

# Save the Model
model.save('dog_skin_disease_model.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#from google.colab import drive

# Mount Google Drive
#drive.mount('/content/drive')

preprocessed_data_dir = r'C:\Users\vishn\Desktop\tfDataset\tfDataset'  # ***ADJUST THIS PATH IF NEEDED***

classes = ['Bacterial_dermatosis_tf_augmented', 'Fungal_infections_tf_augmented', 'Healthy_tf_augmented', 'Hypersensitivity_allergic_dermatosis_tf_augmented']

images = []
labels = []

for i, class_name in enumerate(classes):
    class_dir = os.path.join(preprocessed_data_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Error: Directory not found: {class_dir}")
        continue

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        images.append(image_path)
        labels.append(i)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

rotation_layer = RandomRotation(0.2)
zoom_layer = RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))

def create_tf_dataset(images, labels, augment=False):
    def load_and_decode(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0

        if augment:
            image = tf.image.random_flip_left_right(image)
            image = rotation_layer(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = zoom_layer(image)

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(load_and_decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(train_images, train_labels, augment=True)
val_dataset = create_tf_dataset(val_images, val_labels)
test_dataset = create_tf_dataset(test_images, test_labels)

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping, reduce_lr])

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy}')

# Save the Model (as .h5)
model.save('dog_skin_disease_densenet121_model.h5')

# Fine-tuning
fine_tune_at = len(base_model.layers) // 2
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping, reduce_lr])

test_loss_fine, test_accuracy_fine = model.evaluate(test_dataset)
print(f'Fine Tuned Test Accuracy: {test_accuracy_fine}')


model.save('dog_skin_disease_densenet121_finetuned_model.h5')


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('dog_skin_disease_densenet121_finetuned_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as: dog_skin_disease_densenet121_finetuned_model.tflite")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




