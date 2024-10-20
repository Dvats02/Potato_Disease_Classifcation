
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models, layers

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3 #RGB
EPOCHS=50

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'PlantVillage',
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names
class_names

len(dataset) #Since each image converted in to images of 32 batch size Total 68 batches possible

for image_batch, label_batch in dataset.take(1): #Image batch contians 32 images label batch labels images in 3 classes
  print(image_batch.shape)
  print(label_batch.numpy())#Since every batch is tensor you need to convert it to numpy

# 80% Training
# 20%=> 10% Validation, 10%Test

def get_dataset_partition(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
  ds_size=len(ds)
  if shuffle:
    ds=ds.shuffle(shuffle_size, seed=12)

    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)

    train_ds=ds.take(train_size)

    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partition(dataset)

len(train_ds)

len(val_ds)

len(test_ds)

train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),#Used such that if some image comes small aor larger yu can resize it fianly
    layers.Rescaling(1.0/255)
])

#Data_Augmentation_process
data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Use Conv2D instead of conv2D
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Remove unnecessary input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Remove unnecessary input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Remove unnecessary input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Remove unnecessary input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  # Remove unnecessary input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # Use Dense instead of dense
    layers.Dense(CHANNELS, activation='softmax')  # Use Dense instead of dense
])

model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(
    train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
    validation_data=val_ds
)

scores=model.evaluate(test_ds)

M1="my_model"
model.save("MMy_model.keras")
