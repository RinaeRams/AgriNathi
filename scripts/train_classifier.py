"""Train a classifier on a directory-structured dataset using MobileNetV2 transfer learning.

Structure expected:
  data/plant_dataset/train/<class>/*.jpg
  data/plant_dataset/val/<class>/*.jpg

This script saves a Keras model to models/plant_mobilenetv2.h5 and labels to models/labels.json
"""
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATA_DIR = os.environ.get('PLANT_DATA_DIR', 'data/plant_dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
MODEL_OUT = os.environ.get('MODEL_OUT', 'models/plant_mobilenetv2.h5')
LABELS_OUT = os.environ.get('LABELS_OUT', 'models/labels.json')


def build_and_train():
    train_path = os.path.join(DATA_DIR, 'train')
    val_path = os.path.join(DATA_DIR, 'val')

    train_ds = image_dataset_from_directory(train_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = image_dataset_from_directory(val_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print('Classes:', class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Optionally fine-tune a few layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=3)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    with open(LABELS_OUT, 'w') as f:
        json.dump(class_names, f)
    print('Saved model to', MODEL_OUT)


if __name__ == '__main__':
    build_and_train()
