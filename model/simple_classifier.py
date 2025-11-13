"""
Simple classifier for very small datasets.
Uses transfer learning with MobileNetV2 (pre-trained on ImageNet).
This works much better with limited data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_simple_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a simple transfer learning model using MobileNetV2.
    Works better with small datasets.
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build model with preprocessing layer
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # Rescale to [-1, 1] for MobileNetV2
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x)
    
    # Pre-trained base
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
