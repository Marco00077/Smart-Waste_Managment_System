"""
Training script using transfer learning (MobileNetV2).
Works better with small datasets.
"""

import os
import tensorflow as tf
from tensorflow import keras
from simple_classifier import create_simple_model

def load_dataset(data_dir):
    """Load images with heavy augmentation for small datasets"""
    img_size = (224, 224)
    batch_size = 8  # Smaller batch for small dataset
    
    # Load training data
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Load validation data
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    
    # Cache and prefetch for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names

def train(data_dir, epochs=50, save_path='waste_classifier_model.h5'):
    """
    Train using transfer learning.
    More epochs needed for small datasets.
    """
    print("Loading dataset...")
    train_ds, val_ds, class_names = load_dataset(data_dir)
    
    print("\nCreating transfer learning model (MobileNetV2)...")
    model = create_simple_model(num_classes=len(class_names))
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    print("\nTraining model...")
    print("Note: With only 6 images, accuracy will be limited!")
    print("Recommended: 100+ images per class for good results.\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print("\nNote: Low accuracy is expected with only 6 images.")
    print("Add more images to improve performance!")
    print("="*60)
    
    return model, history

if __name__ == "__main__":
    DATA_DIR = "dataset"
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        print("\nPlease organize your dataset as:")
        print("dataset/")
        print("  biodegradable/")
        print("    image1.jpg")
        print("    image2.jpg")
        print("  non_biodegradable/")
        print("    image1.jpg")
        print("    image2.jpg")
    else:
        # Count images
        bio_count = len(os.listdir(os.path.join(DATA_DIR, 'biodegradable')))
        non_bio_count = len(os.listdir(os.path.join(DATA_DIR, 'non_biodegradable')))
        total = bio_count + non_bio_count
        
        print(f"\nDataset Summary:")
        print(f"  Biodegradable: {bio_count} images")
        print(f"  Non-biodegradable: {non_bio_count} images")
        print(f"  Total: {total} images")
        
        if total < 20:
            print("\n⚠️  WARNING: Very small dataset!")
            print("   Recommended: 100+ images per class")
            print("   Current dataset may not train well.\n")
        
        model, history = train(DATA_DIR, epochs=50)
