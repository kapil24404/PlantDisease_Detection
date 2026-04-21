import os
import kagglehub
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
import shutil

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def download_and_split_data(base_output_dir="dataset"):
    print("Downloading dataset using kagglehub...")
    # This downloads the dataset. Kagglehub caches it.
    path = kagglehub.dataset_download("emmarex/plantdisease")
    print(f"Dataset downloaded to: {path}")
    
    # In 'emmarex/plantdisease', the main directory containing classes is usually 'PlantVillage'
    # Let's find the correct directory containing subfolders (classes).
    source_dir = path
    for root, dirs, files in os.walk(path):
        if len(dirs) > 5: # Assuming more than 5 classes
            source_dir = root
            break
            
    print(f"Using source directory for splitting: {source_dir}")

    if not os.path.exists(base_output_dir):
        print(f"Splitting data into {base_output_dir} (70% train, 15% val, 15% test)...")
        # Split into train, val, test (0.7, 0.15, 0.15)
        splitfolders.ratio(source_dir, output=base_output_dir,
                           seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None)
        print("Data splitting complete.")
    else:
        print(f"Directory {base_output_dir} already exists. Skipping splitting.")

def get_data_generators(dataset_dir="dataset"):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and Testing data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_generator.num_classes

if __name__ == "__main__":
    download_and_split_data()
    train_gen, val_gen, test_gen, num_classes = get_data_generators()
    print(f"Number of classes detected: {num_classes}")
