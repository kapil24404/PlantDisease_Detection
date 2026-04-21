import os
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_setup import get_data_generators, download_and_split_data
from models import get_model

# Constants
EPOCHS = 5 # Can be adjusted (20-30 as per requirements)
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def plot_history(history, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_learning_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def train_and_evaluate():
    download_and_split_data()
    train_gen, val_gen, test_gen, num_classes = get_data_generators()
    
    # To map indices back to class names
    class_indices = train_gen.class_indices
    class_names = list(class_indices.keys())
    
    # Save class indices for inference
    np.save(os.path.join(MODELS_DIR, 'class_indices.npy'), class_indices)
    
    models_to_train = [
        'Hybrid CNN-Transformer'
    ]
    
    results = []
    
    for model_name in models_to_train:
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        model = get_model(model_name, num_classes)
        
        # Callbacks
        model_path = os.path.join(MODELS_DIR, f'{model_name}.h5')
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        start_time = time.time()
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop]
        )
        
        training_time = time.time() - start_time
        
        # Plot and save learning curves
        plot_history(history, model_name)
        
        # Evaluation on Test Set
        print(f"\nEvaluating {model_name} on Test Set...")
        
        # Reset test generator to ensure correct order
        test_gen.reset()
        
        # Time inference for a batch to estimate inference time per image
        inf_start = time.time()
        preds = model.predict(test_gen, verbose=1)
        inf_end = time.time()
        
        total_test_images = test_gen.samples
        inference_time_ms = ((inf_end - inf_start) / total_test_images) * 1000
        
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes
        
        # Metrics
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        val_loss = min(history.history['val_loss'])
        train_acc = max(history.history['accuracy']) * 100
        
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        actual_epochs = len(history.epoch)
        
        results.append({
            'Model': model_name,
            'Train Acc': round(train_acc, 2),
            'Test Acc': round(acc, 2),
            'Precision': round(prec, 2),
            'Recall': round(rec, 2),
            'F1 Score': round(f1, 2),
            'Val Loss': round(val_loss, 3),
            'Time (ms)': round(inference_time_ms, 2),
            'Size (MB)': round(model_size_mb, 2),
            'Epochs': actual_epochs
        })
        
        # Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, class_names, model_name)
        
        # Clean up memory
        del model
        tf.keras.backend.clear_session()
        
    # Generate Comparison Table
    df_results = pd.DataFrame(results)
    print("\n\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE AND EFFICIENCY EVALUATION OF MODELS ON PLANTVILLAGE DATASET")
    print("="*80)
    print(df_results.to_markdown(index=False))
    
    df_results.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/model_comparison_results.csv")

if __name__ == '__main__':
    train_and_evaluate()
