import tensorflow as tf
import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")


TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"
TEST_DIR = "dataset_split/test"

def augmentation():
    # Add data augmentation for training to improve generalization
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
    
    # Only rescaling for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        class_mode="sparse"
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        class_mode="sparse"
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        class_mode="sparse"
    )
    
    return train_generator, val_generator, test_generator

def model_without_finetuning():
    with mlflow.start_run(run_name="No_Finetuning"):
        # Load pre-trained MobileNetV2 as feature extractor
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze the base model
        
        # Build model architecture with better regularization
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(5, activation='softmax')  
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get data generators
        train_gen, val_gen, test_gen = augmentation()
        
        # Log model parameters
        mlflow.log_param("base_model", "MobileNetV2_frozen")
        mlflow.log_param("trainable_layers", "None (frozen)")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 32)
        
        # Training with MLflow logging and improved callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        history = model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(test_gen)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log the model
        mlflow.tensorflow.log_model(model, "model")
        
        print(f"Model without fine-tuning - Test Accuracy: {test_accuracy:.4f}")
    
    return model

def model_with_finetuning():
    with mlflow.start_run(run_name="Finetuning"):
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        
        # Build model architecture with better regularization
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(5, activation='softmax') 
        ])
        
        # Unfreeze base model for fine-tuning
        base_model.trainable = True
        
        # Fine-tune from the top layers of the base model
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get data generators
        train_gen, val_gen, test_gen = augmentation()
        
        # Log model parameters
        mlflow.log_param("base_model", "MobileNetV2")
        mlflow.log_param("trainable_layers", "Fine-tuning")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("fine_tune_at", fine_tune_at)
        
        # Training with fine-tuning
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        print("Fine-tuning...")
        history = model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(test_gen)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log the model
        mlflow.tensorflow.log_model(model, "model")
        
        print(f"Model with fine-tuning - Test Accuracy: {test_accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    # Set MLflow tracking URI if needed
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print("Training model without fine-tuning...")
    model1 = model_without_finetuning()
    
    print("\nTraining model with fine-tuning...")
    model2 = model_with_finetuning()