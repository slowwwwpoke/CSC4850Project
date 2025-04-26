import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard

dataset_dirs = ['datasets/combined-resized-denoised-normalized']

def train_and_save_model(dataset_dir, model_save_path):
    # Load preprocessed dataset
    data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        label_mode='int',
        image_size=(160, 160),
        batch_size=32
    )

    # Get class names and number of classes
    class_names = data.class_names
    num_classes = len(class_names)
    print(f" Class Names: {class_names}")
    print(f" Number of Classes: {num_classes}")

    # Normalize data to [0, 1]
    data = data.map(lambda x, y: (x / 255.0, y))

    # Split dataset into train, validation, and test sets
    data_size = len(data)
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    test_size = int(data_size * 0.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    # Define CNN Model
    model = Sequential([
        Conv2D(16, (3, 3), strides=1, activation='relu', input_shape=(160, 160, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), strides=1, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), strides=1, activation='relu'), 
        MaxPooling2D(),
        Conv2D(32, (3, 3), strides=1, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()

    # Define TensorBoard callback
    logdir = f'logs/{os.path.basename(dataset_dir)}'
    tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch=0)

    # Train the model
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # Save model
    model.save(model_save_path)
    print(f" Model saved to {model_save_path}")

    # Save class names for later use
    class_names_path = model_save_path.replace('.h5', '_classes.npy')
    np.save(class_names_path, class_names)
    print(f" Class names saved to {class_names_path}")

    # --- Confusion Matrix ---
    print(" Generating confusion matrix...")

    # Collect test data in numpy arrays
    y_true = []
    y_pred = []

    for batch_images, batch_labels in test:
        preds = model.predict(batch_images)
        pred_labels = np.argmax(preds, axis=1)

        y_true.extend(batch_labels.numpy())
        y_pred.extend(pred_labels)

    cm = confusion_matrix(y_true, y_pred)
    print(" Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model, class_names

for dataset_dir in dataset_dirs:
    model_save_path = f"model_{os.path.basename(dataset_dir)}.h5"

    # Train and save model for the current dataset
    model, class_names = train_and_save_model(dataset_dir, model_save_path)
