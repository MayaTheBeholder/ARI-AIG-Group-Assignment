import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Define a function to load and preprocess the dataset.
def load_data(data_dir, img_size=(30, 30)):
    """
    Reads images from data_dir, which contains subdirectories (0 to 42) representing sign classes.
    Each image is read using OpenCV, resized to img_size, and appended to the list.
    Returns: tuple (images, labels)
    """
    images = []
    labels = []
    
    # Iterate over all folders (each corresponding to a traffic sign category).
    # We assume folder names are numeric strings.
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # label conversion: folder name is the label
        label = int(folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # Read image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                continue
            # Resize image to a fixed size (30x30) for consistency
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Helper function to plot a confusion matrix.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    # Check command-line arguments.
    if len(sys.argv) < 2:
        sys.exit("Usage: python traMic.py <data_directory> [model_filename]")
    
    data_dir = sys.argv[1]
    model_filename = sys.argv[2] if len(sys.argv) >= 3 else None

    print("Loading data from directory:", data_dir)
    X, y = load_data(data_dir)
    print(f"Total images loaded: {len(X)}")
    
    # Normalize images - converting pixel values from 0-255 to 0-1.
    X = X.astype('float32') / 255.0

    # Convert labels to one-hot encoding.
    num_classes = 43
    y = to_categorical(y, num_classes)
    
    # Split dataset into training and testing sets (80/20 split here).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Dataset split:")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    
    # Define a simple Convolutional Neural Network.
    model = Sequential()
    
    # First convolutional block.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional block.
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adding dropout for regularization.
    model.add(Dropout(0.25))
    
    # Flatten and add dense layers.
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    # Train the model; adjust epochs and batch_size per your experimentation.
    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=2)
    
    # Evaluate model performance on test set.
    print("Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    
    # Create predictions and compute the confusion matrix.
    y_pred = model.predict(X_test)
    # Convert one-hot encoding back to labels.
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    # Plot the confusion matrix.
    plot_confusion_matrix(cm, classes=[str(i) for i in range(num_classes)], normalize=True,
                          title='Normalized Confusion Matrix')
    
    # Optionally, save the model if a filename is provided.
    if model_filename:
        model.save(model_filename)
        print("Model saved to", model_filename)
    else:
        # Save by default to model.h5 if desired.
        model.save('model.h5')
        print("Model saved to model.h5")

if __name__ == "__main__":
    main()
