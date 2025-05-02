import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Default constants, can be overridden by command-line arguments
IMG_WIDTH = 30
IMG_HEIGHT = 30
EPOCHS = 10
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
N_SPLITS = 5
OUTPUT_DIR = "output_plots"  # Directory to save plots

def parse_arguments():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs for training")
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH, help="Width of input images")
    parser.add_argument("--img_height", type=int, default=IMG_HEIGHT, help="Height of input images")
    parser.add_argument("--base_data_dir", type=str, default=r"E:\Github\AI Data", help="Base directory for image data")
    parser.add_argument("--train_csv", type=str, default="Train.csv", help="Path to the training CSV file")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Update global constants
    global IMG_WIDTH, IMG_HEIGHT, EPOCHS
    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    EPOCHS = args.epochs
    base_data_dir = args.base_data_dir
    train_csv_path = args.train_csv

    print(f"Using {EPOCHS} epochs, image size {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Base data directory: {base_data_dir}")
    print(f"Train CSV path: {train_csv_path}")

    # Load all available data for cross-validation
    print(f"Loading all data from {train_csv_path} for cross-validation...")
    all_images, all_labels = load_data(train_csv_path, base_data_dir)

    # Initialize K-Fold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Lists to store results from each fold
    fold_scores = []
    fold_histories = []
    all_true_labels = []
    all_predictions = []

    fold_no = 1
    for train_index, val_index in kf.split(all_images):
        print(f"--- Processing Fold {fold_no}/{N_SPLITS} ---")

        # Split data for this fold
        x_train, x_val = all_images[train_index], all_images[val_index]
        y_train, y_val = all_labels[train_index], all_labels[val_index]

        # Prepare labels (convert to categorical format)
        y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CATEGORIES)
        y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CATEGORIES)

        # Get a compiled neural network
        model = get_model()

        # Train model
        print(f"Training on {len(x_train)} samples, validating on {len(x_val)} samples...")
        history = model.fit(
            x_train,
            y_train_cat,
            epochs=EPOCHS,
            validation_data=(x_val, y_val_cat),
            verbose=1
        )
        fold_histories.append(history)

        # Evaluate model
        print("Evaluating model on validation data for this fold...")
        scores = model.evaluate(x_val, y_val_cat, verbose=0)
        fold_scores.append(scores)
        print(f"Fold {fold_no} - Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")

        # Generate predictions for the confusion matrix
        predictions_proba = model.predict(x_val)
        predictions_classes = np.argmax(predictions_proba, axis=1)

        # Store predictions and true labels
        all_true_labels.extend(y_val)
        all_predictions.extend(predictions_classes)

        fold_no += 1

    # --- Post-Cross-Validation Analysis ---
    avg_scores = np.mean(fold_scores, axis=0)
    print("\n--- Average Cross-Validation Scores ---")
    print(f"Loss: {avg_scores[0]:.4f}, Accuracy: {avg_scores[1]:.4f}")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Plot learning curves for each fold
    print(f"\nPlotting and saving learning curves to {OUTPUT_DIR}...")
    for i, history in enumerate(fold_histories):
        plot_learning_curves(history, i + 1, OUTPUT_DIR)

    # Plot confusion matrix
    print(f"\nPlotting and saving overall confusion matrix to {OUTPUT_DIR}...")
    plot_confusion_matrix(all_true_labels, all_predictions, NUM_CATEGORIES, OUTPUT_DIR)
    print("\nAll plots saved successfully.")

def load_data(csv_path, base_dir):
    images = []
    labels = []
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        sys.exit(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        sys.exit(f"Error reading CSV file {csv_path}: {e}")

    print(f"Loading {len(data)} images from {csv_path}...")
    for index, row in data.iterrows():
        label = int(row['ClassId'])
        relative_path = row['Path']
        img_path = os.path.join(base_dir, *relative_path.split('/'))

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    if not images:
        sys.exit(f"Error: No images were successfully loaded from {csv_path}")
    return np.array(images), np.array(labels)

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_learning_curves(history, fold_num, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'Fold {fold_num} Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'Fold {fold_num} Loss')
    axes[1].legend()
    save_path = os.path.join(output_dir, f"learning_curves_fold_{fold_num}.png")
    plt.savefig(save_path)
    plt.close(fig)

def plot_confusion_matrix(true_labels, predictions, num_classes, output_dir):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Overall Confusion Matrix')
    save_path = os.path.join(output_dir, "confusion_matrix_overall.png")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
