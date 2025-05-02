import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
N_SPLITS = 5
OUTPUT_DIR = "output_plots" # Directory to save plots


def main():
    # Define base directory and CSV path for training data
    base_data_dir = r"E:\Github\AI Data"
    train_csv_path = os.path.join("Train.csv")

    # Check command-line arguments
    if len(sys.argv) != 1:
        sys.exit("Usage: python traffic.py")

    # Load ALL available data (from Train.csv) for cross-validation
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

        # Get a compiled neural network (fresh model for each fold)
        model = get_model()

        # Fit model on training data for this fold, validating on the validation set
        print(f"Training on {len(x_train)} samples, validating on {len(x_val)} samples...")
        history = model.fit(
            x_train,
            y_train_cat,
            epochs=EPOCHS,
            validation_data=(x_val, y_val_cat),
            verbose=1 # Show progress per epoch
        )
        fold_histories.append(history)

        # Evaluate neural network performance on the validation set for this fold
        print("Evaluating model on validation data for this fold...")
        scores = model.evaluate(x_val, y_val_cat, verbose=0)
        fold_scores.append(scores)
        print(f"Score for fold {fold_no}: Loss={scores[0]:.4f}, Accuracy={scores[1]:.4f}, Precision={scores[2]:.4f}, Recall={scores[3]:.4f}")

        # Generate predictions for the confusion matrix
        predictions_proba = model.predict(x_val)
        predictions_classes = np.argmax(predictions_proba, axis=1)

        # Store predictions and true labels for the overall confusion matrix
        all_true_labels.extend(y_val) # Use original integer labels
        all_predictions.extend(predictions_classes)

        fold_no += 1

    # --- Post-Cross-Validation Analysis ---

    # Calculate and print average scores across folds
    avg_scores = np.mean(fold_scores, axis=0)
    print("\n--- Average Cross-Validation Scores ---")
    # Assuming the order in model.compile metrics: loss, accuracy, precision, recall
    metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall'] # Manually define based on get_model()
    for name, value in zip(metric_names, avg_scores):
         print(f"Average {name}: {value:.4f}")
    print("-------------------------------------")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Plot learning curves for each fold and save them
    print(f"\nPlotting and saving learning curves to {OUTPUT_DIR}...")
    for i, history in enumerate(fold_histories):
        plot_learning_curves(history, i + 1, OUTPUT_DIR)

    # Calculate, plot, and save the overall confusion matrix
    print(f"\nCalculating, plotting, and saving overall confusion matrix to {OUTPUT_DIR}...")
    plot_confusion_matrix(all_true_labels, all_predictions, NUM_CATEGORIES, OUTPUT_DIR)

    # plt.show() # No longer needed, plots are saved
    print("\nAll plots saved successfully.")


def load_data(csv_path, base_dir):
    """
    Load image data based on a CSV file.

    Args:
        csv_path (str): Path to the CSV file (e.g., Train.csv or Test.csv).
        base_dir (str): The base directory containing the image folders (e.g., 'E:\Github\AI Data').

    The CSV file is assumed to have columns 'ClassId' and 'Path'.
    'Path' contains the relative path to the image from `base_dir`.

    Returns tuple `(images, labels)`. `images` is a list of numpy ndarrays
    (IMG_WIDTH x IMG_HEIGHT x 3), and `labels` is a list of integer labels.
    """
    images = []
    labels = []
    try:
        # Attempt to read the CSV file
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(f"Exiting: Could not load data from {csv_path}")
    except Exception as e:
        # Handle any other errors during CSV reading
        print(f"Error reading CSV file {csv_path}: {e}")
        sys.exit(f"Exiting: Could not load data from {csv_path}")

    print(f"Loading {len(data)} images from {csv_path}...")

    # Loop through each row in the CSV file to load images and labels
    for index, row in data.iterrows():
        label = int(row['ClassId'])
        relative_path = row['Path']
        
        # Construct the full path to the image
        img_path = os.path.join(base_dir, *relative_path.split('/'))

        try:
            # Attempt to read the image from disk
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path} (referenced in {csv_path})")
                continue  # Skip the image if it cannot be read
            # Resize the image to the specified dimensions
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)  # Add the processed image to the list
            labels.append(label)  # Add the corresponding label to the list
        except Exception as e:
            # Handle any other errors during image processing
            print(f"Error processing image {img_path} (referenced in {csv_path}): {e}")

    # Check if no images were successfully loaded
    if not images:
        print(f"Error: No images were successfully loaded from {csv_path}. Check paths and file integrity.")
        sys.exit(f"Exiting: Failed to load images from {csv_path}")

    # Return the loaded images and labels as numpy arrays
    return (np.array(images), np.array(labels))


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)` and the output
    layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

        # Second max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model


# Helper function to plot learning curves and save to file
def plot_learning_curves(history, fold_num, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'Fold {fold_num}: Accuracy vs. Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'Fold {fold_num}: Loss vs. Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    # Save the figure
    save_path = os.path.join(output_dir, f"learning_curves_fold_{fold_num}.png")
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free memory

# Helper function to plot confusion matrix and save to file
def plot_confusion_matrix(true_labels, predictions, num_classes, output_dir):
    cm = confusion_matrix(true_labels, predictions)
    # Define class labels if available, otherwise use range
    class_labels = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 10)) # Adjust size as needed
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.title('Overall Confusion Matrix (Across Folds)')
    plt.tight_layout()
    # Save the figure
    save_path = os.path.join(output_dir, "confusion_matrix_overall.png")
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    main()
