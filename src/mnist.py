from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pygame
import numpy as np
import cv2
from scipy import ndimage

model_path = "mnist/mnist_model.h5"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# You can also check during execution
# For example, before model.fit() or model.predict()
# tf.debugging.set_log_device_placement(True) # This will print device placement for all ops


def get_model():
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return load_model(model_path)

    print("Training a new model...")
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)

    # CNN model
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    # Build model
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        datagen.flow(x_train, y_train_categorical, batch_size=32),
        epochs=20,
        validation_data=(x_test, y_test_categorical),
    )

    # Save model
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def center_digits(image):
    # Ensure image is uint8 for certain OpenCV operations if not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Crop to bounding box of non-zero pixels (more robust)
    coords = np.argwhere(image > 0)
    if coords.size == 0:
        # If image is blank (e.g., after too much cropping or empty input)
        return np.zeros((28, 28), dtype=np.uint8)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    image = image[y_min : y_max + 1, x_min : x_max + 1]

    rows, cols = image.shape
    if rows == 0 or cols == 0:
        return np.zeros((28, 28), dtype=np.uint8) # Should be caught by coords.size == 0

    # Pad to make it square
    if rows > cols:
        pad = (rows - cols) // 2
        image = np.pad(image, ((0, 0), (pad, pad)), mode="constant")
    elif cols > rows:
        pad = (cols - rows) // 2
        image = np.pad(image, ((pad, pad), (0, 0)), mode="constant")

    # Ensure image is not empty after padding (edge case)
    if image.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    # Shift center of mass to center of image
    # Calculate center of mass. If image is all zeros, com is (-1,-1)
    cy, cx = ndimage.center_of_mass(image)

    # Target center for a 28x28 image (0-indexed, so 13.5, 13.5 is true center)
    # Using 14 as an integer approximation.
    target_cx, target_cy = 13.5, 13.5

    shiftx = np.round(target_cx - cx).astype(int)
    shifty = np.round(target_cy - cy).astype(int)

    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    # WarpAffine resizes and shifts. Output is 28x28.
    image = cv2.warpAffine(image, M, (28, 28), flags=cv2.INTER_LINEAR)
    return image


def preprocess(pygame_surface):
    raw_data = pygame.surfarray.array3d(pygame_surface)
    gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)

    # Skip nearly empty canvas
    if np.count_nonzero(gray) < 50: # Threshold for minimum ink
        return None

    # Preprocessing steps
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold to make digit white (255) and background black (0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Get contours sorted by area
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    margin = 20  # Add some margin around the digit
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    # Ensure width and height don't exceed image boundaries
    w = min(w + 2 * margin, thresh.shape[1] - x)
    h = min(h + 2 * margin, thresh.shape[0] - y)
    roi = thresh[y : y + h, x : x + w]

    if roi.size == 0: # Check if ROI is empty
        return None

    # Center and resize (center_digits returns a 28x28 image)
    roi_centered = center_digits(roi)

    # Normalize to 0-1 range (as expected by the model)
    # Output of center_digits is uint8, possibly with interpolated values
    roi_normalized = roi_centered.astype("float32") / 255.0
    # Reshape for the model: (1, 28, 28, 1)
    return np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)


# Pygame setup
pygame.init()
width, height = 400, 400
white, black = (255, 255, 255), (0, 0, 0)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a digit (0-9) - Press 'C' to clear")
screen.fill(black)
pygame.display.update() # Initial update to show black screen
clock = pygame.time.Clock()
try:
    font = pygame.font.SysFont("Arial", 36)
except pygame.error:
    font = pygame.font.Font(None, 48) # Fallback font

model = get_model()
drawing = False
prediction = None
last_pos = None
# Removed unused 'start_post' variable

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left mouse button
                drawing = True
                # last_pos = event.pos # Start line from click point immediately

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left mouse button
                drawing = False
                last_pos = None # Reset last_pos to avoid disconnected lines on next draw
                input_image = preprocess(screen)
                if input_image is not None:
                    pred_probabilities = model.predict(input_image)
                    prediction = np.argmax(pred_probabilities)
                    # print(f"Prediction: {prediction}, Confidence: {pred_probabilities[0][prediction]:.2f}")
                else:
                    # If input is too sparse, clear previous prediction or keep it?
                    # For now, let's clear it if drawing was attempted but resulted in None
                    # prediction = None # Or keep old one, current behavior keeps old one
                    pass


        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill(black)
                prediction = None # Clear prediction text

    # Drawing logic
    if drawing:
        current_pos = pygame.mouse.get_pos()
        if last_pos is not None:
            pygame.draw.line(screen, white, last_pos, current_pos, 8) # Thickness 8
        # Draw a circle at the current point to make it smoother and cover single clicks
        pygame.draw.circle(screen, white, current_pos, 4) # Radius 4
        last_pos = current_pos
    # else: # This was causing last_pos to be None too early if MOUSEBUTTONUP didn't reset it
    #    last_pos = None

    # Display prediction
    # Clear the top area before blitting new text to avoid overlap
    screen.fill(black, (10, 0, width - 20, 50)) # Clear previous text area
    if prediction is not None:
        label = font.render(f"Prediction: {prediction}", True, white)
        screen.blit(label, (10, 10))

    pygame.display.flip() # Use flip for full surface update, or update specific rects
    clock.tick(60) # Limit to 60 FPS

pygame.quit()
