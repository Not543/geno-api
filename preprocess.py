import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_fingerprint(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to highlight fingerprint patterns
    preprocessed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Save the preprocessed image
    cv2.imwrite(output_path, preprocessed)

    # (Optional) If you want to save the plot image instead of showing it, you can do:
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    #
    # plt.subplot(1, 2, 2)
    # plt.title("Preprocessed Image")
    # plt.imshow(preprocessed, cmap="gray")
    # plt.axis("off")
    #
    # plt.savefig(output_path.replace('.png', '_plot.png'))  # Save plot image if you want

    # Do NOT call plt.show() in a server environment!
