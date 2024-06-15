# Handwritten Digit Classification with TensorFlow

## Project Description
This project builds and trains a neural network to classify handwritten digits (0-9) from the MNIST dataset using TensorFlow's Keras API.

## Setup and Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/handwritten-digit-classification.git
    cd handwritten-digit-classification
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install tensorflow matplotlib
    ```

## Running the Project

1. **Run the script**:
    ```python
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt

    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Shuffle the training data
    indices = tf.range(train_images.shape[0])
    tf.random.shuffle(indices)
    train_images, train_labels = tf.gather(train_images, indices), tf.gather(train_labels, indices)

    # Build the model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=50, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy:', test_acc)

    # Make predictions
    predictions = model.predict(test_images)

    # Display a few test images and their predicted labels
    num_images_to_display = 5
    plt.figure(figsize=(10, 5))
    for i in range(num_images_to_display):
        plt.subplot(1, num_images_to_display, i+1)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.title(f"Pred: {tf.argmax(predictions[i])}, True: {test_labels[i]}")
        plt.axis('off')
    plt.show()
    ```
