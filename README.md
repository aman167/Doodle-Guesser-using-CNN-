# Doodle-Guesser-using-CNN-

"🎨🤖 Doodle-Guesser Game: Draw doodles &amp; let AI guess! Powered by a CNN trained on Google's QuickDraw dataset (1 million+ images and 345 classes), it predicts in real-time and shows the top 5 probabilities with a pie chart 🥧. Built using TensorFlow, Keras, &amp; Streamlit. A fun blend of AI &amp; gaming! 🚀"

Welcome to the Doodle-Guesser Game! This interactive web application allows you to draw a doodle on a canvas, and the AI predicts what you're drawing in real-time. Built using TensorFlow, Keras, and Streamlit, this project merges fun with AI and Machine Learning to bring you an exciting gaming experience. 😄

---

## 🚀 Features

- **Real-Time Predictions**: As you draw, the CNN model predicts your doodle in real-time!
- **Top 5 Predictions**: View the top 5 possible guesses along with their probabilities.
- **Interactive UI**: A smooth and engaging user interface created using Streamlit.
- **Visualization**: A pie chart displays the probabilities of the top 5 classes for easy comparison.

---

## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for creating the interactive UI.
- **Backend**: [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/) for the CNN model and training.
- **Data Processing**: NumPy, Pandas, and OpenCV for image processing and data manipulation.

---

## 📊 Dataset: Google’s QuickDraw

The model is trained using Google's [QuickDraw](https://quickdraw.withgoogle.com/data) dataset, which contains millions of doodles across 345 different categories. This dataset helps the CNN model to learn a variety of drawing styles and identify objects in real-time.

- **Number of classes**: 345
- **Number of images**: 50 million+
- **Image format**: PNG (greyscale images)
- **Categories**: Various categories such as animals, objects, vehicles, and more.

---

## 🧠 CNN Model

The core of this project is a Convolutional Neural Network (CNN) that is trained to recognize doodles. The model uses multiple layers of convolutional and pooling layers to extract features and classify images into one of 345 classes. The model is trained on the QuickDraw dataset and is optimized for fast, real-time predictions.

**Model Architecture**:

- **Input Layer**: 28x28 grayscale images
- **Convolutional Layers**: Multiple layers with activation functions to extract features.
- **Pooling Layers**: Max-pooling layers to reduce the spatial dimensions.
- **Fully Connected Layer**: Final dense layer with 345 neurons corresponding to the number of classes.
- **Softmax Activation**: To output probabilities for each class.
  ![Screenshot1](Doodle-Guesser-using-CNN-/screenshots/CNN.png)

---

## 📊 Model Performance

The CNN model trained on the QuickDraw dataset achieved excellent results:

- **Training Accuracy**: 94.02% 🏆
- **Validation Accuracy**: 94.06% 🎯
- **Test Accuracy**: 94.06% ✅
  ![Screenshot1](Doodle-Guesser-using-CNN-/screenshots/accuracy.png)

---

These results show the model's ability to generalize well, maintaining high accuracy across different data splits. This demonstrates the effectiveness of the architecture and the quality of the dataset used.

## 🌐 Streamlit Interface

The game is deployed using [Streamlit](https://streamlit.io/), an open-source app framework that allows rapid prototyping of interactive applications. The real-time prediction is displayed as you draw on the canvas. The UI updates live as the CNN model predicts your doodle. The top 5 predictions are shown along with their probabilities in the form of a pie chart.

---

## 📹 Demo Video

[Doodle-Guesser-using-CNN-/screenshots/video_demo.mp4]

---

## 🖼 Screenshots

Here are a few screenshots of the Doodle-Guesser Game:

![Screenshot1]
(Doodle-Guesser-using-CNN-/screenshots/airplane_demo.png)
![Screenshot2](Doodle-Guesser-using-CNN-/screenshots/car_demo.png)

---

## 🛠 Installation

To run the Doodle-Guesser Game locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/doodle-guesser-game.git

   ```
2. Run the CNN.ipynb:
   This will create -> train -> download the model in your local system.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Happy doodling! 🎉 anytime, anywhere, online or offline.

## 🧑‍💻 Development Breakdown

### Part 1: Setup

- **Main Program and Main State**: Define the main program structure, including the state of the game, user interactions, and transitions between different game phases.
- **Substates of Main State**: Set up the game states such as initialization, waiting for input, making predictions, and displaying results.
- **Game Loop**: Build the game loop where the primary logic of the game (drawing, predictions, etc.) is handled.
- **User Interface Class**: Design a class for managing the user interface that allows interaction with the game, including drawing on the canvas and displaying predictions.

### Part 2: Getting Data

- **CNN Class**: Create a CNN class to handle the loading and training of the model.
- **Loading Datasets**: Import three small datasets (car, fish, and snowman) to test and train the CNN model.
- **Splitting and Shuffling Data**: Split the datasets into training and test data. Shuffle the data to ensure randomness during training.

### Part 3: Building the Model

- **CNN Architecture**: Construct a sequential CNN model with multiple convolutional and pooling layers.
- **Model Layers**: Add different layers (convolutional, pooling, dense, etc.) to the CNN model.
- **Compile Model**: Compile the model with an appropriate optimizer, loss function, and evaluation metrics.

### Part 4: Training the Model

- **Fetching Batches**: Create a function to fetch batches of data for training and testing.
- **Training and Evaluation**: Train the CNN model using the training dataset, and evaluate its performance on the test dataset.
- **Plotting Graphs**: Plot graphs of model loss and accuracy during the training process to visualize the model’s performance.

### Part 5: Predicting Samples

- **Fetching Sample Data**: Fetch batches of sample data to be passed through the trained model.
- **Prediction Function**: Implement a function to predict the class of the fetched sample data using the trained CNN model.

### Part 6: Drawing Doodles

- **Painter Class**: Create a Painter class that lets users draw doodles with the mouse.
- **Drawing Area and Pencil**: Define the drawing area and the pencil object to allow users to draw on the canvas.
- **Smoothing Lines**: Implement a function to smooth out the lines drawn by users, using quadratic curves to make the drawing appear cleaner.

### Part 7: Recognizing Doodles

- **Resizing Drawings**: Resize the doodle drawing to the required 28x28 pixel size to match the input size expected by the CNN model.
- **Normalizing Data**: Normalize the pixel values to ensure the data is in a suitable format for the CNN model.
- **Making Predictions**: Pass the normalized data through the CNN model to predict the doodle class.

### Part 8: Adding More Doodle Categories

- **Expanding the Dataset**: Add additional doodle categories to the game, increasing the diversity of predictions.
- **Model Update**: Retrain the CNN model with the new categories to improve its prediction capabilities.

## 💬 Contributions

Feel free to contribute to this project by submitting issues or pull requests. Whether you have a bug fix, feature suggestion, or improvement idea, contributions are always welcome! 😊

To get started:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Open a pull request with a detailed description of your changes.

Happy coding! 🚀
