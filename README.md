# Brain Tumor Classifier

The Brain Tumor Classifier is an application that utilizes deep learning techniques to classify brain tumor images into different categories. It is designed to assist in the detection and classification of brain tumors for medical diagnosis and treatment planning.

## Dataset

The dataset used for training and evaluating the Brain Tumor Classifier can be obtained from [Brain tumour dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It provides a diverse collection of brain tumor images for different tumor types, including meningioma, glioma, pituitary tumor, and glioblastoma.

## Methodology

The Brain Tumor Classifier employs a deep learning approach using convolutional neural networks (CNNs). The model architecture consists of multiple convolutional and pooling layers, followed by fully connected layers. It leverages the TensorFlow framework and the Keras API to build and train the model.

## Installation

To set up the Brain Tumor Classifier, follow these steps:

1. Clone the repository: `git clone https://github.com/vivekmenonm/braintumour-classifier.git`

2. Navigate to the project directory

3. Create a virtual environment using virtualenv (assuming you have virtualenv installed):
4. Activate the virtual environment:
- For Windows:
  ```
  venv\Scripts\activate
  ```
- For Linux/Mac:
  ```
  source venv/bin/activate
  ```

5. Install the required dependencies:
`pip install -r requirements.txt`


## Usage

1. Make sure you have activated the virtual environment (see Installation steps).

2. Run the Streamlit web interface:

3. It will automatically take you to `http://localhost:8501` in browser else manually paste the link to access the Brain Tumor Classifier interface.

4. Upload an MRI brain scan image through the interface.

5. The model will analyze the image and provide the predicted tumor class and confidence score.

## Future Enhancements

- Incorporating advanced deep learning techniques for improved performance.
- Expanding the dataset to include more diverse brain tumor images.
- Enhancing the user interface with additional features for better visualization.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

The Brain Tumor Classifier is released under the [MIT License](LICENSE).

## Acknowledgments

We would like to acknowledge the creators of the brain tumor dataset used in this project for their valuable contribution.

## References

- [Kaggle - Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

