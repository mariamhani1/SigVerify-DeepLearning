# Signature Verification Using Siamese Neural Networks

This project implements a signature verification system using Siamese Neural Networks. The system is designed to distinguish between genuine and forged signatures by learning a similarity metric. It uses TensorFlow and Keras for deep learning and processes datasets of handwritten signatures.

## Features

- **Preprocessing**: Includes deskewing, noise reduction, contrast enhancement, binarization, morphological processing, skeletonization, and edge detection.
- **Siamese Neural Network**: Utilizes a VGG16-based feature extraction model and a custom contrastive loss function.
- **Dataset Support**: Supports multiple datasets, including:
  - CEDAR
  - BHSig260-Bengali
  - BHSig260-Hindi
- **Training and Evaluation**: Includes training with TPU support, validation, and testing with metrics like accuracy and loss.

## Project Structure

```
Signature Verification/
├── SigVerify.ipynb       # Jupyter Notebook containing the implementation
├── s41598-024-79167-8.pdf # Reference document
```

## Requirements

- Python 3.10 or later
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Scikit-image
- KaggleHub

Install the required libraries using:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-image kagglehub
```

## Usage

1. **Download the Dataset**: The project uses datasets from Kaggle. Modify the dataset paths in the notebook as needed.
2. **Run the Notebook**: Open `SigVerify.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially.
3. **Evaluate the Model**: The notebook includes evaluation steps for CEDAR, Bengali, and Hindi datasets.

## Key Functions

- `preprocess_image(image_path)`: Preprocesses an image into a multi-channel input for the CNN.
- `load_dataset(dataset_path)`: Loads and prepares the dataset for training, validation, and testing.
- `contrastive_loss(y_true, y_pred)`: Custom loss function for training the Siamese network.
- `create_tf_dataset(pairs, labels)`: Creates TensorFlow datasets for training and evaluation.

## Results

The model achieves high accuracy on the test datasets:
- **CEDAR**: Test Accuracy: ~92%
- **Bengali**: Test Accuracy: ~92%
- **Hindi**: Test Accuracy: ~91%

## References

- [Scientific Reports Article](s41598-024-79167-8.pdf): Reference document for the project.
- [Kaggle Dataset](https://www.kaggle.com/ishanikathuria/handwritten-signature-datasets): Source of the datasets.

## License

This project is for educational purposes and follows the license of the referenced datasets.
