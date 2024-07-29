# Tongue Diagnosis AI
###WARNING, THIS MODEL NEEDS MORE DATA, WE HAVE REQUESTEDX SOME SAMPLES FOR AN ACCUPUNTURE INSTITUTION AND WILL HAVE MORE DATA SOON##
## Description
Tongue Diagnosis AI is an innovative web application that utilizes artificial intelligence to analyze tongue images for traditional Chinese medicine diagnosis. This tool aims to assist practitioners and individuals in quickly assessing tongue conditions, which are considered important indicators of overall health in Chinese medicine.

## Features
- Upload tongue images for instant analysis
- Capture tongue images directly through the device's camera
- Receive immediate diagnosis results with confidence levels
- User-friendly interface for easy navigation and use

## Technology Stack
- **Backend**: Python with Flask framework
- **Frontend**: HTML, CSS, JavaScript
- **AI Model**: TensorFlow and Keras
- **Image Processing**: OpenCV
- **Data Visualization**: Matplotlib and Seaborn
- **Version Control**: Git

## AI Model
The core of this application is a deep learning model built with TensorFlow and Keras. It uses transfer learning with a MobileNetV2 base, fine-tuned on a dataset of tongue images. The model is trained to classify various tongue conditions relevant to Chinese medicine diagnosis.

## Setup and Installation
1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Flask application: `python run.py`

## Usage
1. Access the web interface through a browser
2. Choose to either upload an image or capture one using your device's camera
3. Submit the image for analysis
4. View the diagnosis results, including the predicted condition and confidence level


## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments
- Thanks to my girlfiend who will be the best acupuncturist in the world


## Disclaimer
This tool is intended for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
