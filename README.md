# Socially Relevant Project

# Facial Expression-Based Lie Detection System


![image](https://github.com/MachinaDominus/srp/assets/141066776/a1c3676b-0b2d-4a3a-9f83-1f0786ad2f97)

credits : https://www.analyticsinsight.net/ai-in-forensic-investigation-and-crime-detection/

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The **Facial Expression-Based Lie Detection System** is a socially impactful project that aims to assist interrogators and investigators in determining the likelihood of a person lying or being truthful based on their facial expressions. The system leverages advanced computer vision techniques, including Convolutional Neural Networks (CNNs), to analyze facial expressions and provide insights to the interrogators.

This project is built with the vision of enhancing the efficiency and accuracy of lie detection processes while promoting ethical considerations and the responsible use of technology in sensitive situations.

## Features

Behavior Detection using Pre-trained Models: The project utilizes pre-trained deep learning models to detect a range of behaviors, including:

- Raised eyebrows
- Trembling hands
- Avoiding eye contact
- Eyes squinting
- Lip twitching
- Crossing arms
  
**Real-time Detection**:The system provides real-time behavior detection through the webcam stream. It analyzes each frame and assesses the likelihood of each behavior being exhibited.

**User-friendly Interface**: The project offers a user-friendly interface for displaying detected behaviors directly on the webcam feed, making it intuitive for users to observe and assess the situation.

**Customizable Thresholds**: The system allows you to customize detection thresholds for each behavior, enabling you to adjust sensitivity and false-positive rates according to specific scenarios.

**State-of-the-art Deep Learning**: The behavior detection is powered by state-of-the-art deep learning models. These models have been pre-trained on a diverse dataset to recognize subtle facial cues associated with the targeted behaviors.

**Efficient Pre-processing**: The project employs efficient image preprocessing techniques to prepare frames for input into the deep learning models. This ensures accurate and reliable behavior predictions.

**Ethical Considerations**: The project acknowledges the importance of ethical considerations when using technology for behavior analysis. It aims to contribute to responsible and ethical use of AI in sensitive scenarios.



## Demo

![Demo GIF](demo.gif)

[Watch the full demo video](demo_video_link)

## Installation

1. Clone this repository to your local machine using:
   ```sh
   git clone https://github.com/MachinaDominus/srp.git



## Usage

### Installation

1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/MachinaDominus/srp.git

### Steps 
1. Navigate to the project directory
    ```sh
    cd srp
2. install the requirements using pip
    ```sh
    pip install -r requirements.txt

3. Running the System, Execute the main script to start the lie detection system:
    ```sh
    python main.py
Grant necessary permissions for webcam access.

**Real-time Detection**

1. Once the system is running, it will analyze the webcam feed in real-time.
2. Detected behaviors will be highlighted directly on the webcam feed.

**Customization (Optional)**

1. If desired, customize detection thresholds for each behavior:
2. Modify the settings in the configuration file.

**Observation and Assessment**

1. Observe the detected behaviors on the webcam feed.
2. Assess the likelihood of the person lying or being truthful based on displayed cues.

**Ethical Use**

1. Always consider ethical considerations when using the system for behavior analysis.
2. Use technology responsibly, respecting privacy and consent.



## Dataset

The training and evaluation of the Facial Expression-Based Lie Detection System were conducted using a carefully curated dataset that includes a diverse range of facial expressions and behaviors. The dataset was collected with the following considerations:

- **Size**: The dataset contains a substantial number of samples to ensure model generalization and robustness.

- **Variety**: It encompasses a wide variety of facial expressions, encompassing different ages, genders, ethnicities, and emotional states.

- **Ethical Considerations**: The dataset was collected in accordance with ethical guidelines, ensuring the privacy and consent of individuals participating in data collection.

- **Annotations**: Each sample in the dataset is annotated with labels corresponding to the specific behaviors or expressions exhibited.

- **Validation Split**: The dataset was divided into training, validation, and testing subsets to assess the model's performance effectively.

- **Data Augmentation**: To enhance model performance, data augmentation techniques were applied to the dataset, increasing its diversity and variability.

Please note that due to privacy concerns and ethical considerations, the dataset used in this project may not be publicly available. However, there are publicly available facial expression datasets that you can explore for similar purposes, such as the CK+ dataset, AffectNet, or the MMI facial expression database.

For information about how to access and use specific datasets, refer to their respective documentation and terms of use.



## Model

The Facial Expression-Based Lie Detection System utilizes a state-of-the-art deep learning model for accurate behavior and expression recognition. The model architecture and training process were designed to achieve high performance in real-time detection scenarios.

### Model Architecture

The deep learning model employs a Convolutional Neural Network (CNN) architecture that has been fine-tuned to identify subtle facial cues associated with different behaviors and expressions. The architecture consists of multiple convolutional layers, pooling layers, and fully connected layers to effectively capture spatial features in the input facial images.

### Training

The model was trained on a carefully curated dataset, encompassing diverse facial expressions and behaviors. The training process involved the following steps:

- Data Preprocessing: Images were resized, normalized, and augmented to ensure model robustness.

- Model Initialization: A pre-trained CNN architecture was used as a base, leveraging transfer learning for improved performance.

- Fine-tuning: The model was fine-tuned on the specific behavior and expression detection task using appropriate loss functions and optimization techniques.

- Validation and Testing: Validation and testing subsets from the dataset were used to monitor and evaluate the model's performance at different stages of training.

### Performance

The trained model demonstrated impressive accuracy and speed in real-time behavior detection. It successfully identifies and highlights behaviors such as raised eyebrows, trembling hands, avoiding eye contact, squinting eyes, twitching lips, and crossing arms.

Please note that the exact architecture and implementation details may vary depending on the specific version of the project and the underlying deep learning framework used.

For more technical details about the model architecture, training process, and hyperparameters, refer to the project's code and documentation.




## Contributing

Thank you for considering contributing to the Facial Expression-Based Lie Detection System! Contributions from the open-source community help improve and enhance the project for everyone.

### How to Contribute

1. Fork the Repository: Start by forking this repository to your GitHub account.

2. Clone the Fork: Clone your forked repository to your local development environment.

   ```sh
   git clone https://github.com/YourUsername/srp.git
   
1. Create a Branch: Create a new branch for your work. 
    ```sh
    git checkout -b feature/new-feature

2. Make Changes: Make your desired changes and improvements to the project.

3. Test: Ensure that your changes work as intended and do not introduce any unexpected issues.

4. Commit and Push: Commit your changes and push them to your forked repository.
    ```sh
    git add .
    git commit -m "Add your descriptive commit message"
    git push origin feature/new-feature
    
5. Create a Pull Request: Open a pull request from your forked repository to this main repository. Provide a clear and descriptive title and explanation of your changes.


**Code Style and Guidelines**

1. Follow the existing code style and formatting used in the project.
2. Keep your code clean, well-documented, and modular.
3. If your contribution involves new features or significant changes, consider updating the documentation accordingly.

**Feedback and Issues**

If you encounter any issues, have questions, or want to provide feedback, please open an issue in the repository. We value your input and will work together to address any concerns.

**Attribution**

Contributors will be acknowledged and credited in the project's documentation. Your contributions help make this project more valuable and impactful for the community.

Thank you for your interest in contributing to the Facial Expression-Based Lie Detection System!


MIT License

Copyright (c) 2023 Mohammed Sohail

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


