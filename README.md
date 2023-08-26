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




