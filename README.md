<!-- TOC -->

- [tensorflow-on-android](#tensorflow-on-android)
- [Problem](#problem)
- [Literature Review](#literature-review)
- [Plan](#plan)

<!-- /TOC -->

# tensorflow-on-android
OSU CSE 5523 - Machine Learning Project

# Problem

Tensorflow on Android devices. Sophisticated learning tools on mobile devices are necessary for complex machine learning problems. One such problem is the use of ML techniques to learn concepts from mobile data (data collected from mobile devices). Since computational resources on mobile devices are limited, recent research has shown promising results using ML methods in a distributed manner. Moreover, recent work has also shown the ability to use the popular ML library Tensorflow to perform inference on mobile devices. We propose extending Tensorflow from its current state to support not only inference but also training on Android devices.

# Literature Review

See our [literature review](literature-review.md) file.

# Plan

Support training Tensorflow models on Android devices.

Goals:

- Proof of concepts
    - Implement an XOR neural network and show that we can switch it to a !XOR
    - Parity problem
- More complex applications (possibly in order of implementation)
    - RNN for audio generation
    - [Crowd-ML](https://github.com/jihunhamm/Crowd-ML) (MNIST binary and multi-class classification)

The project will begin with deploying Tensorflow on an Android device. Building out the training capabilities of Tensorflow to facilitate training and model optimization on the device. Next, we’ll implement basic proof of concept models followed by more complex applications.
