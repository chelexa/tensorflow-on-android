# Tensorflow Training on Android

This guide was written for MacOS environments. Windows users beware! Linux users are probably fine.

## Step 1. Download Tensorflow from source
In order for Android to have the capabilities of training TF graphs, you need to
be able to refer to the Java libraries written by the TF team. Therefore, you must
clone the Tensorflow repo:

```bash
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
```
