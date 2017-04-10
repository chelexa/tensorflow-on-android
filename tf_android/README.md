# Tensorflow Training On Android

This guide was written for MacOS environments. Windows users beware! Linux users are probably fine.
When in doubt, refer to the official [Android Tensorflow README](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).
This guide assumes you already have [Android Studio](https://developer.android.com/studio/index.html) installed.

## Step 1. Download Tensorflow From Source
In order for Android to have the capabilities of training TF graphs, you need to
be able to refer to the Java libraries written by the TF team. Therefore, you must
clone the Tensorflow repo.

```bash
# Replace 'workspace' with your preferred working directory
cd ~/workspace/
# Note that --recurse-submodules is necessary to prevent some issues with protobuf compilation. - TF Team
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
```
If you have previously cloned from the TF repo and are thinking about skipping this step, DON'T! The TF repo is updated constantly,
and you will most likely get errors by not pulling the latest version.

## Step 2. Install Bazel
[Bazel](bazel.build) is Google's own build tool and is the primary build system for Tensorflow. Get it [here](https://bazel.build/versions/master/docs/install.html).

If you have previously installed Bazel and are thinking about skipping this step, DON'T! Not updating to this latest version of Bazel
is a potential source of problems. If on MacOS use:

```bash
brew upgrade bazel
```

## Step 3. Install Android Prerequisites
Now that you have Bazel installed on your system, you need to have the Android NDK and SDK installed to be able to build with it.

### Step 3.1 NDK
The Android NDK is required to build the native (C/C++) TensorFlow code. The current recommended version is 12b, which may be found [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-12b-downloads).

### Step 3.2 SDK
The Android SDK and build tools may be obtained [here](https://developer.android.com/tools/revisions/build-tools.html), or alternatively as part of Android Studio. Build tools API >= 23 is required to build the TF Android training demo (though it will run on API >= 21 devices).

## Step 4. Edit WORKSPACE
The Android entries in (tensorflow/WORKSPACE)[https://github.com/tensorflow/tensorflow/blob/master/WORKSPACE#L19-L32] must be uncommented with the paths filled in appropriately depending on
where you installed the NDK and SDK. Otherwise an error such as: "The external label '//external:android/sdk' is not bound to
anything" will be reported.

Also edit the API levels for the SDK in WORKSPACE to the highest level you have installed in your SDK. This must be >= 23
(this is completely independent of the API level of the demo, which is defined in AndroidManifest.xml). The NDK API level may
remain at 14.

## Step 5. Confirm Your Setup Is Working
The Tensorflow Team has provided an Android demo as an example application. Build this app with:

```bash
bazel build -c opt //tensorflow/examples/android:tensorflow_demo
```

If you get build errors about protocol buffers, run git submodule update --init and make sure that you've modified your
WORKSPACE file as instructed, then try building again.

This build command must succeed to ensure that your setup is functional.

## Step 6. Android Studio Setup
Now that you have confirmed your Bazel build is working, you can now open this project in Android Studio.

First, change the [pathToTF](https://github.com/chelexa/tensorflow-on-android/blob/master/tf_android/build.gradle#L29)
to reflect your system path to the tensorflow root (wherever you put it in Step 1).

For the most part, the build.gradle file shouldn't need to be modified.
However, you may need to fix some environment issues in the build.gradle file. For example, the [buildToolsVersion](https://github.com/chelexa/tensorflow-on-android/blob/master/tf_android/build.gradle#L70)
in the current version may need to be updated to your specific setup. You also should verify that the [bazelLocation](https://github.com/chelexa/tensorflow-on-android/blob/master/tf_android/build.gradle#L47)
matches the output of

```bash
which bazel
```

Any errors here will need to be addressed before building.

## Step 7. Confirm Your Android Studio Setup Is Working
The build.gradle file loads sources sets from Tensorflow's Java API. You should see two additional source roots appear in your project tree.

Once you see these sources, you now need to add the TensorflowTrainingInterface to your project.
Find the TensorFlowTrainingInterface.java file at the root of this repo. Copy and paste this file into the tensorflow/tensorflow/contrib/android/java directory. You can now build and run the application (green play button).

## Common Errors
Most common errors can be fixed by ensuring:
 - You have you most up-to-date version of the TF repo
 - You have the most up-to-date version of Bazel installed

### NoClassDefFoundError
This is a runtime error related to running code from either of the external TF sources (The TF Inference library, or the TF Java API).
This was solved for me by running from the menu bar Build > Clean Project
