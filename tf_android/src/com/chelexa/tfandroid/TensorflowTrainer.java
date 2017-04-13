package com.chelexa.tfandroid;

import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowTrainingInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by tylerzeller on 4/1/17.
 */

public class TensorflowTrainer {

    private static final String TAG = "TensorFlowTrainer";

    private static int BATCH_SIZE = 1; // When this is 1, algorithm is SGD
    private static int D = 784;
    private static int K = 10;
    private static int N = 60000;
    private static int testN = 1000;
    private static int stepsToTest = 10;
    private static float[] testFeatures;
    private static float[] testLabels;

    // Config values.
    private String inputName;
    private String outputName;
    private String costName;
    private String trainName;
    private String initName;
    private String testName;
    private AssetManager assetManager;
    private float[] outputs;
    private float[] weights;

    private boolean logStats = false;

    private TensorFlowTrainingInterface trainingInterface;

    private TensorflowTrainer() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param costName The filepath of label file for classes.
     * @param initName The input size. A square image of inputSize x inputSize is assumed.
     * @param trainName The assumed mean of the image values.
     * @param inputName The name of the input node.
     * @param labelName The name of the label node.
     * @param testName The name of the accuracy node.
     */
    public static TensorflowTrainer create(
            AssetManager assetManager,
            String modelFilename,
            String initName,
            String costName,
            String trainName,
            String inputName,
            String labelName,
            String testName) {
        TensorflowTrainer c = new TensorflowTrainer();
        c.inputName = inputName;
        c.outputName = labelName;
        c.costName = costName;
        c.trainName = trainName;
        c.initName = initName;
        c.testName = testName;
        c.outputs = new float[1];
        c.weights = new float[K * D];
        c.assetManager = assetManager;
        c.trainingInterface = new TensorFlowTrainingInterface(assetManager, modelFilename);

        // testing data
        c.readTestingFeatures();
        c.readTestingLabels();

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
//        final Operation operation = c.trainingInterface.graphOperation(outputName);
//        final int numClasses = (int) operation.output(0).shape().size(1);
//        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
//        c.inputSize = inputSize;
//        c.imageMean = imageMean;
//        c.imageStd = imageStd;

        // Pre-allocate buffers.
//        c.outputNames = new String[] {outputName};
//        c.intValues = new int[inputSize * inputSize];
//        c.floatValues = new float[inputSize * inputSize * 3];
//        c.outputs = new float[numClasses];

        return c;
    }

    public void begin(final int numIterations, final TrainingActivity trainingActivity) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("beginTraining");

//        float[] featureFloats = new float[100 * 32];
//        float[] labelFloats = new float[100 * 8];
//
//        Random rng = new Random();
//        for (int i = 0; i < 100; i++){
//            for (int j = 0; j < 32; j++){
//                featureFloats[32 * i + j] = rng.nextFloat();
//            }
//
//            for (int j = 0; j < 8; j++){
//                labelFloats[8 * i + j] = rng.nextFloat();
//            }
//        }

        Trace.beginSection("init_vars");
        //trainingInterface.feed(initName, new float[0], 0);
        trainingInterface.run(new String[]{}, new String[]{initName}, logStats);
        Trace.endSection();

        for (int i = 0; i < numIterations; i++) {

            float[] trainFeatureBatch;
            float[] trainLabelBatch;

            int[] indices = new int[BATCH_SIZE];
            for (int j = 0; j < BATCH_SIZE; j++){
                indices[j] = new Random().nextInt(N);
            }

            // Get the training feature
            trainFeatureBatch = getFeatureBatch(indices);
            // Get the training label
            trainLabelBatch = getLabelBatch(indices);

            // Copy the input data into TensorFlow.
            //Trace.beginSection("feed");
            //trainingInterface.feed(inputName, featureFloats, 100, 32);
            //trainingInterface.feed(outputName, labelFloats, 100, 8);
            //Trace.endSection();

            // Run the inference call.
            //Trace.beginSection("run");
            //trainingInterface.run(new String[]{costName}, new String[]{}, logStats);
            //Trace.endSection();

            // Copy the output Tensor back into the output array.
            //Trace.beginSection("fetch");
            //trainingInterface.fetch(costName, outputs);
            //Trace.endSection();

            //trainingActivity.LogToView("L2 loss", outputs[0] + "");

            // Copy the training data into TensorFlow.
            Trace.beginSection("feed");
            trainingInterface.feed(inputName, trainFeatureBatch, BATCH_SIZE, D);
            trainingInterface.feed(outputName, trainLabelBatch, BATCH_SIZE, K);
            Trace.endSection();

            // Run a single step of training
            Trace.beginSection("train");
            trainingInterface.run(new String[]{}, new String[]{trainName}, logStats);
            Trace.endSection();

            // Copy the weights Tensor into the weights array.
            //Trace.beginSection("fetch");
            //trainingInterface.fetch("weights", weights);
            //Trace.endSection();
            //Log.d("weights", weights[0] + "");

            trainingActivity.LogToView("Iteration", i + "", "iteration");

            if (i == 0 || (i+1) % stepsToTest == 0){
                // Copy the test data into TensorFlow.
                Trace.beginSection("feed");
                trainingInterface.feed(inputName, testFeatures, testN, D);
                trainingInterface.feed(outputName, testLabels, testN, K);
                Trace.endSection();

                // Run the inference call.
                Trace.beginSection("test");
                trainingInterface.run(new String[]{testName}, new String[]{}, logStats);
                Trace.endSection();

                // Copy the accuracy Tensor back into the output array.
                Trace.beginSection("fetch");
                trainingInterface.fetch(testName, outputs);
                Trace.endSection();

                trainingActivity.LogToView("Test Accuracy", outputs[0] * 100 + "%", "accuracy");
                Trace.endSection(); // "beginTraining"
                if (((int)outputs[0]) == 1){
                    break;
                }
            }
        }

        // Copy the test data into TensorFlow.
//        Trace.beginSection("feed");
//        trainingInterface.feed(inputName, testFeatures, 220, 784);
//        trainingInterface.feed(outputName, testLabels, 220, 2);
//        Trace.endSection();
//
//        // Run the inference call.
//        Trace.beginSection("test");
//        trainingInterface.run(new String[]{testName}, new String[]{}, logStats);
//        Trace.endSection();
//
//        // Copy the output Tensor back into the output array.
//        Trace.beginSection("fetch");
//        trainingInterface.fetch(testName, outputs);
//        Trace.endSection();
//
//        trainingActivity.LogToView("Test accuracy", outputs[0] + "");
//        Trace.endSection(); // "beginTraining"
    }

    private float[] getFeatureBatch(int[] indices) {
        // android studio was giving me some bs error about using Arrays.asList ???
        List<Integer> idcs = new ArrayList<>();
        for (int id : indices){
            idcs.add(id);
        }

        float[] trainingFeatures = new float[BATCH_SIZE * D];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(assetManager.open("mnist_data/MNISTTrainImages.dat")));

            // do reading, usually loop until end of file reading
            int count = 0;
            int samples = 0;
            String line;
            String[] features;
            while ((line = reader.readLine()) != null) {
                // we've gathered all the batch samples
                if (samples == idcs.size()){
                    break;
                }

                //process line
                if (idcs.contains(count)){
                    features = line.split(",| ");
                    for(int i = 0; i < D; i++) {
                        trainingFeatures[samples * D + i] = Float.parseFloat(features[i]);
                        //sampleFeatures[i] = Double.parseDouble(features[i]);
                    }
                    samples++;
                }
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }

        return trainingFeatures;
    }

    private void readTestingFeatures() {
        Log.d("readTestingFeatures","Begin");
        testFeatures = new float[testN * D];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(assetManager.open("mnist_data/MNISTTestImages.dat")));

            // do reading, usually loop until end of file reading
            int count = 0;
            String line;
            String[] features;
            while ((line = reader.readLine()) != null) {
                //process line
                features = line.split(",| ");
                for(int i = 0; i < D; i++) {
                    testFeatures[count * D + i] = Float.parseFloat(features[i]);
                    //sampleFeatures[i] = Double.parseDouble(features[i]);
                }
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    private float[] getLabelBatch(int[] indices) {
        Log.d("getLabelBatch","Begin");
        // android studio was giving me some bs error about using Arrays.asList ???
        List<Integer> idcs = new ArrayList<>();
        for (int id : indices){
            idcs.add(id);
        }

        float[] trainingLabels = new float[BATCH_SIZE * K];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(assetManager.open("mnist_data/MNISTTrainLabels.dat")));

            // do reading, usually loop until end of file reading
            int i = 0;
            int sample = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                if (sample == idcs.size()){
                    break;
                }
                //process line
                if (idcs.contains(i)){
                    line = line.trim();
                    int sampleLabel = Integer.parseInt(line);
                    // For 1-hot encoding
                    trainingLabels[K * sample + sampleLabel] = 1;
                    sample++;
                }
                i++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return trainingLabels;
    }

    private void readTestingLabels() {
        Log.d("readTestingLabels","Begin");
        testLabels = new float[testN * K];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(assetManager.open("mnist_data/MNISTTestLabels.dat")));

            // do reading, usually loop until end of file reading
            int i = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                //process line
                line = line.trim();
                int sampleLabel = Integer.parseInt(line);
                // For 1-hot encoding
                testLabels[K * i + sampleLabel] = 1;
                i++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    public String getStatString() {
        return trainingInterface.getStatString();
    }

    public void close() {
        trainingInterface.close();
        testFeatures = null;
        testLabels = null;
    }
}
