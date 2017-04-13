package com.chelexa.tfandroid;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

public class TrainingActivity extends Activity implements View.OnClickListener {

    private static final String MODEL_FILE = "file:///android_asset/mnist_50_mlp.pb";

    private static final String INIT_NAME = "init";
    private static final String INPUT_NAME = "x";
    private static final String LABEL_NAME = "y";
    private static final String COST_NAME = "cost";
    private static final String TRAIN_NAME = "train";
    private static final String TEST_NAME = "test";

    private TensorflowTrainer trainer;

    private boolean debug = false;
    private boolean computing = false;

    private Handler handler;
    private HandlerThread handlerThread;

    private ToggleButton toggleButton;
    private TextView iterationView;
    private TextView accuracyView;
    private EditText numIterations;

    private void startHandlerThread(){
        if (handlerThread == null) {
            handlerThread = new HandlerThread("training");
            handlerThread.start();
            handler = new Handler(handlerThread.getLooper());
        }
    }

    private void stopHandlerThread(){
        if (handlerThread != null){
            handlerThread.quitSafely();
            try {
                handlerThread.join();
                handlerThread = null;
                handler = null;
            } catch (final InterruptedException e) {
                Log.e("stopHandlerThread", "Exception!");
            }
        }
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        Log.d("onCreate", "onCreate");
        super.onCreate(null);
        setContentView(R.layout.training_activity);
        toggleButton = (ToggleButton)findViewById(R.id.toggleButton);
        toggleButton.setOnClickListener(this);
        iterationView = (TextView)findViewById(R.id.textView);
        //iterationView.setMovementMethod(new ScrollingMovementMethod());
        accuracyView = (TextView)findViewById(R.id.accuracy);
        numIterations = (EditText)findViewById(R.id.editText);
    }

    @Override
    public synchronized void onStart() {
        Log.d("onStart", "onStart");
        super.onStart();
    }

    @Override
    public synchronized void onResume() {
        Log.d("onResume", "onResume");
        super.onResume();
    }

    @Override
    public synchronized void onPause() {
        Log.d("onStart", "onStart");

        if (!isFinishing()) {
            Log.d("onPause", "requesting finish");
            finish();
        }

        stopHandlerThread();
        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        Log.d("onStop", "onStop");
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        Log.d("onDestroy", "onDestroy");
        super.onDestroy();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    public void onClick(View view) {
        if (toggleButton.isChecked()){
            if (numIterations.getText().toString().isEmpty()){
                Toast.makeText(this, "Please enter a number for the number of iterations.", Toast.LENGTH_SHORT).show();
                toggleButton.setChecked(false);
            } else {
                startHandlerThread();

                Trace.beginSection("beginTrain");

                computing = true;

                trainer =
                        TensorflowTrainer.create(
                                getAssets(),
                                MODEL_FILE,
                                INIT_NAME,
                                COST_NAME,
                                TRAIN_NAME,
                                INPUT_NAME,
                                LABEL_NAME,
                                TEST_NAME);

                runInBackground(
                        new Runnable() {
                            @Override
                            public void run() {
                                trainer.begin(Integer.parseInt(numIterations.getText().toString()), TrainingActivity.this);
                                computing = false;
                            }
                        });

                Trace.endSection();
            }
        } else {
            runInBackground(new Runnable() {
                @Override
                public void run() {
                    trainer.close();

                }
            });
            stopHandlerThread();
        }
    }

    public void LogToView(final String title, final String message, final String which) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Log.d(title, message);
                if (which.equals("iteration")){
                    iterationView.setText(title + ":  " + message);
                }
                else if (which.equals("accuracy")){
                    accuracyView.setText(title + ":  " + message);
                }
            }
        });

    }
}
