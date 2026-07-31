package com.ai.countlens;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;

public class SplashActivity extends AppCompatActivity {
    private static final String TAG = "SplashActivity";
    private static final long MINIMUM_SPLASH_MILLIS = 550L;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Runnable openMainScreen = () -> {
        if (!isFinishing() && !isDestroyed()) {
            startActivity(new Intent(this, MainActivity.class));
            finish();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed");
            Toast.makeText(this, R.string.error_opencv_init, Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        Log.i(TAG, "OpenCV loaded successfully");
        handler.postDelayed(openMainScreen, MINIMUM_SPLASH_MILLIS);
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacks(openMainScreen);
        super.onDestroy();
    }
}
