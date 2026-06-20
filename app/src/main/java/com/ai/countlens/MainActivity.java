package com.ai.countlens;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Button btnTakePhoto = findViewById(R.id.btn_take_photo);
        Button btnChooseGallery = findViewById(R.id.btn_choose_gallery);
        Button btnSettings = findViewById(R.id.btn_settings);
        Button btnAbout = findViewById(R.id.btn_about);

        btnTakePhoto.setOnClickListener(v -> {
            // TODO: Implement camera logic
            startImageDetection(true);
        });

        btnChooseGallery.setOnClickListener(v -> {
            // TODO: Implement gallery logic
            startImageDetection(false);
        });

        btnSettings.setOnClickListener(v -> {
            startActivity(new Intent(this, SettingsActivity.class));
        });

        btnAbout.setOnClickListener(v -> {
            // TODO: Show About Screen or Dialog
            showAbout();
        });
    }

    private void startImageDetection(boolean fromCamera) {
        Intent intent = new Intent(this, ImageDetectionActivity.class);
        intent.putExtra("from_camera", fromCamera);
        startActivity(intent);
    }

    private void showAbout() {
        Intent intent = new Intent(this, AboutActivity.class);
        startActivity(intent);
    }
}