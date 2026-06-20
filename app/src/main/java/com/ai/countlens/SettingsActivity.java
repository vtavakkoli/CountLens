package com.ai.countlens;

import android.os.Bundle;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.slider.Slider;

public class SettingsActivity extends AppCompatActivity {

    private SettingsManager settingsManager;
    private TextView tvThresholdLabel;
    private Slider sliderThreshold;
    private RadioGroup rgShape;
    private RadioButton rbRectangle, rbCircle;
    private Button btnBack;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        settingsManager = new SettingsManager(this);

        tvThresholdLabel = findViewById(R.id.tv_threshold_label);
        sliderThreshold = findViewById(R.id.slider_threshold);
        rgShape = findViewById(R.id.rg_shape);
        rbRectangle = findViewById(R.id.rb_rectangle);
        rbCircle = findViewById(R.id.rb_circle);
        btnBack = findViewById(R.id.btn_back);

        // Load current settings
        float currentThreshold = settingsManager.getThreshold();
        sliderThreshold.setValue(currentThreshold);
        updateThresholdLabel(currentThreshold);

        String currentShape = settingsManager.getSelectionShape();
        if (SettingsManager.SHAPE_CIRCLE.equals(currentShape)) {
            rbCircle.setChecked(true);
        } else {
            rbRectangle.setChecked(true);
        }

        // Listeners
        sliderThreshold.addOnChangeListener((slider, value, fromUser) -> {
            settingsManager.setThreshold(value);
            updateThresholdLabel(value);
        });

        rgShape.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rb_rectangle) {
                settingsManager.setSelectionShape(SettingsManager.SHAPE_RECTANGLE);
            } else if (checkedId == R.id.rb_circle) {
                settingsManager.setSelectionShape(SettingsManager.SHAPE_CIRCLE);
            }
        });

        btnBack.setOnClickListener(v -> finish());
    }

    private void updateThresholdLabel(float value) {
        tvThresholdLabel.setText(getString(R.string.label_threshold, value));
    }
}