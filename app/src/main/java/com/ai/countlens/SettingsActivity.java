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
    private TextView tvNmsLabel;
    private TextView tvImageSizeLabel;
    private Slider sliderThreshold;
    private Slider sliderNms;
    private Slider sliderImageSize;
    private RadioGroup rgShape;
    private RadioButton rbRectangle, rbCircle;
    private Button btnBack;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        settingsManager = new SettingsManager(this);

        tvThresholdLabel = findViewById(R.id.tv_threshold_label);
        tvNmsLabel = findViewById(R.id.tv_nms_label);
        tvImageSizeLabel = findViewById(R.id.tv_image_size_label);
        sliderThreshold = findViewById(R.id.slider_threshold);
        sliderNms = findViewById(R.id.slider_nms);
        sliderImageSize = findViewById(R.id.slider_image_size);
        rgShape = findViewById(R.id.rg_shape);
        rbRectangle = findViewById(R.id.rb_rectangle);
        rbCircle = findViewById(R.id.rb_circle);
        btnBack = findViewById(R.id.btn_back);

        float currentThreshold = clamp(settingsManager.getThreshold(), sliderThreshold.getValueFrom(), sliderThreshold.getValueTo());
        sliderThreshold.setValue(currentThreshold);
        updateThresholdLabel(currentThreshold);

        float currentNms = clamp(settingsManager.getNmsThreshold(), sliderNms.getValueFrom(), sliderNms.getValueTo());
        sliderNms.setValue(currentNms);
        updateNmsLabel(currentNms);

        float currentImageSize = clamp(settingsManager.getMaxImageSize(), sliderImageSize.getValueFrom(), sliderImageSize.getValueTo());
        sliderImageSize.setValue(currentImageSize);
        updateImageSizeLabel(Math.round(currentImageSize));

        String currentShape = settingsManager.getSelectionShape();
        if (SettingsManager.SHAPE_CIRCLE.equals(currentShape)) {
            rbCircle.setChecked(true);
        } else {
            rbRectangle.setChecked(true);
        }

        sliderThreshold.addOnChangeListener((slider, value, fromUser) -> {
            settingsManager.setThreshold(value);
            updateThresholdLabel(value);
        });

        sliderNms.addOnChangeListener((slider, value, fromUser) -> {
            settingsManager.setNmsThreshold(value);
            updateNmsLabel(value);
        });

        sliderImageSize.addOnChangeListener((slider, value, fromUser) -> {
            int rounded = Math.round(value / 64f) * 64;
            settingsManager.setMaxImageSize(rounded);
            updateImageSizeLabel(rounded);
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

    private void updateNmsLabel(float value) {
        tvNmsLabel.setText(getString(R.string.label_nms_threshold, value));
    }

    private void updateImageSizeLabel(int value) {
        tvImageSizeLabel.setText(getString(R.string.label_image_size, value));
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
}
