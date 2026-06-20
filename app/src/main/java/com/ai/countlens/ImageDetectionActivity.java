package com.ai.countlens;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;

public class ImageDetectionActivity extends AppCompatActivity {

    private ImageView ivPhoto;
    private SelectionOverlayView selectionOverlay;
    private TextView tvStatus, tvResultCount;
    private Button btnDetect, btnReset, btnSave;
    private ImageButton btnBack;
    private ProgressBar progressDetection;
    private Bitmap sourceBitmap;
    private Bitmap resultBitmap;
    private SettingsManager settingsManager;

    private final ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Bundle extras = result.getData().getExtras();
                    Bitmap capturedBitmap = extras != null ? (Bitmap) extras.get("data") : null;
                    if (capturedBitmap != null) {
                        setSourceBitmap(capturedBitmap);
                    } else {
                        Toast.makeText(this, R.string.error_no_image, Toast.LENGTH_SHORT).show();
                    }
                } else {
                    finish();
                }
            }
    );

    private final ActivityResultLauncher<String> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            uri -> {
                if (uri != null) {
                    loadImageFromGallery(uri);
                } else {
                    finish();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);

        settingsManager = new SettingsManager(this);

        ivPhoto = findViewById(R.id.iv_photo);
        selectionOverlay = findViewById(R.id.selection_overlay);
        tvStatus = findViewById(R.id.tv_status);
        tvResultCount = findViewById(R.id.tv_result_count);
        btnDetect = findViewById(R.id.btn_detect);
        btnReset = findViewById(R.id.btn_reset);
        btnSave = findViewById(R.id.btn_save);
        btnBack = findViewById(R.id.btn_back);
        progressDetection = findViewById(R.id.progress_detection);

        selectionOverlay.setSelectionShape(settingsManager.getSelectionShape());
        tvStatus.setText(R.string.loading_image);

        boolean fromCamera = getIntent().getBooleanExtra("from_camera", false);
        if (fromCamera) {
            launchCamera();
        } else {
            launchGallery();
        }

        btnBack.setOnClickListener(v -> finish());
        btnDetect.setOnClickListener(v -> performDetection());
        btnReset.setOnClickListener(v -> resetSelection());
        btnSave.setOnClickListener(v -> saveResult());
    }

    private void launchCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(takePictureIntent);
    }

    private void launchGallery() {
        galleryLauncher.launch("image/*");
    }

    private void loadImageFromGallery(Uri uri) {
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            setSourceBitmap(bitmap);
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    private void setSourceBitmap(Bitmap bitmap) {
        sourceBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        resultBitmap = null;
        ivPhoto.setImageBitmap(sourceBitmap);
        selectionOverlay.reset();
        tvResultCount.setVisibility(View.GONE);
        tvStatus.setText(R.string.status_ready);
    }

    private void performDetection() {
        if (sourceBitmap == null) {
            Toast.makeText(this, R.string.error_no_image, Toast.LENGTH_SHORT).show();
            return;
        }

        RectF selection = selectionOverlay.getSelectionRect();
        if (selection.width() <= 8 || selection.height() <= 8) {
            Toast.makeText(this, R.string.error_select_object, Toast.LENGTH_SHORT).show();
            return;
        }

        RectF bitmapSelection = convertToBitmapCoordinates(selection);
        if (bitmapSelection.width() <= 8 || bitmapSelection.height() <= 8) {
            Toast.makeText(this, R.string.error_select_object, Toast.LENGTH_SHORT).show();
            return;
        }

        setDetectingState(true);

        new Thread(() -> OpenCVDetectionHelper.detectSimilarObjects(
                sourceBitmap,
                bitmapSelection,
                settingsManager.getThreshold(),
                settingsManager.getNmsThreshold(),
                (detectedBitmap, count) -> runOnUiThread(() -> {
                    resultBitmap = detectedBitmap;
                    ivPhoto.setImageBitmap(resultBitmap);
                    tvResultCount.setText(getString(R.string.result_count_format, count));
                    tvResultCount.setVisibility(View.VISIBLE);
                    tvStatus.setText(count > 0 ? R.string.status_detection_complete : R.string.error_detection_failed);
                    setDetectingState(false);
                })
        )).start();
    }

    private void setDetectingState(boolean detecting) {
        progressDetection.setVisibility(detecting ? View.VISIBLE : View.GONE);
        btnDetect.setEnabled(!detecting);
        btnReset.setEnabled(!detecting);
        btnSave.setEnabled(!detecting);
        selectionOverlay.setEnabled(!detecting);
        if (detecting) {
            tvStatus.setText(R.string.status_detecting);
        }
    }

    private RectF convertToBitmapCoordinates(RectF viewRect) {
        float viewWidth = ivPhoto.getWidth();
        float viewHeight = ivPhoto.getHeight();
        float bitmapWidth = sourceBitmap.getWidth();
        float bitmapHeight = sourceBitmap.getHeight();

        if (viewWidth <= 0 || viewHeight <= 0 || bitmapWidth <= 0 || bitmapHeight <= 0) {
            return new RectF();
        }

        float viewRatio = viewWidth / viewHeight;
        float bitmapRatio = bitmapWidth / bitmapHeight;

        float scale;
        float xOffset = 0f;
        float yOffset = 0f;

        if (bitmapRatio > viewRatio) {
            scale = viewWidth / bitmapWidth;
            yOffset = (viewHeight - bitmapHeight * scale) / 2f;
        } else {
            scale = viewHeight / bitmapHeight;
            xOffset = (viewWidth - bitmapWidth * scale) / 2f;
        }

        float left = (viewRect.left - xOffset) / scale;
        float top = (viewRect.top - yOffset) / scale;
        float right = (viewRect.right - xOffset) / scale;
        float bottom = (viewRect.bottom - yOffset) / scale;

        left = clamp(left, 0, bitmapWidth - 1);
        top = clamp(top, 0, bitmapHeight - 1);
        right = clamp(right, left + 1, bitmapWidth);
        bottom = clamp(bottom, top + 1, bitmapHeight);

        return new RectF(left, top, right, bottom);
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private void resetSelection() {
        selectionOverlay.reset();
        if (sourceBitmap != null) {
            ivPhoto.setImageBitmap(sourceBitmap);
        }
        resultBitmap = null;
        tvResultCount.setVisibility(View.GONE);
        tvStatus.setText(R.string.status_select_object);
    }

    private void saveResult() {
        Bitmap bitmapToSave = resultBitmap != null ? resultBitmap : sourceBitmap;
        if (bitmapToSave == null) {
            Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show();
            return;
        }

        String savedImageURL = MediaStore.Images.Media.insertImage(
                getContentResolver(),
                bitmapToSave,
                "CountLens_" + System.currentTimeMillis(),
                "Detection Result from CountLens"
        );

        if (savedImageURL != null) {
            Toast.makeText(this, "Image saved to gallery", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show();
        }
    }
}
