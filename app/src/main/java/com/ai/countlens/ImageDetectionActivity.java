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
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;

public class ImageDetectionActivity extends AppCompatActivity {

    private ZoomableImageView ivPhoto;
    private SelectionOverlayView selectionOverlay;
    private TextView tvStatus, tvTopHint, tvResultCount;
    private Button btnDetect, btnReset, btnSave, btnZoomMode, btnFitImage;
    private ImageButton btnBack;
    private ProgressBar progressDetection;
    private Bitmap sourceBitmap;
    private Bitmap resultBitmap;
    private SettingsManager settingsManager;
    private boolean zoomMode = false;

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
        tvTopHint = findViewById(R.id.tv_top_hint);
        tvResultCount = findViewById(R.id.tv_result_count);
        btnDetect = findViewById(R.id.btn_detect);
        btnReset = findViewById(R.id.btn_reset);
        btnSave = findViewById(R.id.btn_save);
        btnZoomMode = findViewById(R.id.btn_zoom_mode);
        btnFitImage = findViewById(R.id.btn_fit_image);
        btnBack = findViewById(R.id.btn_back);
        progressDetection = findViewById(R.id.progress_detection);

        selectionOverlay.setSelectionShape(settingsManager.getSelectionShape());
        setStatusText(R.string.loading_image);
        setZoomMode(false);

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
        btnZoomMode.setOnClickListener(v -> setZoomMode(!zoomMode));
        btnFitImage.setOnClickListener(v -> ivPhoto.resetZoom());
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
        // Large phone photos can be 3000–6000 px and make multi-scale CV very slow.
        // CountLens now keeps an analysis-sized bitmap with the same aspect ratio.
        // This makes detection much faster and still accurate enough for bounding boxes.
        sourceBitmap = resizeBitmapForAnalysis(bitmap, settingsManager.getMaxImageSize());
        resultBitmap = null;
        ivPhoto.setImageBitmap(sourceBitmap);
        ivPhoto.resetZoom();
        selectionOverlay.reset();
        selectionOverlay.setVisibility(View.VISIBLE);
        tvResultCount.setVisibility(View.GONE);
        setStatusText(R.string.status_ready);
        setZoomMode(false);
    }

    private Bitmap resizeBitmapForAnalysis(Bitmap bitmap, int maxSide) {
        if (bitmap == null) return null;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int longest = Math.max(width, height);
        if (longest <= maxSide) {
            return bitmap.copy(Bitmap.Config.ARGB_8888, true);
        }

        float scale = maxSide / (float) longest;
        int newWidth = Math.max(1, Math.round(width * scale));
        int newHeight = Math.max(1, Math.round(height * scale));
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
        return resized.copy(Bitmap.Config.ARGB_8888, true);
    }

    private void performDetection() {
        if (sourceBitmap == null) {
            Toast.makeText(this, R.string.error_no_image, Toast.LENGTH_SHORT).show();
            return;
        }

        if (zoomMode) {
            setZoomMode(false);
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
                selectionOverlay.getRotationAngle(),
                settingsManager.getThreshold(),
                settingsManager.getNmsThreshold(),
                (detectedBitmap, count) -> runOnUiThread(() -> {
                    resultBitmap = detectedBitmap;
                    ivPhoto.setImageBitmap(resultBitmap);
                    // Hide the orange selection handles after detection. Otherwise users can confuse
                    // the selected template box with real detections.
                    selectionOverlay.setVisibility(View.GONE);
                    tvResultCount.setText(getString(R.string.result_count_format, count));
                    tvResultCount.setVisibility(View.VISIBLE);
                    setStatusText(count > 0 ? R.string.status_detection_complete : R.string.error_detection_failed);
                    setDetectingState(false);
                })
        )).start();
    }

    private void setDetectingState(boolean detecting) {
        progressDetection.setVisibility(detecting ? View.VISIBLE : View.GONE);

        // Keep the buttons visually readable. Material disabled colors were too pale on
        // some phones, so we explicitly control alpha and re-enable every control when
        // detection is finished.
        setButtonState(btnDetect, !detecting);
        setButtonState(btnReset, !detecting);
        setButtonState(btnSave, !detecting);
        setButtonState(btnZoomMode, !detecting);
        setButtonState(btnFitImage, !detecting);

        selectionOverlay.setEnabled(!detecting && !zoomMode);
        ivPhoto.setZoomEnabled(!detecting && zoomMode);
        if (detecting) {
            setStatusText(R.string.status_detecting_fast);
        }
    }

    private void setButtonState(Button button, boolean enabled) {
        button.setEnabled(enabled);
        button.setAlpha(enabled ? 1.0f : 0.70f);
    }

    private void setStatusText(int stringResId) {
        tvStatus.setText(stringResId);
        if (tvTopHint != null) {
            tvTopHint.setText(stringResId);
        }
    }

    private void setZoomMode(boolean enabled) {
        zoomMode = enabled;
        ivPhoto.setZoomEnabled(enabled);
        selectionOverlay.setEnabled(!enabled);
        selectionOverlay.setVisibility(enabled ? View.GONE : View.VISIBLE);
        btnZoomMode.setText(enabled ? R.string.btn_select_mode : R.string.btn_zoom_mode);
        setStatusText(enabled ? R.string.status_zoom_mode : R.string.status_select_object);
    }

    private RectF convertToBitmapCoordinates(RectF viewRect) {
        return ivPhoto.viewRectToBitmapRect(viewRect);
    }

    private void resetSelection() {
        selectionOverlay.reset();
        selectionOverlay.setVisibility(View.VISIBLE);
        ivPhoto.resetZoom();
        if (sourceBitmap != null) {
            ivPhoto.setImageBitmap(sourceBitmap);
        }
        resultBitmap = null;
        tvResultCount.setVisibility(View.GONE);
        setStatusText(R.string.status_select_object);
        setZoomMode(false);
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
