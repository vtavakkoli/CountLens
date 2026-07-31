package com.ai.countlens;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ImageDetectionActivity extends AppCompatActivity {
    private static final String STATE_CAMERA_URI = "state_camera_uri";

    private ZoomableImageView ivPhoto;
    private SelectionOverlayView selectionOverlay;
    private TextView tvStatus, tvTopHint, tvResultCount;
    private Button btnDetect, btnAutoCount, btnReset, btnSave, btnZoomMode, btnFitImage;
    private ImageButton btnBack;
    private ProgressBar progressDetection;
    private Bitmap sourceBitmap;
    private Bitmap resultBitmap;
    private Bitmap pendingSaveBitmap;
    private SettingsManager settingsManager;
    private boolean zoomMode = false;
    private Uri pendingCameraUri;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Future<?> runningTask;

    private final ActivityResultLauncher<Uri> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.TakePicture(),
            success -> {
                if (success && pendingCameraUri != null) {
                    loadImage(pendingCameraUri);
                } else {
                    Toast.makeText(this, R.string.error_camera_cancelled, Toast.LENGTH_SHORT).show();
                    finish();
                }
            }
    );

    private final ActivityResultLauncher<String> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            uri -> {
                if (uri != null) {
                    loadImage(uri);
                } else {
                    finish();
                }
            }
    );

    private final ActivityResultLauncher<String> storagePermissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(),
            granted -> {
                Bitmap bitmap = pendingSaveBitmap;
                pendingSaveBitmap = null;
                if (granted && bitmap != null) {
                    saveBitmapAsync(bitmap);
                } else {
                    Toast.makeText(this, R.string.error_storage_permission, Toast.LENGTH_LONG).show();
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);

        settingsManager = new SettingsManager(this);
        if (savedInstanceState != null) {
            String cameraUri = savedInstanceState.getString(STATE_CAMERA_URI);
            if (cameraUri != null) {
                pendingCameraUri = Uri.parse(cameraUri);
            }
        }

        ivPhoto = findViewById(R.id.iv_photo);
        selectionOverlay = findViewById(R.id.selection_overlay);
        tvStatus = findViewById(R.id.tv_status);
        tvTopHint = findViewById(R.id.tv_top_hint);
        tvResultCount = findViewById(R.id.tv_result_count);
        btnDetect = findViewById(R.id.btn_detect);
        btnAutoCount = findViewById(R.id.btn_auto_count);
        btnReset = findViewById(R.id.btn_reset);
        btnSave = findViewById(R.id.btn_save);
        btnZoomMode = findViewById(R.id.btn_zoom_mode);
        btnFitImage = findViewById(R.id.btn_fit_image);
        btnBack = findViewById(R.id.btn_back);
        progressDetection = findViewById(R.id.progress_detection);

        selectionOverlay.setSelectionShape(settingsManager.getSelectionShape());
        setZoomMode(false);
        setStatusText(R.string.loading_image);

        btnBack.setOnClickListener(v -> finish());
        btnDetect.setOnClickListener(v -> performDetection());
        btnAutoCount.setOnClickListener(v -> performAutoCount());
        btnReset.setOnClickListener(v -> resetSelection());
        btnSave.setOnClickListener(v -> saveResult());
        btnZoomMode.setOnClickListener(v -> setZoomMode(!zoomMode));
        btnFitImage.setOnClickListener(v -> ivPhoto.resetZoom());

        if (savedInstanceState == null) {
            boolean fromCamera = getIntent().getBooleanExtra("from_camera", false);
            if (fromCamera) {
                launchCamera();
            } else {
                galleryLauncher.launch("image/*");
            }
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (pendingCameraUri != null) {
            outState.putString(STATE_CAMERA_URI, pendingCameraUri.toString());
        }
    }

    private void launchCamera() {
        try {
            pendingCameraUri = ImageIoUtils.createCameraUri(this);
            cameraLauncher.launch(pendingCameraUri);
        } catch (IOException error) {
            Toast.makeText(this, R.string.error_camera_start, Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void loadImage(Uri uri) {
        setDetectingState(true);
        setStatusText(R.string.loading_image);
        submitTask(() -> {
            try {
                Bitmap bitmap = ImageIoUtils.decodeBitmap(this, uri, settingsManager.getMaxImageSize());
                runOnUiThreadSafe(() -> setSourceBitmap(bitmap));
            } catch (IOException | RuntimeException error) {
                runOnUiThreadSafe(() -> {
                    setDetectingState(false);
                    Toast.makeText(this, R.string.error_load_image, Toast.LENGTH_LONG).show();
                    finish();
                });
            }
        });
    }

    private void setSourceBitmap(Bitmap bitmap) {
        sourceBitmap = bitmap;
        resultBitmap = null;
        ivPhoto.setImageBitmap(sourceBitmap);
        ivPhoto.resetZoom();
        selectionOverlay.reset();
        selectionOverlay.setVisibility(View.VISIBLE);
        tvResultCount.setVisibility(View.GONE);
        setZoomMode(false);
        setStatusText(R.string.status_ready);
        setDetectingState(false);
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

        RectF bitmapSelection = ivPhoto.viewRectToBitmapRect(selection);
        if (bitmapSelection.width() <= 8 || bitmapSelection.height() <= 8) {
            Toast.makeText(this, R.string.error_select_object, Toast.LENGTH_SHORT).show();
            return;
        }

        setDetectingState(true);
        setStatusText(R.string.status_detecting);
        submitTask(() -> OpenCVDetectionHelper.detectSimilarObjects(
                sourceBitmap,
                bitmapSelection,
                selectionOverlay.getRotationAngle(),
                settingsManager.getThreshold(),
                settingsManager.getNmsThreshold(),
                this::completeDetection
        ));
    }

    private void performAutoCount() {
        if (sourceBitmap == null) {
            Toast.makeText(this, R.string.error_no_image, Toast.LENGTH_SHORT).show();
            return;
        }
        if (zoomMode) {
            setZoomMode(false);
        }

        setDetectingState(true);
        setStatusText(R.string.status_auto_counting);
        submitTask(() -> AdaptiveAutoCounter.countRepeatedObjects(
                sourceBitmap,
                settingsManager.getThreshold(),
                this::completeDetection
        ));
    }

    private void completeDetection(Bitmap detectedBitmap, int count) {
        if (Thread.currentThread().isInterrupted()) {
            if (detectedBitmap != sourceBitmap && !detectedBitmap.isRecycled()) {
                detectedBitmap.recycle();
            }
            return;
        }
        runOnUiThreadSafe(() -> {
            resultBitmap = detectedBitmap;
            ivPhoto.setImageBitmap(resultBitmap);
            ivPhoto.resetZoom();
            selectionOverlay.setVisibility(View.GONE);
            tvResultCount.setText(getString(R.string.result_count_format, count));
            tvResultCount.setVisibility(View.VISIBLE);
            setStatusText(count > 0
                    ? R.string.status_detection_complete
                    : R.string.error_detection_failed);
            setDetectingState(false);
        });
    }

    private void setDetectingState(boolean detecting) {
        progressDetection.setVisibility(detecting ? View.VISIBLE : View.GONE);
        setButtonState(btnDetect, !detecting);
        setButtonState(btnAutoCount, !detecting);
        setButtonState(btnReset, !detecting);
        setButtonState(btnSave, !detecting);
        setButtonState(btnZoomMode, !detecting);
        setButtonState(btnFitImage, !detecting);
        selectionOverlay.setEnabled(!detecting && !zoomMode);
        ivPhoto.setZoomEnabled(!detecting && zoomMode);
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
        if (sourceBitmap != null) {
            setStatusText(enabled ? R.string.status_zoom_mode : R.string.status_select_object);
        }
    }

    private void resetSelection() {
        cancelRunningTask();
        selectionOverlay.reset();
        selectionOverlay.setVisibility(View.VISIBLE);
        if (sourceBitmap != null) {
            ivPhoto.setImageBitmap(sourceBitmap);
            ivPhoto.resetZoom();
        }
        resultBitmap = null;
        tvResultCount.setVisibility(View.GONE);
        setZoomMode(false);
        setStatusText(R.string.status_select_object);
        setDetectingState(false);
    }

    private void saveResult() {
        Bitmap bitmapToSave = resultBitmap != null ? resultBitmap : sourceBitmap;
        if (bitmapToSave == null) {
            Toast.makeText(this, R.string.error_no_image_to_save, Toast.LENGTH_SHORT).show();
            return;
        }

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P
                && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            pendingSaveBitmap = bitmapToSave;
            storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            return;
        }
        saveBitmapAsync(bitmapToSave);
    }

    private void saveBitmapAsync(Bitmap bitmap) {
        setDetectingState(true);
        setStatusText(R.string.status_saving);
        submitTask(() -> {
            try {
                ImageIoUtils.saveToGallery(this, bitmap);
                runOnUiThreadSafe(() -> {
                    setDetectingState(false);
                    setStatusText(resultBitmap != null
                            ? R.string.status_detection_complete
                            : R.string.status_ready);
                    Toast.makeText(this, R.string.message_image_saved, Toast.LENGTH_LONG).show();
                });
            } catch (IOException | RuntimeException error) {
                runOnUiThreadSafe(() -> {
                    setDetectingState(false);
                    setStatusText(R.string.error_save_image);
                    Toast.makeText(this, R.string.error_save_image, Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private void submitTask(Runnable task) {
        cancelRunningTask();
        runningTask = worker.submit(task);
    }

    private void cancelRunningTask() {
        if (runningTask != null && !runningTask.isDone()) {
            runningTask.cancel(true);
        }
        runningTask = null;
    }

    private void runOnUiThreadSafe(Runnable action) {
        runOnUiThread(() -> {
            if (!isFinishing() && !isDestroyed()) {
                action.run();
            }
        });
    }

    @Override
    protected void onDestroy() {
        cancelRunningTask();
        worker.shutdownNow();
        super.onDestroy();
    }
}
