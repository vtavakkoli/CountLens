package com.ai.countlens;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
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
    private Bitmap selectedBitmap;
    private SettingsManager settingsManager;

    private final ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Bundle extras = result.getData().getExtras();
                    selectedBitmap = (Bitmap) extras.get("data");
                    ivPhoto.setImageBitmap(selectedBitmap);
                }
            }
    );

    private final ActivityResultLauncher<String> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            uri -> {
                if (uri != null) {
                    try {
                        selectedBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                        ivPhoto.setImageBitmap(selectedBitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                    }
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

        selectionOverlay.setSelectionShape(settingsManager.getSelectionShape());

        boolean fromCamera = getIntent().getBooleanExtra("from_camera", false);
        if (fromCamera) {
            launchCamera();
        } else {
            launchGallery();
        }

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

    private void performDetection() {
        if (selectedBitmap == null) {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show();
            return;
        }

        RectF selection = selectionOverlay.getSelectionRect();
        if (selection.width() <= 0 || selection.height() <= 0) {
            Toast.makeText(this, "Please select an object first", Toast.LENGTH_SHORT).show();
            return;
        }

        // Convert selection from View coordinates to Bitmap coordinates
        RectF bitmapSelection = convertToBitmapCoordinates(selection);

        tvStatus.setText("Detecting...");
        OpenCVDetectionHelper.detectSimilarObjects(selectedBitmap, bitmapSelection, settingsManager.getThreshold(), (resultBitmap, count) -> {
            runOnUiThread(() -> {
                ivPhoto.setImageBitmap(resultBitmap);
                selectedBitmap = resultBitmap; // Update for further saving
                tvResultCount.setText(getString(R.string.result_count_format, count));
                tvResultCount.setVisibility(View.VISIBLE);
                tvStatus.setText("Detection Complete");
            });
        });
    }

    private RectF convertToBitmapCoordinates(RectF viewRect) {
        // This is a simplified conversion
        // Real conversion needs to account for ImageView scaleType (fitCenter)
        float viewWidth = ivPhoto.getWidth();
        float viewHeight = ivPhoto.getHeight();
        float bitmapWidth = selectedBitmap.getWidth();
        float bitmapHeight = selectedBitmap.getHeight();

        float scale;
        float xOffset = 0;
        float yOffset = 0;

        if (bitmapWidth / bitmapHeight > viewWidth / viewHeight) {
            scale = viewWidth / bitmapWidth;
            yOffset = (viewHeight - bitmapHeight * scale) / 2;
        } else {
            scale = viewHeight / bitmapHeight;
            xOffset = (viewWidth - bitmapWidth * scale) / 2;
        }

        float left = (viewRect.left - xOffset) / scale;
        float top = (viewRect.top - yOffset) / scale;
        float right = (viewRect.right - xOffset) / scale;
        float bottom = (viewRect.bottom - yOffset) / scale;

        return new RectF(left, top, right, bottom);
    }

    private void resetSelection() {
        selectionOverlay.reset();
        tvResultCount.setVisibility(View.GONE);
        tvStatus.setText(R.string.status_select_object);
    }

    private void saveResult() {
        if (selectedBitmap == null) {
            Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show();
            return;
        }
        
        String savedImageURL = MediaStore.Images.Media.insertImage(
                getContentResolver(),
                selectedBitmap,
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