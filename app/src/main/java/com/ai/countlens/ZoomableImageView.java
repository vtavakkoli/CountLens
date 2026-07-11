package com.ai.countlens;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import androidx.appcompat.widget.AppCompatImageView;

public class ZoomableImageView extends AppCompatImageView {
    private final Matrix imageMatrixValues = new Matrix();
    private ScaleGestureDetector scaleDetector;
    private boolean zoomEnabled = false;
    private float lastX;
    private float lastY;

    public ZoomableImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setScaleType(ScaleType.MATRIX);
        scaleDetector = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector detector) {
                imageMatrixValues.postScale(detector.getScaleFactor(), detector.getScaleFactor(), detector.getFocusX(), detector.getFocusY());
                setImageMatrix(imageMatrixValues);
                return true;
            }
        });
    }

    public void setZoomEnabled(boolean enabled) { zoomEnabled = enabled; }

    public void resetZoom() {
        imageMatrixValues.reset();
        if (getDrawable() == null || getWidth() == 0 || getHeight() == 0) {
            setImageMatrix(imageMatrixValues);
            return;
        }
        float scale = Math.min(getWidth() / (float) getDrawable().getIntrinsicWidth(), getHeight() / (float) getDrawable().getIntrinsicHeight());
        float dx = (getWidth() - getDrawable().getIntrinsicWidth() * scale) / 2f;
        float dy = (getHeight() - getDrawable().getIntrinsicHeight() * scale) / 2f;
        imageMatrixValues.postScale(scale, scale);
        imageMatrixValues.postTranslate(dx, dy);
        setImageMatrix(imageMatrixValues);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        if (!zoomEnabled) return false;
        scaleDetector.onTouchEvent(event);
        if (event.getPointerCount() == 1) {
            if (event.getAction() == MotionEvent.ACTION_DOWN) { lastX = event.getX(); lastY = event.getY(); return true; }
            if (event.getAction() == MotionEvent.ACTION_MOVE) {
                imageMatrixValues.postTranslate(event.getX() - lastX, event.getY() - lastY);
                lastX = event.getX(); lastY = event.getY(); setImageMatrix(imageMatrixValues); return true;
            }
        }
        return true;
    }

    public RectF viewRectToBitmapRect(RectF viewRect) {
        Matrix inverse = new Matrix();
        getImageMatrix().invert(inverse);
        RectF mapped = new RectF(viewRect);
        inverse.mapRect(mapped);
        if (getDrawable() != null) {
            mapped.intersect(0, 0, getDrawable().getIntrinsicWidth(), getDrawable().getIntrinsicHeight());
        }
        return mapped;
    }
}
