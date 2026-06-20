package com.ai.countlens;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

public class SelectionOverlayView extends View {

    private static final float MIN_SELECTION_SIZE = 16f;

    private Paint borderPaint;
    private Paint fillPaint;
    private Paint handlePaint;
    private Paint guidePaint;
    private RectF selectionRect;
    private float rotationAngle = 0f; // in degrees
    private float startX, startY;
    private float lastTouchX, lastTouchY;
    private boolean isDrawing = false;
    private boolean isRotating = false;
    private boolean isMoving = false;
    private String selectionShape = SettingsManager.SHAPE_RECTANGLE;

    private static final float ROTATION_HANDLE_OFFSET = 60f;
    private static final float TOUCH_SLOP = 40f;

    public SelectionOverlayView(Context context) {
        super(context);
        init();
    }

    public SelectionOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setWillNotDraw(false);

        borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        borderPaint.setColor(Color.rgb(245, 158, 11));
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(5f);

        fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        fillPaint.setColor(Color.argb(42, 245, 158, 11));
        fillPaint.setStyle(Paint.Style.FILL);

        handlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        handlePaint.setColor(Color.WHITE);
        handlePaint.setStyle(Paint.Style.FILL);

        guidePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        guidePaint.setColor(Color.argb(160, 255, 255, 255));
        guidePaint.setStyle(Paint.Style.STROKE);
        guidePaint.setStrokeWidth(2f);

        selectionRect = new RectF();
    }

    public void setSelectionShape(String shape) {
        this.selectionShape = shape;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (!hasSelection()) {
            return;
        }

        canvas.save();
        canvas.rotate(rotationAngle, selectionRect.centerX(), selectionRect.centerY());

        if (SettingsManager.SHAPE_CIRCLE.equals(selectionShape)) {
            float centerX = selectionRect.centerX();
            float centerY = selectionRect.centerY();
            float radius = Math.min(selectionRect.width(), selectionRect.height()) / 2f;
            canvas.drawCircle(centerX, centerY, radius, fillPaint);
            canvas.drawCircle(centerX, centerY, radius, borderPaint);
        } else {
            canvas.drawRoundRect(selectionRect, 8f, 8f, fillPaint);
            canvas.drawRoundRect(selectionRect, 8f, 8f, borderPaint);
        }

        drawGuides(canvas);
        drawHandles(canvas);
        
        canvas.restore();
    }

    private void drawGuides(Canvas canvas) {
        // Vertical and horizontal center lines
        canvas.drawLine(selectionRect.centerX(), selectionRect.top, selectionRect.centerX(), selectionRect.bottom, guidePaint);
        canvas.drawLine(selectionRect.left, selectionRect.centerY(), selectionRect.right, selectionRect.centerY(), guidePaint);
        
        // Rotation handle stem
        canvas.drawLine(selectionRect.centerX(), selectionRect.top, selectionRect.centerX(), selectionRect.top - ROTATION_HANDLE_OFFSET, borderPaint);
    }

    private void drawHandles(Canvas canvas) {
        float radius = 10f;
        // Corner handles
        canvas.drawCircle(selectionRect.left, selectionRect.top, radius, handlePaint);
        canvas.drawCircle(selectionRect.right, selectionRect.top, radius, handlePaint);
        canvas.drawCircle(selectionRect.left, selectionRect.bottom, radius, handlePaint);
        canvas.drawCircle(selectionRect.right, selectionRect.bottom, radius, handlePaint);
        
        // Rotation handle
        canvas.drawCircle(selectionRect.centerX(), selectionRect.top - ROTATION_HANDLE_OFFSET, radius + 4f, borderPaint);
        canvas.drawCircle(selectionRect.centerX(), selectionRect.top - ROTATION_HANDLE_OFFSET, radius, handlePaint);
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!isEnabled()) {
            return false;
        }

        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                if (isNearRotationHandle(x, y)) {
                    isRotating = true;
                } else if (selectionRect.contains(x, y)) {
                    isMoving = true;
                    lastTouchX = x;
                    lastTouchY = y;
                } else {
                    startX = x;
                    startY = y;
                    selectionRect.set(startX, startY, startX, startY);
                    rotationAngle = 0f;
                    isDrawing = true;
                }
                invalidate();
                return true;

            case MotionEvent.ACTION_MOVE:
                if (isRotating) {
                    rotationAngle = calculateRotationAngle(x, y);
                } else if (isMoving) {
                    float dx = x - lastTouchX;
                    float dy = y - lastTouchY;
                    selectionRect.offset(dx, dy);
                    lastTouchX = x;
                    lastTouchY = y;
                } else if (isDrawing) {
                    updateRect(x, y);
                }
                invalidate();
                return true;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                if (!isMoving && !isRotating && !isDrawing) {
                    performClick();
                }
                isDrawing = false;
                isRotating = false;
                isMoving = false;
                invalidate();
                return true;
        }
        return super.onTouchEvent(event);
    }

    private boolean isNearRotationHandle(float x, float y) {
        // We need to check the handle position IN VIEW SPACE (rotated)
        float centerX = selectionRect.centerX();
        float centerY = selectionRect.centerY();
        
        // The handle relative to center BEFORE rotation
        float hx = 0;
        float hy = (selectionRect.top - centerY) - ROTATION_HANDLE_OFFSET;
        
        // Rotate the handle point
        double rad = Math.toRadians(rotationAngle);
        float rotatedHx = (float) (hx * Math.cos(rad) - hy * Math.sin(rad)) + centerX;
        float rotatedHy = (float) (hx * Math.sin(rad) + hy * Math.cos(rad)) + centerY;
        
        float dist = (float) Math.sqrt(Math.pow(x - rotatedHx, 2) + Math.pow(y - rotatedHy, 2));
        return dist < TOUCH_SLOP + 20f;
    }

    private float calculateRotationAngle(float x, float y) {
        float centerX = selectionRect.centerX();
        float centerY = selectionRect.centerY();
        double angleRad = Math.atan2(x - centerX, centerY - y);
        return (float) Math.toDegrees(angleRad);
    }

    public float getRotationAngle() {
        return rotationAngle;
    }

    private void updateRect(float x, float y) {
        selectionRect.set(
                Math.min(startX, x),
                Math.min(startY, y),
                Math.max(startX, x),
                Math.max(startY, y)
        );
    }

    private boolean hasSelection() {
        return isDrawing || (selectionRect.width() >= MIN_SELECTION_SIZE && selectionRect.height() >= MIN_SELECTION_SIZE);
    }

    public RectF getSelectionRect() {
        return new RectF(selectionRect);
    }

    public void reset() {
        selectionRect.set(0, 0, 0, 0);
        rotationAngle = 0f;
        invalidate();
    }
}
