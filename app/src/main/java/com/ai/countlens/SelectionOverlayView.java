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
    private float startX, startY;
    private boolean isDrawing = false;
    private String selectionShape = SettingsManager.SHAPE_RECTANGLE;

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

        if (SettingsManager.SHAPE_CIRCLE.equals(selectionShape)) {
            float centerX = selectionRect.centerX();
            float centerY = selectionRect.centerY();
            float radius = Math.min(selectionRect.width(), selectionRect.height()) / 2f;
            canvas.drawCircle(centerX, centerY, radius, fillPaint);
            canvas.drawCircle(centerX, centerY, radius, borderPaint);
        } else {
            canvas.drawRoundRect(selectionRect, 12f, 12f, fillPaint);
            canvas.drawRoundRect(selectionRect, 12f, 12f, borderPaint);
        }

        drawGuides(canvas);
        drawHandles(canvas);
    }

    private void drawGuides(Canvas canvas) {
        canvas.drawLine(selectionRect.centerX(), selectionRect.top, selectionRect.centerX(), selectionRect.bottom, guidePaint);
        canvas.drawLine(selectionRect.left, selectionRect.centerY(), selectionRect.right, selectionRect.centerY(), guidePaint);
    }

    private void drawHandles(Canvas canvas) {
        float radius = 7f;
        canvas.drawCircle(selectionRect.left, selectionRect.top, radius, handlePaint);
        canvas.drawCircle(selectionRect.right, selectionRect.top, radius, handlePaint);
        canvas.drawCircle(selectionRect.left, selectionRect.bottom, radius, handlePaint);
        canvas.drawCircle(selectionRect.right, selectionRect.bottom, radius, handlePaint);
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!isEnabled()) {
            return false;
        }

        float x = clamp(event.getX(), 0, getWidth());
        float y = clamp(event.getY(), 0, getHeight());

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                startX = x;
                startY = y;
                selectionRect.set(startX, startY, startX, startY);
                isDrawing = true;
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                updateRect(x, y);
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                updateRect(x, y);
                isDrawing = false;
                performClick();
                invalidate();
                return true;
        }
        return super.onTouchEvent(event);
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

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    public RectF getSelectionRect() {
        return new RectF(selectionRect);
    }

    public void reset() {
        selectionRect.set(0, 0, 0, 0);
        invalidate();
    }
}
