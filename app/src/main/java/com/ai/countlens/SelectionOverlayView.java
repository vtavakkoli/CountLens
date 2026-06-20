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

    private Paint paint;
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
        paint = new Paint();
        paint.setColor(Color.YELLOW);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5f);
        paint.setAntiAlias(true);
        selectionRect = new RectF();
    }

    public void setSelectionShape(String shape) {
        this.selectionShape = shape;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (isDrawing || (selectionRect.width() > 0 && selectionRect.height() > 0)) {
            if (SettingsManager.SHAPE_CIRCLE.equals(selectionShape)) {
                float centerX = selectionRect.centerX();
                float centerY = selectionRect.centerY();
                float radius = Math.min(selectionRect.width(), selectionRect.height()) / 2;
                canvas.drawCircle(centerX, centerY, radius, paint);
            } else {
                canvas.drawRect(selectionRect, paint);
            }
        }
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                startX = x;
                startY = y;
                selectionRect.set(startX, startY, startX, startY);
                isDrawing = true;
                invalidate();
                performClick();
                return true;
            case MotionEvent.ACTION_MOVE:
                selectionRect.set(
                        Math.min(startX, x),
                        Math.min(startY, y),
                        Math.max(startX, x),
                        Math.max(startY, y)
                );
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                isDrawing = false;
                invalidate();
                return true;
        }
        return super.onTouchEvent(event);
    }

    public RectF getSelectionRect() {
        return selectionRect;
    }

    public void reset() {
        selectionRect.set(0, 0, 0, 0);
        invalidate();
    }
}