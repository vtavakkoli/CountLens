package com.ai.countlens;

import android.content.Context;
import android.content.SharedPreferences;

public class SettingsManager {
    private static final String PREF_NAME = "countlens_prefs";
    private static final String KEY_THRESHOLD = "pref_threshold";
    private static final String KEY_NMS_THRESHOLD = "pref_nms_threshold";
    private static final String KEY_SELECTION_SHAPE = "pref_selection_shape";
    private static final String KEY_MAX_IMAGE_SIZE = "pref_max_image_size";

    public static final String SHAPE_RECTANGLE = "rectangle";
    public static final String SHAPE_CIRCLE = "circle";

    private static final float DEFAULT_THRESHOLD = 0.64f;
    private static final float DEFAULT_NMS_THRESHOLD = 0.30f;
    private static final int DEFAULT_MAX_IMAGE_SIZE = 1280;

    private final SharedPreferences prefs;

    public SettingsManager(Context context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    public float getThreshold() {
        return prefs.getFloat(KEY_THRESHOLD, DEFAULT_THRESHOLD);
    }

    public void setThreshold(float threshold) {
        prefs.edit().putFloat(KEY_THRESHOLD, threshold).apply();
    }

    public float getNmsThreshold() {
        return prefs.getFloat(KEY_NMS_THRESHOLD, DEFAULT_NMS_THRESHOLD);
    }

    public void setNmsThreshold(float nmsThreshold) {
        prefs.edit().putFloat(KEY_NMS_THRESHOLD, nmsThreshold).apply();
    }

    public String getSelectionShape() {
        return prefs.getString(KEY_SELECTION_SHAPE, SHAPE_RECTANGLE);
    }

    public void setSelectionShape(String shape) {
        prefs.edit().putString(KEY_SELECTION_SHAPE, shape).apply();
    }

    public int getMaxImageSize() {
        return prefs.getInt(KEY_MAX_IMAGE_SIZE, DEFAULT_MAX_IMAGE_SIZE);
    }

    public void setMaxImageSize(int maxImageSize) {
        int clamped = Math.max(640, Math.min(2048, maxImageSize));
        prefs.edit().putInt(KEY_MAX_IMAGE_SIZE, clamped).apply();
    }
}
