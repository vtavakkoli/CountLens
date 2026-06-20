package com.ai.countlens;

import android.content.Context;
import android.content.SharedPreferences;

public class SettingsManager {
    private static final String PREF_NAME = "countlens_prefs";
    private static final String KEY_THRESHOLD = "pref_threshold";
    private static final String KEY_SELECTION_SHAPE = "pref_selection_shape";

    public static final String SHAPE_RECTANGLE = "rectangle";
    public static final String SHAPE_CIRCLE = "circle";

    private final SharedPreferences prefs;

    public SettingsManager(Context context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    public float getThreshold() {
        return prefs.getFloat(KEY_THRESHOLD, 0.8f);
    }

    public void setThreshold(float threshold) {
        prefs.edit().putFloat(KEY_THRESHOLD, threshold).apply();
    }

    public String getSelectionShape() {
        return prefs.getString(KEY_SELECTION_SHAPE, SHAPE_RECTANGLE);
    }

    public void setSelectionShape(String shape) {
        prefs.edit().putString(KEY_SELECTION_SHAPE, shape).apply();
    }
}