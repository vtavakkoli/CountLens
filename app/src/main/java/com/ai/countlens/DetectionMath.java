package com.ai.countlens;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Pure-Java geometry/statistics helpers used by the counting pipeline. */
final class DetectionMath {
    private DetectionMath() {
    }

    static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    static double intersectionOverUnion(
            double ax, double ay, double aw, double ah,
            double bx, double by, double bw, double bh
    ) {
        double intersection = intersectionArea(ax, ay, aw, ah, bx, by, bw, bh);
        double union = Math.max(0.0, aw) * Math.max(0.0, ah)
                + Math.max(0.0, bw) * Math.max(0.0, bh)
                - intersection;
        return union <= 0.0 ? 0.0 : intersection / union;
    }

    static double containmentRatio(
            double ax, double ay, double aw, double ah,
            double bx, double by, double bw, double bh
    ) {
        double intersection = intersectionArea(ax, ay, aw, ah, bx, by, bw, bh);
        double smallerArea = Math.min(
                Math.max(0.0, aw) * Math.max(0.0, ah),
                Math.max(0.0, bw) * Math.max(0.0, bh)
        );
        return smallerArea <= 0.0 ? 0.0 : intersection / smallerArea;
    }

    static double median(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        List<Double> sorted = new ArrayList<>(values.size());
        for (Double value : values) {
            if (value != null && Double.isFinite(value) && value >= 0.0) {
                sorted.add(value);
            }
        }
        if (sorted.isEmpty()) {
            return 0.0;
        }
        Collections.sort(sorted);
        int middle = sorted.size() / 2;
        if ((sorted.size() & 1) == 1) {
            return sorted.get(middle);
        }
        return (sorted.get(middle - 1) + sorted.get(middle)) / 2.0;
    }

    static double robustCoefficientOfVariation(List<Double> values) {
        if (values == null || values.size() < 2) {
            return 0.0;
        }
        double center = median(values);
        if (center <= 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        List<Double> deviations = new ArrayList<>(values.size());
        for (Double value : values) {
            if (value != null && Double.isFinite(value) && value >= 0.0) {
                deviations.add(Math.abs(value - center));
            }
        }
        return median(deviations) / center;
    }

    private static double intersectionArea(
            double ax, double ay, double aw, double ah,
            double bx, double by, double bw, double bh
    ) {
        double left = Math.max(ax, bx);
        double top = Math.max(ay, by);
        double right = Math.min(ax + Math.max(0.0, aw), bx + Math.max(0.0, bw));
        double bottom = Math.min(ay + Math.max(0.0, ah), by + Math.max(0.0, bh));
        return Math.max(0.0, right - left) * Math.max(0.0, bottom - top);
    }
}
