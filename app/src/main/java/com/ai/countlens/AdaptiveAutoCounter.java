package com.ai.countlens;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.geometry.Geometry;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Reference-free detector for scenes containing repeated objects.
 *
 * <p>There is no universally correct way to count every semantic object without a model or
 * reference example. This pipeline therefore targets the useful offline case: many visually
 * separate objects with a dominant physical scale (coins, pills, fruit, packages, icons, etc.).
 * It fuses adaptive thresholding, automatically tuned Canny edges and circle proposals, then
 * removes duplicate/nested boxes and suppresses scale outliers.</p>
 */
final class AdaptiveAutoCounter {
    private static final String TAG = "AdaptiveAutoCounter";
    private static final int MAX_CANDIDATES = 1200;
    private static final int MAX_DETECTIONS = 900;

    private AdaptiveAutoCounter() {
    }

    static void countRepeatedObjects(
            Bitmap sourceBitmap,
            double sensitivity,
            OpenCVDetectionHelper.DetectionCallback callback
    ) {
        Mat rgba = new Mat();
        Mat gray = new Mat();
        Mat normalized = new Mat();
        Mat blurred = new Mat();
        Mat edges = new Mat();
        Mat edgeMask = new Mat();
        Mat thresholdLight = new Mat();
        Mat thresholdDark = new Mat();
        Mat foregroundMask = new Mat();

        try {
            Utils.bitmapToMat(sourceBitmap, rgba);
            if (rgba.empty() || rgba.cols() < 16 || rgba.rows() < 16) {
                callback.onDetectionComplete(sourceBitmap, 0);
                return;
            }

            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.equalizeHist(gray, normalized);
            Imgproc.GaussianBlur(normalized, blurred, new Size(5, 5), 0.0);

            double strictness = DetectionMath.clamp((sensitivity - 0.45) / 0.45, 0.0, 1.0);
            buildAdaptiveEdgeMask(blurred, edges, edgeMask, strictness);
            buildAdaptiveForegroundMask(normalized, thresholdLight, thresholdDark, foregroundMask, strictness);

            double imageArea = Math.max(1.0, rgba.cols() * (double) rgba.rows());
            int minDimension = Math.min(rgba.cols(), rgba.rows());
            List<Candidate> candidates = new ArrayList<>();

            collectContourCandidates(edgeMask, 0.92, strictness, imageArea, minDimension, candidates);
            collectContourCandidates(foregroundMask, 1.10, strictness, imageArea, minDimension, candidates);
            collectCircleCandidates(blurred, edges, strictness, candidates);

            List<Candidate> unique = suppressDuplicates(candidates);
            List<Candidate> repeatedScale = keepDominantRepeatedScale(unique);
            repeatedScale.sort(Comparator
                    .comparingInt((Candidate candidate) -> candidate.rect.y)
                    .thenComparingInt(candidate -> candidate.rect.x));

            callback.onDetectionComplete(drawDetections(rgba, repeatedScale), repeatedScale.size());
        } catch (RuntimeException error) {
            Log.e(TAG, "Adaptive counting failed", error);
            callback.onDetectionComplete(sourceBitmap, 0);
        } finally {
            rgba.release();
            gray.release();
            normalized.release();
            blurred.release();
            edges.release();
            edgeMask.release();
            thresholdLight.release();
            thresholdDark.release();
            foregroundMask.release();
        }
    }

    private static void buildAdaptiveEdgeMask(Mat gray, Mat edges, Mat edgeMask, double strictness) {
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble deviation = new MatOfDouble();
        Mat kernel = new Mat();
        try {
            Core.meanStdDev(gray, mean, deviation);
            double center = firstOr(mean.toArray(), 110.0);
            double sigma = Math.max(12.0, firstOr(deviation.toArray(), 35.0));
            double low = DetectionMath.clamp(center - sigma * (0.92 - strictness * 0.22), 8.0, 170.0);
            double high = DetectionMath.clamp(center + sigma * (1.15 + strictness * 0.20), low + 20.0, 245.0);
            Imgproc.Canny(gray, edges, low, high, 3, true);

            int kernelSize = oddBetween(Math.min(gray.cols(), gray.rows()) / 180, 3, 9);
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize));
            Imgproc.morphologyEx(edges, edgeMask, Imgproc.MORPH_CLOSE, kernel);
        } finally {
            mean.release();
            deviation.release();
            kernel.release();
        }
    }

    private static void buildAdaptiveForegroundMask(
            Mat gray,
            Mat thresholdLight,
            Mat thresholdDark,
            Mat output,
            double strictness
    ) {
        int minDimension = Math.min(gray.cols(), gray.rows());
        int blockSize = oddBetween(minDimension / 18, 21, 81);
        double constant = 3.0 + strictness * 5.0;

        Imgproc.adaptiveThreshold(
                gray,
                thresholdLight,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY,
                blockSize,
                constant
        );
        Imgproc.adaptiveThreshold(
                gray,
                thresholdDark,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV,
                blockSize,
                constant
        );

        Mat selected = maskQuality(thresholdLight) >= maskQuality(thresholdDark)
                ? thresholdLight
                : thresholdDark;
        selected.copyTo(output);

        Mat openKernel = new Mat();
        Mat closeKernel = new Mat();
        try {
            int openSize = oddBetween(minDimension / 320, 3, 5);
            int closeSize = oddBetween(minDimension / 150, 3, nineOrLess(minDimension));
            openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(openSize, openSize));
            closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(closeSize, closeSize));
            Imgproc.morphologyEx(output, output, Imgproc.MORPH_OPEN, openKernel);
            Imgproc.morphologyEx(output, output, Imgproc.MORPH_CLOSE, closeKernel);
        } finally {
            openKernel.release();
            closeKernel.release();
        }
    }

    private static int nineOrLess(int minDimension) {
        return minDimension >= 900 ? 9 : 7;
    }

    private static double maskQuality(Mat mask) {
        double totalPixels = Math.max(1.0, mask.rows() * (double) mask.cols());
        double occupancy = Core.countNonZero(mask) / totalPixels;
        if (occupancy < 0.002 || occupancy > 0.88) {
            return -10.0;
        }

        int border = Math.max(2, Math.min(mask.rows(), mask.cols()) / 45);
        double borderOccupancy = borderOccupancy(mask, border);
        double occupancyPreference = 1.0 - Math.min(1.0, Math.abs(occupancy - 0.24) / 0.62);
        return occupancyPreference * 1.35 + (1.0 - borderOccupancy) * 2.0;
    }

    private static double borderOccupancy(Mat mask, int border) {
        int rows = mask.rows();
        int cols = mask.cols();
        if (rows <= border * 2 || cols <= border * 2) {
            return 1.0;
        }

        Rect topRect = new Rect(0, 0, cols, border);
        Rect bottomRect = new Rect(0, rows - border, cols, border);
        Rect leftRect = new Rect(0, border, border, rows - border * 2);
        Rect rightRect = new Rect(cols - border, border, border, rows - border * 2);
        long nonZero = countNonZero(mask, topRect)
                + countNonZero(mask, bottomRect)
                + countNonZero(mask, leftRect)
                + countNonZero(mask, rightRect);
        long pixels = (long) cols * border * 2L + (long) (rows - border * 2) * border * 2L;
        return pixels <= 0 ? 1.0 : nonZero / (double) pixels;
    }

    private static long countNonZero(Mat source, Rect rect) {
        Mat region = new Mat(source, rect);
        try {
            return Core.countNonZero(region);
        } finally {
            region.release();
        }
    }

    private static void collectContourCandidates(
            Mat mask,
            double sourceWeight,
            double strictness,
            double imageArea,
            int minDimension,
            List<Candidate> output
    ) {
        Mat contourInput = mask.clone();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        try {
            Imgproc.findContours(
                    contourInput,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            double minBoxArea = Math.max(
                    45.0,
                    imageArea * (0.00010 + strictness * 0.00026)
            );
            double maxBoxArea = imageArea * 0.42;
            int minSide = Math.max(6, minDimension / 240);

            for (MatOfPoint contour : contours) {
                if (output.size() >= MAX_CANDIDATES) {
                    break;
                }
                Rect rect = Geometry.boundingRect(contour);
                double boxArea = rect.width * (double) rect.height;
                if (boxArea < minBoxArea || boxArea > maxBoxArea
                        || rect.width < minSide || rect.height < minSide) {
                    continue;
                }
                if (rect.width > mask.cols() * 0.88 || rect.height > mask.rows() * 0.88) {
                    continue;
                }

                double contourArea = Math.max(0.0, Geometry.contourArea(contour));
                double fill = contourArea / Math.max(1.0, boxArea);
                double aspect = Math.max(rect.width, rect.height)
                        / (double) Math.max(1, Math.min(rect.width, rect.height));
                if (fill < 0.10 || aspect > 9.0) {
                    continue;
                }

                double borderPenalty = touchesImageBorder(rect, mask.cols(), mask.rows()) ? 0.82 : 1.0;
                double score = boxArea * (0.35 + Math.min(1.0, fill)) * sourceWeight * borderPenalty;
                output.add(new Candidate(expand(rect, 2, mask.cols(), mask.rows()), score, boxArea));
            }
        } finally {
            for (MatOfPoint contour : contours) {
                contour.release();
            }
            contourInput.release();
            hierarchy.release();
        }
    }

    private static void collectCircleCandidates(
            Mat gray,
            Mat edges,
            double strictness,
            List<Candidate> output
    ) {
        int minDimension = Math.min(gray.cols(), gray.rows());
        int minRadius = Math.max(5, minDimension / 110);
        int maxRadius = Math.max(minRadius + 2, minDimension / 5);
        Mat circles = new Mat();
        try {
            Imgproc.HoughCircles(
                    gray,
                    circles,
                    Imgproc.HOUGH_GRADIENT,
                    1.25,
                    Math.max(10.0, minDimension / 35.0),
                    110.0 + strictness * 35.0,
                    20.0 + strictness * 12.0,
                    minRadius,
                    maxRadius
            );

            if (circles.empty() || circles.cols() < 3 || circles.cols() > 450) {
                return;
            }

            List<Double> radii = new ArrayList<>();
            for (int index = 0; index < circles.cols(); index++) {
                double[] circle = circles.get(0, index);
                if (circle != null && circle.length >= 3 && circle[2] > 0.0) {
                    radii.add(circle[2]);
                }
            }
            double medianRadius = DetectionMath.median(radii);
            if (medianRadius <= 0.0 || DetectionMath.robustCoefficientOfVariation(radii) > 0.38) {
                return;
            }

            for (int index = 0; index < circles.cols() && output.size() < MAX_CANDIDATES; index++) {
                double[] circle = circles.get(0, index);
                if (circle == null || circle.length < 3) {
                    continue;
                }
                double radius = circle[2];
                if (radius < medianRadius * 0.55 || radius > medianRadius * 1.75) {
                    continue;
                }
                if (circleEdgeSupport(edges, circle[0], circle[1], radius) < 0.14) {
                    continue;
                }

                int left = (int) Math.floor(circle[0] - radius);
                int top = (int) Math.floor(circle[1] - radius);
                int diameter = Math.max(1, (int) Math.ceil(radius * 2.0));
                Rect rect = clampRect(new Rect(left, top, diameter, diameter), gray.cols(), gray.rows());
                double area = Math.PI * radius * radius;
                output.add(new Candidate(rect, area * 1.55, rect.width * (double) rect.height));
            }
        } finally {
            circles.release();
        }
    }

    private static double circleEdgeSupport(Mat edges, double centerX, double centerY, double radius) {
        int samples = 32;
        int hits = 0;
        for (int index = 0; index < samples; index++) {
            double angle = Math.PI * 2.0 * index / samples;
            int x = (int) Math.round(centerX + Math.cos(angle) * radius);
            int y = (int) Math.round(centerY + Math.sin(angle) * radius);
            if (x < 0 || y < 0 || x >= edges.cols() || y >= edges.rows()) {
                continue;
            }
            double[] value = edges.get(y, x);
            if (value != null && value.length > 0 && value[0] > 0.0) {
                hits++;
            }
        }
        return hits / (double) samples;
    }

    private static List<Candidate> suppressDuplicates(List<Candidate> candidates) {
        candidates.sort((left, right) -> Double.compare(right.score, left.score));
        List<Candidate> kept = new ArrayList<>();
        for (Candidate candidate : candidates) {
            boolean duplicate = false;
            for (Candidate existing : kept) {
                double iou = DetectionMath.intersectionOverUnion(
                        candidate.rect.x, candidate.rect.y, candidate.rect.width, candidate.rect.height,
                        existing.rect.x, existing.rect.y, existing.rect.width, existing.rect.height
                );
                double containment = DetectionMath.containmentRatio(
                        candidate.rect.x, candidate.rect.y, candidate.rect.width, candidate.rect.height,
                        existing.rect.x, existing.rect.y, existing.rect.width, existing.rect.height
                );
                if (iou > 0.32 || containment > 0.78 || sameCenter(candidate.rect, existing.rect)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                kept.add(candidate);
                if (kept.size() >= MAX_DETECTIONS) {
                    break;
                }
            }
        }
        return kept;
    }

    private static boolean sameCenter(Rect first, Rect second) {
        double areaRatio = Math.max(
                first.width * (double) first.height,
                second.width * (double) second.height
        ) / Math.max(1.0, Math.min(
                first.width * (double) first.height,
                second.width * (double) second.height
        ));
        if (areaRatio > 2.2) {
            return false;
        }
        double firstX = first.x + first.width / 2.0;
        double firstY = first.y + first.height / 2.0;
        double secondX = second.x + second.width / 2.0;
        double secondY = second.y + second.height / 2.0;
        double distance = Math.hypot(firstX - secondX, firstY - secondY);
        double scale = Math.min(Math.hypot(first.width, first.height), Math.hypot(second.width, second.height));
        return distance < scale * 0.16;
    }

    private static List<Candidate> keepDominantRepeatedScale(List<Candidate> candidates) {
        if (candidates.size() < 4) {
            return candidates;
        }

        Candidate bestAnchor = null;
        int bestSupport = 0;
        double bestSupportScore = 0.0;
        for (Candidate anchor : candidates) {
            int support = 0;
            double supportScore = 0.0;
            for (Candidate candidate : candidates) {
                double ratio = Math.max(anchor.area, candidate.area)
                        / Math.max(1.0, Math.min(anchor.area, candidate.area));
                if (ratio <= 2.8) {
                    support++;
                    supportScore += Math.log1p(candidate.score);
                }
            }
            if (support > bestSupport || (support == bestSupport && supportScore > bestSupportScore)) {
                bestAnchor = anchor;
                bestSupport = support;
                bestSupportScore = supportScore;
            }
        }

        if (bestAnchor == null || bestSupport < 3) {
            return candidates;
        }

        List<Candidate> cluster = new ArrayList<>();
        for (Candidate candidate : candidates) {
            double ratio = Math.max(bestAnchor.area, candidate.area)
                    / Math.max(1.0, Math.min(bestAnchor.area, candidate.area));
            if (ratio <= 2.8) {
                cluster.add(candidate);
            }
        }
        return cluster;
    }

    private static Bitmap drawDetections(Mat source, List<Candidate> detections) {
        Mat output = source.clone();
        try {
            int total = detections.size();
            int thickness = total > 80 ? 2 : total > 30 ? 3 : 4;
            double labelScale = total > 120 ? 0.38 : total > 50 ? 0.50 : 0.78;
            int labelThickness = total > 80 ? 1 : 2;
            boolean drawLabels = total <= 180;

            int index = 1;
            for (Candidate candidate : detections) {
                Rect rect = clampRect(candidate.rect, output.cols(), output.rows());
                Imgproc.rectangle(output, rect.tl(), rect.br(), new Scalar(34, 197, 94, 255), thickness);
                if (drawLabels) {
                    Imgproc.putText(
                            output,
                            String.valueOf(index),
                            new Point(rect.x + 3, Math.max(14, rect.y + 18)),
                            Imgproc.FONT_HERSHEY_SIMPLEX,
                            labelScale,
                            new Scalar(245, 158, 11, 255),
                            labelThickness
                    );
                }
                index++;
            }

            Bitmap bitmap = Bitmap.createBitmap(output.cols(), output.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(output, bitmap);
            return bitmap;
        } finally {
            output.release();
        }
    }

    private static boolean touchesImageBorder(Rect rect, int width, int height) {
        return rect.x <= 1 || rect.y <= 1 || rect.x + rect.width >= width - 1 || rect.y + rect.height >= height - 1;
    }

    private static Rect expand(Rect rect, int padding, int width, int height) {
        return clampRect(new Rect(
                rect.x - padding,
                rect.y - padding,
                rect.width + padding * 2,
                rect.height + padding * 2
        ), width, height);
    }

    private static Rect clampRect(Rect rect, int width, int height) {
        int left = Math.max(0, Math.min(width - 1, rect.x));
        int top = Math.max(0, Math.min(height - 1, rect.y));
        int right = Math.max(left + 1, Math.min(width, rect.x + Math.max(1, rect.width)));
        int bottom = Math.max(top + 1, Math.min(height, rect.y + Math.max(1, rect.height)));
        return new Rect(left, top, right - left, bottom - top);
    }

    private static int oddBetween(int value, int minimum, int maximum) {
        int clamped = Math.max(minimum, Math.min(maximum, value));
        return (clamped & 1) == 0 ? Math.min(maximum, clamped + 1) : clamped;
    }

    private static double firstOr(double[] values, double fallback) {
        return values == null || values.length == 0 || !Double.isFinite(values[0])
                ? fallback
                : values[0];
    }

    private static final class Candidate {
        final Rect rect;
        final double score;
        final double area;

        Candidate(Rect rect, double score, double area) {
            this.rect = rect;
            this.score = score;
            this.area = Math.max(1.0, area);
        }
    }
}
