package com.ai.countlens;

import android.graphics.Bitmap;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class OpenCVDetectionHelper {

    private static final int MIN_TEMPLATE_SIZE = 12;
    private static final int MAX_MATCHES_PER_SCALE = 40;
    private static final double[] TEMPLATE_SCALES = {
            0.55, 0.65, 0.75, 0.85, 1.00, 1.15, 1.30, 1.50, 1.70
    };

    public interface DetectionCallback {
        void onDetectionComplete(Bitmap resultBitmap, int count);
    }

    private static class DetectionCandidate {
        final Rect rect;
        final double score;

        DetectionCandidate(Rect rect, double score) {
            this.rect = rect;
            this.score = score;
        }
    }

    public static void detectSimilarObjects(
            Bitmap sourceBitmap,
            RectF selectionRect,
            double threshold,
            double nmsThreshold,
            DetectionCallback callback
    ) {
        Mat fullMat = new Mat();
        Mat grayFull = new Mat();
        Mat template = new Mat();
        Mat grayTemplate = new Mat();

        try {
            Utils.bitmapToMat(sourceBitmap, fullMat);
            Imgproc.cvtColor(fullMat, grayFull, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.GaussianBlur(grayFull, grayFull, new Size(3, 3), 0);

            Rect roi = toSafeRect(selectionRect, fullMat.cols(), fullMat.rows());
            if (roi.width < MIN_TEMPLATE_SIZE || roi.height < MIN_TEMPLATE_SIZE) {
                callback.onDetectionComplete(sourceBitmap, 0);
                return;
            }

            template = new Mat(grayFull, roi).clone();
            Imgproc.GaussianBlur(template, grayTemplate, new Size(3, 3), 0);

            List<DetectionCandidate> candidates = new ArrayList<>();
            for (double scale : TEMPLATE_SCALES) {
                Mat scaledTemplate = new Mat();
                Mat result = new Mat();
                try {
                    int scaledWidth = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(grayTemplate.cols() * scale));
                    int scaledHeight = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(grayTemplate.rows() * scale));

                    if (scaledWidth >= grayFull.cols() || scaledHeight >= grayFull.rows()) {
                        continue;
                    }

                    Imgproc.resize(grayTemplate, scaledTemplate, new Size(scaledWidth, scaledHeight));
                    int resultCols = grayFull.cols() - scaledTemplate.cols() + 1;
                    int resultRows = grayFull.rows() - scaledTemplate.rows() + 1;
                    if (resultCols <= 0 || resultRows <= 0) {
                        continue;
                    }

                    result.create(resultRows, resultCols, CvType.CV_32FC1);
                    Imgproc.matchTemplate(grayFull, scaledTemplate, result, Imgproc.TM_CCOEFF_NORMED);
                    collectLocalMaxima(result, scaledTemplate.cols(), scaledTemplate.rows(), threshold, candidates);
                } finally {
                    scaledTemplate.release();
                    result.release();
                }
            }

            List<DetectionCandidate> finalDetections = applyNMS(candidates, nmsThreshold);

            Mat output = fullMat.clone();
            int index = 1;
            for (DetectionCandidate candidate : finalDetections) {
                Rect rect = candidate.rect;
                Imgproc.rectangle(output, rect.tl(), rect.br(), new Scalar(34, 197, 94, 255), 4);
                Imgproc.putText(
                        output,
                        String.valueOf(index++),
                        new Point(rect.x + 8, Math.max(24, rect.y + 28)),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        new Scalar(245, 158, 11, 255),
                        3
                );
            }

            Bitmap resultBitmap = Bitmap.createBitmap(output.cols(), output.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(output, resultBitmap);
            output.release();

            callback.onDetectionComplete(resultBitmap, finalDetections.size());
        } catch (Exception e) {
            callback.onDetectionComplete(sourceBitmap, 0);
        } finally {
            fullMat.release();
            grayFull.release();
            template.release();
            grayTemplate.release();
        }
    }

    private static Rect toSafeRect(RectF rectF, int maxWidth, int maxHeight) {
        int left = clamp(Math.round(Math.min(rectF.left, rectF.right)), 0, maxWidth - 1);
        int top = clamp(Math.round(Math.min(rectF.top, rectF.bottom)), 0, maxHeight - 1);
        int right = clamp(Math.round(Math.max(rectF.left, rectF.right)), left + 1, maxWidth);
        int bottom = clamp(Math.round(Math.max(rectF.top, rectF.bottom)), top + 1, maxHeight);
        return new Rect(left, top, right - left, bottom - top);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void collectLocalMaxima(
            Mat matchResult,
            int templateWidth,
            int templateHeight,
            double threshold,
            List<DetectionCandidate> candidates
    ) {
        int collected = 0;
        while (collected < MAX_MATCHES_PER_SCALE) {
            Core.MinMaxLocResult minMax = Core.minMaxLoc(matchResult);
            if (minMax.maxVal < threshold) {
                break;
            }

            int x = (int) Math.round(minMax.maxLoc.x);
            int y = (int) Math.round(minMax.maxLoc.y);
            candidates.add(new DetectionCandidate(new Rect(x, y, templateWidth, templateHeight), minMax.maxVal));
            collected++;

            int suppressWidth = Math.max(6, templateWidth / 2);
            int suppressHeight = Math.max(6, templateHeight / 2);
            Point p1 = new Point(
                    Math.max(0, x - suppressWidth / 2.0),
                    Math.max(0, y - suppressHeight / 2.0)
            );
            Point p2 = new Point(
                    Math.min(matchResult.cols() - 1, x + suppressWidth / 2.0),
                    Math.min(matchResult.rows() - 1, y + suppressHeight / 2.0)
            );
            Imgproc.rectangle(matchResult, p1, p2, new Scalar(0), -1);
        }
    }

    private static List<DetectionCandidate> applyNMS(List<DetectionCandidate> candidates, double overlapThreshold) {
        List<DetectionCandidate> result = new ArrayList<>();
        if (candidates.isEmpty()) return result;

        List<DetectionCandidate> sorted = new ArrayList<>(candidates);
        Collections.sort(sorted, (a, b) -> Double.compare(b.score, a.score));

        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            DetectionCandidate current = sorted.get(i);
            result.add(current);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) continue;
                if (getIntersectionOverUnion(current.rect, sorted.get(j).rect) > overlapThreshold) {
                    removed[j] = true;
                }
            }
        }

        result.sort(Comparator.comparingInt((DetectionCandidate c) -> c.rect.y).thenComparingInt(c -> c.rect.x));
        return result;
    }

    private static double getIntersectionOverUnion(Rect r1, Rect r2) {
        int x1 = Math.max(r1.x, r2.x);
        int y1 = Math.max(r1.y, r2.y);
        int x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
        int y2 = Math.min(r1.y + r1.height, r2.y + r2.height);

        if (x1 >= x2 || y1 >= y2) return 0;

        double intersectionArea = (double) (x2 - x1) * (y2 - y1);
        double unionArea = (double) r1.width * r1.height + (double) r2.width * r2.height - intersectionArea;
        if (unionArea <= 0) return 0;
        return intersectionArea / unionArea;
    }
}
