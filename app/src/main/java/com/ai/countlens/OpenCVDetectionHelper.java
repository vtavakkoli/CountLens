package com.ai.countlens;

import android.graphics.Bitmap;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class OpenCVDetectionHelper {

    public interface DetectionCallback {
        void onDetectionComplete(Bitmap resultBitmap, int count);
    }

    public static void detectSimilarObjects(Bitmap fullBitmap, RectF selectionRect, double threshold, DetectionCallback callback) {
        Mat fullMat = new Mat();
        Utils.bitmapToMat(fullBitmap, fullMat);

        // Convert selectionRect to Mat Rect
        // Need to handle scaling if ImageView size != bitmap size
        // For now assume selectionRect is in bitmap coordinates
        Rect roi = new Rect((int) selectionRect.left, (int) selectionRect.top, 
                             (int) selectionRect.width(), (int) selectionRect.height());
        
        if (roi.width <= 0 || roi.height <= 0) {
            callback.onDetectionComplete(fullBitmap, 0);
            return;
        }

        Mat template = new Mat(fullMat, roi).clone();
        
        Mat grayFull = new Mat();
        Mat grayTemplate = new Mat();
        Imgproc.cvtColor(fullMat, grayFull, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(template, grayTemplate, Imgproc.COLOR_BGR2GRAY);

        // Template matching
        Mat result = new Mat();
        Imgproc.matchTemplate(grayFull, grayTemplate, result, Imgproc.TM_CCOEFF_NORMED);

        // Thresholding results
        List<Rect> detections = new ArrayList<>();
        
        for (int r = 0; r < result.rows(); r++) {
            for (int c = 0; c < result.cols(); c++) {
                double[] val = result.get(r, c);
                if (val[0] >= threshold) {
                    detections.add(new Rect(c, r, grayTemplate.cols(), grayTemplate.rows()));
                }
            }
        }

        // Apply NMS (Non-Maximum Suppression) simplified
        List<Rect> finalDetections = applyNMS(detections, 0.3);

        // Draw results
        for (Rect rect : finalDetections) {
            Imgproc.rectangle(fullMat, rect.tl(), rect.br(), new Scalar(0, 255, 0), 3);
        }

        Bitmap resultBitmap = Bitmap.createBitmap(fullMat.cols(), fullMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(fullMat, resultBitmap);

        callback.onDetectionComplete(resultBitmap, finalDetections.size());
        
        // Cleanup
        fullMat.release();
        template.release();
        grayFull.release();
        grayTemplate.release();
        result.release();
    }

    private static List<Rect> applyNMS(List<Rect> rects, double overlapThreshold) {
        List<Rect> result = new ArrayList<>();
        if (rects.isEmpty()) return result;

        // Sort by confidence (here we don't have it explicitly, so just take first)
        // Simplified NMS: just remove overlapping rectangles
        List<Rect> candidates = new ArrayList<>(rects);
        while (!candidates.isEmpty()) {
            Rect current = candidates.remove(0);
            result.add(current);
            candidates.removeIf(next -> getOverlapRatio(current, next) > overlapThreshold);
        }
        return result;
    }

    private static double getOverlapRatio(Rect r1, Rect r2) {
        int x1 = Math.max(r1.x, r2.x);
        int y1 = Math.max(r1.y, r2.y);
        int x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
        int y2 = Math.min(r1.y + r1.height, r2.y + r2.height);

        if (x1 >= x2 || y1 >= y2) return 0;

        double intersectionArea = (x2 - x1) * (y2 - y1);
        double unionArea = r1.width * r1.height + r2.width * r2.height - intersectionArea;
        return intersectionArea / unionArea;
    }
}