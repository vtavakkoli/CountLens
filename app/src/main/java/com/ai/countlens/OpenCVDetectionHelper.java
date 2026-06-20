package com.ai.countlens;

import android.graphics.Bitmap;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.geometry.Geometry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class OpenCVDetectionHelper {

    private static final int MIN_TEMPLATE_SIZE = 10;
    private static final int MAX_MATCHES_PER_VARIANT = 8;
    private static final double SHAPE_MATCH_THRESHOLD = 0.42;
    private static final double SEGMENTATION_SHAPE_THRESHOLD = 0.55;
    private static final double MIN_CONTOUR_AREA = 24.0;

    // These limits are intentionally wider than the first version. The selected rectangle
    // often contains padding/background, while the real object can be much smaller. CountLens
    // now estimates the real foreground object inside the selection and uses that as reference.
    private static final double MIN_WIDTH_HEIGHT_RATIO = 0.18;
    private static final double MAX_WIDTH_HEIGHT_RATIO = 3.60;
    private static final double MIN_AREA_RATIO = 0.035;
    private static final double MAX_AREA_RATIO = 8.00;

    // Robust object-level NMS constants. This combines IoU, containment, center distance,
    // group-box removal, and foreground-component checks. It is much safer than plain IoU-NMS.
    private static final double GROUP_CONTAINMENT_THRESHOLD = 0.58;
    private static final double CONTAINMENT_NMS_THRESHOLD = 0.62;
    private static final double CENTER_DUPLICATE_DISTANCE_RATIO = 0.55;
    private static final double SIZE_RANKING_WEIGHT = 0.11;
    private static final double FOREGROUND_COMPONENT_SPLIT_RATIO = 0.09;
    private static final double LARGE_BOX_COMPONENT_RATIO = 1.45;

    // Fast fallback only. Color/foreground segmentation is used first for photos.
    private static final double[] TEMPLATE_SCALES = {
            0.45, 0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60
    };

    private static final double[] ROTATION_ANGLES = {
            0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
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

    private static class ReferenceInfo {
        final Rect selectionRect;
        final Rect objectRect;
        final Mat foregroundMask;
        final MatOfPoint contour;
        final double contourArea;
        final double foregroundGray;
        final double backgroundGray;
        final boolean usesColorMask;

        ReferenceInfo(
                Rect selectionRect,
                Rect objectRect,
                Mat foregroundMask,
                MatOfPoint contour,
                double contourArea,
                double foregroundGray,
                double backgroundGray,
                boolean usesColorMask
        ) {
            this.selectionRect = selectionRect;
            this.objectRect = objectRect;
            this.foregroundMask = foregroundMask;
            this.contour = contour;
            this.contourArea = contourArea;
            this.foregroundGray = foregroundGray;
            this.backgroundGray = backgroundGray;
            this.usesColorMask = usesColorMask;
        }

        void release() {
            foregroundMask.release();
            if (contour != null) contour.release();
        }
    }

    public static void detectSimilarObjects(
            Bitmap sourceBitmap,
            RectF selectionRect,
            float initialRotation,
            double threshold,
            double nmsThreshold,
            DetectionCallback callback
    ) {
        Mat fullMat = new Mat();
        Mat grayFull = new Mat();
        Mat grayTemplate = new Mat();
        ReferenceInfo referenceInfo = null;

        try {
            Utils.bitmapToMat(sourceBitmap, fullMat);
            Imgproc.cvtColor(fullMat, grayFull, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.GaussianBlur(grayFull, grayFull, new Size(3, 3), 0);

            Rect selectionRoi = toSafeRect(selectionRect, fullMat.cols(), fullMat.rows());
            if (selectionRoi.width < MIN_TEMPLATE_SIZE || selectionRoi.height < MIN_TEMPLATE_SIZE) {
                callback.onDetectionComplete(sourceBitmap, 0);
                return;
            }

            referenceInfo = buildReferenceInfo(fullMat, grayFull, selectionRoi);
            Rect objectRoi = clampRect(referenceInfo.objectRect, fullMat.cols(), fullMat.rows());
            if (objectRoi.width < MIN_TEMPLATE_SIZE || objectRoi.height < MIN_TEMPLATE_SIZE) {
                objectRoi = selectionRoi;
            }

            grayTemplate = new Mat(grayFull, objectRoi).clone();
            
            // If the user rotated the selection, they are likely indicating the principal
            // axis of the object. We rotate the template back to neutral (0 deg) based
            // on the selection rotation so that addRotatedTemplateCandidates starts
            // from the most likely orientation.
            if (Math.abs(initialRotation) > 0.5f) {
                Mat rotatedTemplate = rotateMat(grayTemplate, -initialRotation, estimateBorderMean(grayTemplate));
                grayTemplate.release();
                grayTemplate = rotatedTemplate;
            }

            List<DetectionCandidate> candidates = new ArrayList<>();

            // 1) Fast path for real photos. It detects the dominant foreground color from
            // the selected object, segments similar color regions in the whole image, and
            // extracts single-object components. This is much faster than scanning hundreds
            // of rotated templates and works well for cupcakes, buttons, pills, fruits, etc.
            addColorBlobTemplateCandidates(referenceInfo, candidates);
            addForegroundComponentCandidates(referenceInfo, candidates);

            // 2) Shape fallback for non-color/high-contrast icons. Skip it when the color
            // path already found objects because Otsu thresholding on real photos creates
            // many background contours and slows down/harms the result.
            if (!referenceInfo.usesColorMask || candidates.size() < 2) {
                addContourShapeCandidates(grayFull, referenceInfo, candidates);
            }

            // 3) Template matching is now a last-resort fallback only. This prevents long
            // waiting times on large natural images. ImageDetectionActivity also resizes
            // photos before processing.
            if (candidates.size() < 2) {
                addRotatedTemplateCandidates(grayFull, grayTemplate, threshold, candidates);
            }

            List<DetectionCandidate> refined = refineCandidatesWithForegroundMask(candidates, referenceInfo, fullMat.cols(), fullMat.rows());
            List<DetectionCandidate> finalDetections = applyObjectAwareNMS(refined, referenceInfo, nmsThreshold);

            Mat output = fullMat.clone();
            int index = 1;
            for (DetectionCandidate candidate : finalDetections) {
                Rect rect = clampRect(candidate.rect, output.cols(), output.rows());
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
            grayTemplate.release();
            if (referenceInfo != null) referenceInfo.release();
        }
    }

    private static ReferenceInfo buildReferenceInfo(Mat fullRgba, Mat grayFull, Rect selectionRoi) {
        Mat localReferenceMask = new Mat();
        Mat foregroundMask = new Mat();
        MatOfPoint referenceContour = null;
        Rect objectRect = selectionRoi;
        double contourArea = Math.max(1.0, selectionRoi.width * (double) selectionRoi.height);
        double foregroundGray = 0.0;
        double backgroundGray = 255.0;
        boolean usesColorMask = false;

        try {
            localReferenceMask = createLocalReferenceMask(fullRgba, grayFull, selectionRoi);
            referenceContour = findBestReferenceContour(localReferenceMask);

            if (referenceContour != null && Geometry.contourArea(referenceContour) >= MIN_CONTOUR_AREA) {
                Rect localObjectRect = Geometry.boundingRect(referenceContour);
                objectRect = new Rect(
                        selectionRoi.x + localObjectRect.x,
                        selectionRoi.y + localObjectRect.y,
                        localObjectRect.width,
                        localObjectRect.height
                );
                objectRect = expandRect(objectRect, 3, fullRgba.cols(), fullRgba.rows());
                contourArea = Math.max(MIN_CONTOUR_AREA, Geometry.contourArea(referenceContour));
            } else {
                if (referenceContour != null) referenceContour.release();
                referenceContour = rectangleContour(new Rect(0, 0, selectionRoi.width, selectionRoi.height));
                contourArea = Math.max(1.0, selectionRoi.width * (double) selectionRoi.height);
            }

            Mat grayRoi = new Mat(grayFull, selectionRoi);
            try {
                foregroundGray = Core.mean(grayRoi, localReferenceMask).val[0];
                backgroundGray = estimateBorderMean(grayRoi);
            } finally {
                grayRoi.release();
            }

            MaskBuildResult maskBuildResult = createFullForegroundMask(fullRgba, grayFull, selectionRoi, localReferenceMask, foregroundGray, backgroundGray);
            foregroundMask = maskBuildResult.mask;
            usesColorMask = maskBuildResult.usesColorMask;
        } catch (Exception ignored) {
            foregroundMask = Mat.zeros(grayFull.rows(), grayFull.cols(), CvType.CV_8UC1);
            Imgproc.rectangle(
                    foregroundMask,
                    new Point(selectionRoi.x, selectionRoi.y),
                    new Point(selectionRoi.x + selectionRoi.width, selectionRoi.y + selectionRoi.height),
                    new Scalar(255),
                    -1
            );
            if (referenceContour != null) referenceContour.release();
            referenceContour = rectangleContour(new Rect(0, 0, selectionRoi.width, selectionRoi.height));
        } finally {
            localReferenceMask.release();
        }

        return new ReferenceInfo(
                selectionRoi,
                objectRect,
                foregroundMask,
                referenceContour,
                contourArea,
                foregroundGray,
                backgroundGray,
                usesColorMask
        );
    }

    private static class MaskBuildResult {
        final Mat mask;
        final boolean usesColorMask;

        MaskBuildResult(Mat mask, boolean usesColorMask) {
            this.mask = mask;
            this.usesColorMask = usesColorMask;
        }
    }

    private static class DominantColor {
        final boolean valid;
        final double hue;
        final double saturation;
        final double value;

        DominantColor(boolean valid, double hue, double saturation, double value) {
            this.valid = valid;
            this.hue = hue;
            this.saturation = saturation;
            this.value = value;
        }
    }

    private static Mat createLocalReferenceMask(Mat fullRgba, Mat grayFull, Rect selectionRoi) {
        Mat grayRoi = new Mat(grayFull, selectionRoi);
        Mat diff = new Mat();
        Mat mask = new Mat();
        try {
            double bg = estimateBorderMean(grayRoi);
            Mat bgMat = new Mat(grayRoi.size(), grayRoi.type(), new Scalar(bg));
            Core.absdiff(grayRoi, bgMat, diff);
            bgMat.release();
            Imgproc.threshold(diff, mask, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

            // If Otsu creates an empty/too-small mask, use a conservative fixed contrast mask.
            double nonZero = Core.countNonZero(mask);
            double minPixels = Math.max(MIN_CONTOUR_AREA, grayRoi.rows() * grayRoi.cols() * 0.015);
            if (nonZero < minPixels) {
                Imgproc.threshold(diff, mask, 18, 255, Imgproc.THRESH_BINARY);
            }

            // Add saturated colored pixels from the selection. This is important for photos:
            // red ladybug icing, blue cupcake tops, yellow animals, etc. may not be very
            // different in grayscale but are very clear in HSV color space.
            Mat rgbaRoi = new Mat(fullRgba, selectionRoi);
            Mat rgbRoi = new Mat();
            Mat hsvRoi = new Mat();
            Mat saturatedMask = new Mat();
            try {
                Imgproc.cvtColor(rgbaRoi, rgbRoi, Imgproc.COLOR_RGBA2RGB);
                Imgproc.cvtColor(rgbRoi, hsvRoi, Imgproc.COLOR_RGB2HSV);
                Core.inRange(hsvRoi, new Scalar(0, 45, 35), new Scalar(180, 255, 255), saturatedMask);
                Core.bitwise_or(mask, saturatedMask, mask);
            } finally {
                rgbaRoi.release();
                rgbRoi.release();
                hsvRoi.release();
                saturatedMask.release();
            }

            Mat kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
            int closeSize = Math.max(7, Math.min(25, (Math.min(selectionRoi.width, selectionRoi.height) / 8) | 1));
            Mat kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(closeSize, closeSize));

            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernelOpen);
            Imgproc.dilate(mask, mask, kernelOpen); // Extra dilation to bridge specular highlights
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernelClose);
            kernelOpen.release();
            kernelClose.release();

            if (Core.countNonZero(mask) < minPixels) {
                mask.release();
                mask = Mat.ones(grayRoi.rows(), grayRoi.cols(), CvType.CV_8UC1);
                Core.multiply(mask, new Scalar(255), mask);
            }
            return mask;
        } finally {
            grayRoi.release();
            diff.release();
        }
    }

    private static MaskBuildResult createFullForegroundMask(
            Mat fullRgba,
            Mat grayFull,
            Rect selectionRoi,
            Mat localReferenceMask,
            double foregroundGray,
            double backgroundGray
    ) {
        Mat hsvFull = new Mat();
        Mat hsvRoi = new Mat();
        Mat colorMask = new Mat();
        Mat contrastMask = new Mat();
        Mat finalMask = new Mat();
        boolean useColorMask = false;

        try {
            Mat rgbFull = new Mat();
            Imgproc.cvtColor(fullRgba, rgbFull, Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(rgbFull, hsvFull, Imgproc.COLOR_RGB2HSV);
            rgbFull.release();

            hsvRoi = new Mat(hsvFull, selectionRoi);
            DominantColor dominantColor = estimateDominantColor(hsvRoi, localReferenceMask);
            double hue = dominantColor.hue;
            double saturation = dominantColor.saturation;
            double value = dominantColor.value;

            // Saturated colored objects such as red ladybugs, blue cupcake tops, pills,
            // fruits, components, etc. should be detected by color first. A dominant hue
            // histogram is used instead of simple mean HSV because red wraps around 0/180.
            if (dominantColor.valid && saturation >= 38.0 && value >= 30.0) {
                colorMask = createHueMask(hsvFull, hue, saturation, value);
                int closeSize = Math.max(7, Math.min(15, ((Math.min(selectionRoi.width, selectionRoi.height) / 12) | 1)));
                Mat kernelCloseColor = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(closeSize, closeSize));
                Mat kernelOpenColor = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
                Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_OPEN, kernelOpenColor);
                Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_CLOSE, kernelCloseColor);
                kernelCloseColor.release();
                kernelOpenColor.release();
                useColorMask = Core.countNonZero(colorMask) > MIN_CONTOUR_AREA;
            }

            double mid = (foregroundGray + backgroundGray) / 2.0;
            if (foregroundGray < backgroundGray) {
                Imgproc.threshold(grayFull, contrastMask, mid, 255, Imgproc.THRESH_BINARY_INV);
            } else {
                Imgproc.threshold(grayFull, contrastMask, mid, 255, Imgproc.THRESH_BINARY);
            }

            if (useColorMask) {
                finalMask = colorMask.clone();
            } else {
                finalMask = contrastMask.clone();
            }

            Mat kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
            int closeSize = Math.max(9, Math.min(31, (Math.min(selectionRoi.width, selectionRoi.height) / 7) | 1));
            Mat kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(closeSize, closeSize));

            Imgproc.morphologyEx(finalMask, finalMask, Imgproc.MORPH_OPEN, kernelOpen);
            Imgproc.morphologyEx(finalMask, finalMask, Imgproc.MORPH_CLOSE, kernelClose);
            kernelOpen.release();
            kernelClose.release();

            return new MaskBuildResult(finalMask, useColorMask);
        } finally {
            hsvFull.release();
            hsvRoi.release();
            colorMask.release();
            contrastMask.release();
        }
    }

    private static DominantColor estimateDominantColor(Mat hsvRoi, Mat localMask) {
        int[] hist = new int[180];
        double[] satSum = new double[180];
        double[] valSum = new double[180];
        int total = 0;

        for (int y = 0; y < hsvRoi.rows(); y++) {
            for (int x = 0; x < hsvRoi.cols(); x++) {
                double[] maskValue = localMask.get(y, x);
                if (maskValue == null || maskValue.length == 0 || maskValue[0] <= 0) continue;
                double[] hsv = hsvRoi.get(y, x);
                if (hsv == null || hsv.length < 3) continue;
                double sat = hsv[1];
                double val = hsv[2];
                if (sat < 35.0 || val < 28.0) continue;
                int h = clamp((int) Math.round(hsv[0]), 0, 179);
                hist[h]++;
                satSum[h] += sat;
                valSum[h] += val;
                total++;
            }
        }

        if (total < 20) {
            return new DominantColor(false, 0.0, 0.0, 0.0);
        }

        int bestHue = 0;
        int bestCount = -1;
        for (int h = 0; h < 180; h++) {
            int smoothed = 0;
            for (int offset = -4; offset <= 4; offset++) {
                int idx = (h + offset + 180) % 180;
                smoothed += hist[idx];
            }
            if (smoothed > bestCount) {
                bestCount = smoothed;
                bestHue = h;
            }
        }

        double sumSin = 0.0;
        double sumCos = 0.0;
        double sumSat = 0.0;
        double sumVal = 0.0;
        int count = 0;
        for (int h = 0; h < 180; h++) {
            int dist = Math.abs(h - bestHue);
            dist = Math.min(dist, 180 - dist);
            if (dist <= 12 && hist[h] > 0) {
                double angle = (2.0 * Math.PI * h) / 180.0;
                sumCos += Math.cos(angle) * hist[h];
                sumSin += Math.sin(angle) * hist[h];
                sumSat += satSum[h];
                sumVal += valSum[h];
                count += hist[h];
            }
        }

        if (count < 10) {
            return new DominantColor(false, 0.0, 0.0, 0.0);
        }

        double hue = (Math.atan2(sumSin, sumCos) * 180.0) / (2.0 * Math.PI);
        if (hue < 0) hue += 180.0;
        return new DominantColor(true, hue, sumSat / count, sumVal / count);
    }

    private static Mat createHueMask(Mat hsvFull, double hue, double saturation, double value) {
        int hueTolerance = saturation > 105 ? 18 : 28;
        double minSat = Math.max(68.0, saturation * 0.58);
        double minValue = Math.max(32.0, value * 0.24);
        Mat mask = new Mat();

        double lowHue = hue - hueTolerance;
        double highHue = hue + hueTolerance;
        if (lowHue < 0) {
            Mat maskA = new Mat();
            Mat maskB = new Mat();
            Core.inRange(hsvFull, new Scalar(0, minSat, minValue), new Scalar(highHue, 255, 255), maskA);
            Core.inRange(hsvFull, new Scalar(180 + lowHue, minSat, minValue), new Scalar(180, 255, 255), maskB);
            Core.bitwise_or(maskA, maskB, mask);
            maskA.release();
            maskB.release();
        } else if (highHue > 180) {
            Mat maskA = new Mat();
            Mat maskB = new Mat();
            Core.inRange(hsvFull, new Scalar(lowHue, minSat, minValue), new Scalar(180, 255, 255), maskA);
            Core.inRange(hsvFull, new Scalar(0, minSat, minValue), new Scalar(highHue - 180, 255, 255), maskB);
            Core.bitwise_or(maskA, maskB, mask);
            maskA.release();
            maskB.release();
        } else {
            Core.inRange(hsvFull, new Scalar(lowHue, minSat, minValue), new Scalar(highHue, 255, 255), mask);
        }
        return mask;
    }

    private static void addColorBlobTemplateCandidates(ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        if (!referenceInfo.usesColorMask) return;

        Rect templateRect = clampRect(referenceInfo.objectRect, referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
        if (templateRect.width < MIN_TEMPLATE_SIZE || templateRect.height < MIN_TEMPLATE_SIZE) return;

        Mat referenceTemplate = new Mat(referenceInfo.foregroundMask, templateRect).clone();
        try {
            double foregroundPixels = Core.countNonZero(referenceTemplate);
            double templatePixels = Math.max(1.0, referenceTemplate.cols() * (double) referenceTemplate.rows());
            double fillRatio = foregroundPixels / templatePixels;
            if (fillRatio < 0.08 || fillRatio > 0.92) return;

            double[] colorScales = {0.45, 0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60};
            for (double scale : colorScales) {
                Mat scaledTemplate = new Mat();
                Mat result = new Mat();
                try {
                    int scaledWidth = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(referenceTemplate.cols() * scale));
                    int scaledHeight = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(referenceTemplate.rows() * scale));
                    if (scaledWidth >= referenceInfo.foregroundMask.cols() || scaledHeight >= referenceInfo.foregroundMask.rows()) {
                        continue;
                    }

                    Imgproc.resize(referenceTemplate, scaledTemplate, new Size(scaledWidth, scaledHeight), 0, 0, Imgproc.INTER_NEAREST);
                    Imgproc.threshold(scaledTemplate, scaledTemplate, 127, 255, Imgproc.THRESH_BINARY);
                    if (Core.countNonZero(scaledTemplate) < MIN_CONTOUR_AREA) continue;

                    int resultCols = referenceInfo.foregroundMask.cols() - scaledTemplate.cols() + 1;
                    int resultRows = referenceInfo.foregroundMask.rows() - scaledTemplate.rows() + 1;
                    if (resultCols <= 0 || resultRows <= 0) continue;

                    result.create(resultRows, resultCols, CvType.CV_32FC1);
                    Imgproc.matchTemplate(referenceInfo.foregroundMask, scaledTemplate, result, Imgproc.TM_CCORR_NORMED);
                    collectLocalMaxima(result, scaledTemplate.cols(), scaledTemplate.rows(), 0.46, candidates);
                } finally {
                    scaledTemplate.release();
                    result.release();
                }
            }
        } catch (Exception ignored) {
        } finally {
            referenceTemplate.release();
        }
    }

    private static void addForegroundComponentCandidates(ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat maskCopy = referenceInfo.foregroundMask.clone();
        try {
            Imgproc.findContours(maskCopy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                double area = Geometry.contourArea(contour);
                double minArea = referenceInfo.usesColorMask
                        ? Math.max(MIN_CONTOUR_AREA, referenceInfo.contourArea * 0.10)
                        : Math.max(MIN_CONTOUR_AREA, referenceInfo.contourArea * 0.06);
                if (area < minArea) {
                    contour.release();
                    continue;
                }

                Rect rect = Geometry.boundingRect(contour);
                int padding = referenceInfo.usesColorMask
                        ? Math.max(3, (int) Math.round(Math.min(rect.width, rect.height) * 0.06))
                        : 2;
                Rect candidateRect = expandRect(rect, padding, referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
                if (candidateRect.width < MIN_TEMPLATE_SIZE || candidateRect.height < MIN_TEMPLATE_SIZE || isUnreasonableSize(candidateRect, referenceInfo.objectRect)) {
                    contour.release();
                    continue;
                }

                double shapeScore = Geometry.matchShapes(referenceInfo.contour, contour, Imgproc.CONTOURS_MATCH_I3, 0);
                double areaRatio = area / Math.max(1.0, referenceInfo.contourArea);
                boolean plausibleByShape = referenceInfo.usesColorMask
                        ? shapeScore <= 1.65
                        : shapeScore <= SEGMENTATION_SHAPE_THRESHOLD;
                boolean plausibleBySize = referenceInfo.usesColorMask
                        ? areaRatio >= 0.10 && areaRatio <= 5.50
                        : areaRatio >= MIN_AREA_RATIO && areaRatio <= MAX_AREA_RATIO;

                if (plausibleByShape && plausibleBySize) {
                    double score = referenceInfo.usesColorMask
                            ? 1.72 - 0.04 * Math.abs(Math.log(Math.max(0.05, areaRatio))) - Math.min(0.35, shapeScore * 0.08)
                            : 1.55 - Math.min(1.25, shapeScore) - 0.08 * Math.abs(Math.log(Math.max(0.05, areaRatio)));
                    candidates.add(new DetectionCandidate(candidateRect, score));
                }
                contour.release();
            }
        } catch (Exception ignored) {
        } finally {
            maskCopy.release();
            hierarchy.release();
        }
    }

    private static void addRotatedTemplateCandidates(
            Mat grayFull,
            Mat grayTemplate,
            double threshold,
            List<DetectionCandidate> candidates
    ) {
        double borderValue = estimateBorderMean(grayTemplate);
        double rotatedThreshold = Math.max(0.55, threshold - 0.14);

        for (int mirrorMode = 0; mirrorMode < 2; mirrorMode++) {
            Mat mirrorTemplate = new Mat();
            try {
                if (mirrorMode == 0) {
                    mirrorTemplate = grayTemplate.clone();
                } else {
                    Core.flip(grayTemplate, mirrorTemplate, 1); // horizontal mirror for reversed objects
                }

                for (double angle : ROTATION_ANGLES) {
                    Mat rotatedTemplate = new Mat();
                    try {
                        rotatedTemplate = rotateMat(mirrorTemplate, angle, borderValue);
                        if (rotatedTemplate.cols() < MIN_TEMPLATE_SIZE || rotatedTemplate.rows() < MIN_TEMPLATE_SIZE) {
                            continue;
                        }

                        for (double scale : TEMPLATE_SCALES) {
                            Mat scaledTemplate = new Mat();
                            Mat result = new Mat();
                            try {
                                int scaledWidth = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(rotatedTemplate.cols() * scale));
                                int scaledHeight = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(rotatedTemplate.rows() * scale));

                                if (scaledWidth >= grayFull.cols() || scaledHeight >= grayFull.rows()) {
                                    continue;
                                }

                                Imgproc.resize(rotatedTemplate, scaledTemplate, new Size(scaledWidth, scaledHeight));

                                int resultCols = grayFull.cols() - scaledTemplate.cols() + 1;
                                int resultRows = grayFull.rows() - scaledTemplate.rows() + 1;
                                if (resultCols <= 0 || resultRows <= 0) {
                                    continue;
                                }

                                result.create(resultRows, resultCols, CvType.CV_32FC1);
                                Imgproc.matchTemplate(grayFull, scaledTemplate, result, Imgproc.TM_CCOEFF_NORMED);

                                double activeThreshold = (angle == 0 && mirrorMode == 0) ? Math.max(0.58, threshold - 0.04) : rotatedThreshold;
                                collectLocalMaxima(result, scaledTemplate.cols(), scaledTemplate.rows(), activeThreshold, candidates);
                            } finally {
                                scaledTemplate.release();
                                result.release();
                            }
                        }
                    } finally {
                        rotatedTemplate.release();
                    }
                }
            } finally {
                mirrorTemplate.release();
            }
        }
    }

    private static void addContourShapeCandidates(Mat grayFull, ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        if (referenceInfo.usesColorMask && candidates.size() >= 2) return;
        Mat binary = new Mat();
        try {
            boolean darkObject = referenceInfo.foregroundGray < referenceInfo.backgroundGray;
            int thresholdType = darkObject ? Imgproc.THRESH_BINARY_INV : Imgproc.THRESH_BINARY;
            Imgproc.threshold(grayFull, binary, 0, 255, thresholdType + Imgproc.THRESH_OTSU);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel);
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
            kernel.release();

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Mat binaryCopy = binary.clone();
            Imgproc.findContours(binaryCopy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            binaryCopy.release();
            hierarchy.release();

            for (MatOfPoint contour : contours) {
                double area = Geometry.contourArea(contour);
                if (area < Math.max(MIN_CONTOUR_AREA, referenceInfo.contourArea * 0.05)) {
                    contour.release();
                    continue;
                }

                double areaRatio = area / Math.max(1.0, referenceInfo.contourArea);
                if (areaRatio < MIN_AREA_RATIO || areaRatio > MAX_AREA_RATIO) {
                    contour.release();
                    continue;
                }

                Rect rect = Geometry.boundingRect(contour);
                if (rect.width < MIN_TEMPLATE_SIZE || rect.height < MIN_TEMPLATE_SIZE
                        || isUnreasonableSize(rect, referenceInfo.objectRect)) {
                    contour.release();
                    continue;
                }

                double shapeScore = Geometry.matchShapes(referenceInfo.contour, contour, Imgproc.CONTOURS_MATCH_I3, 0);
                if (shapeScore <= SHAPE_MATCH_THRESHOLD) {
                    double candidateScore = 1.25 - Math.min(1.0, shapeScore);
                    candidates.add(new DetectionCandidate(rect, candidateScore));
                }
                contour.release();
            }
        } catch (Exception ignored) {
            // Shape matching is a fallback. If it cannot be applied to a photo, template matching still runs.
        } finally {
            binary.release();
        }
    }

    private static List<DetectionCandidate> refineCandidatesWithForegroundMask(
            List<DetectionCandidate> candidates,
            ReferenceInfo referenceInfo,
            int maxWidth,
            int maxHeight
    ) {
        List<DetectionCandidate> refined = new ArrayList<>();
        if (candidates.isEmpty()) return refined;

        double minComponentArea = Math.max(MIN_CONTOUR_AREA, referenceInfo.contourArea * FOREGROUND_COMPONENT_SPLIT_RATIO);
        for (DetectionCandidate candidate : candidates) {
            Rect safeRect = clampRect(candidate.rect, maxWidth, maxHeight);
            if (safeRect.width < MIN_TEMPLATE_SIZE || safeRect.height < MIN_TEMPLATE_SIZE) continue;

            List<DetectionCandidate> components = extractValidComponentsInside(safeRect, candidate.score, referenceInfo, minComponentArea);
            if (components.isEmpty()) {
                // Keep high-quality template candidates only when they are not obviously group-sized.
                if (!isUnreasonableSize(safeRect, referenceInfo.objectRect)
                        && candidate.score >= 0.64
                        && getArea(safeRect) <= getArea(referenceInfo.objectRect) * MAX_AREA_RATIO) {
                    refined.add(candidate);
                }
            } else if (components.size() == 1) {
                DetectionCandidate component = components.get(0);
                // Prefer the foreground component. It removes padding and stops the detector from
                // drawing one large box around the full card/background.
                refined.add(component);
            } else {
                // A large template box contains several foreground objects: split it into object boxes.
                refined.addAll(components);
            }
        }

        // Add all foreground components one more time. This protects small objects that had no
        // good template response, such as very small or mirrored hearts.
        addForegroundComponentCandidates(referenceInfo, refined);
        return refined;
    }

    private static List<DetectionCandidate> extractValidComponentsInside(
            Rect candidateRect,
            double baseScore,
            ReferenceInfo referenceInfo,
            double minComponentArea
    ) {
        List<DetectionCandidate> components = new ArrayList<>();
        Rect safeRect = clampRect(candidateRect, referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
        Mat roiMask = new Mat(referenceInfo.foregroundMask, safeRect);
        Mat maskCopy = roiMask.clone();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        try {
            Imgproc.findContours(maskCopy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                double area = Geometry.contourArea(contour);
                if (area < minComponentArea) {
                    contour.release();
                    continue;
                }

                Rect local = Geometry.boundingRect(contour);
                Rect global = new Rect(safeRect.x + local.x, safeRect.y + local.y, local.width, local.height);
                if (global.width < MIN_TEMPLATE_SIZE || global.height < MIN_TEMPLATE_SIZE || isUnreasonableSize(global, referenceInfo.objectRect)) {
                    contour.release();
                    continue;
                }

                double shapeScore = Geometry.matchShapes(referenceInfo.contour, contour, Imgproc.CONTOURS_MATCH_I3, 0);
                double areaRatio = area / Math.max(1.0, referenceInfo.contourArea);
                boolean plausible;
                if (referenceInfo.usesColorMask) {
                    plausible = shapeScore <= 1.65 && areaRatio >= 0.10 && areaRatio <= 5.50;
                } else {
                    plausible = shapeScore <= SEGMENTATION_SHAPE_THRESHOLD
                            && areaRatio >= MIN_AREA_RATIO
                            && areaRatio <= MAX_AREA_RATIO;
                }

                if (plausible) {
                    double score = baseScore + 0.30 - Math.min(0.35, shapeScore * 0.35);
                    components.add(new DetectionCandidate(global, score));
                }
                contour.release();
            }
        } catch (Exception ignored) {
        } finally {
            roiMask.release();
            maskCopy.release();
            hierarchy.release();
        }

        return components;
    }

    private static MatOfPoint findBestReferenceContour(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat maskCopy = mask.clone();
        Imgproc.findContours(maskCopy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        maskCopy.release();

        MatOfPoint best = null;
        double bestScore = 0.0;
        double centerX = mask.cols() / 2.0;
        double centerY = mask.rows() / 2.0;
        double maxDistance = Math.sqrt(centerX * centerX + centerY * centerY) + 1e-6;

        for (MatOfPoint contour : contours) {
            double area = Geometry.contourArea(contour);
            if (area < MIN_CONTOUR_AREA) {
                contour.release();
                continue;
            }
            Rect rect = Geometry.boundingRect(contour);
            double contourCenterX = rect.x + rect.width / 2.0;
            double contourCenterY = rect.y + rect.height / 2.0;
            double distance = Math.sqrt(Math.pow(contourCenterX - centerX, 2) + Math.pow(contourCenterY - centerY, 2));
            double centerWeight = 1.0 - Math.min(1.0, distance / maxDistance);
            double score = area * (1.0 + 0.35 * centerWeight);

            if (score > bestScore) {
                if (best != null) best.release();
                best = contour;
                bestScore = score;
            } else {
                contour.release();
            }
        }
        return best;
    }

    private static MatOfPoint rectangleContour(Rect rect) {
        MatOfPoint contour = new MatOfPoint();
        contour.fromArray(
                new Point(rect.x, rect.y),
                new Point(rect.x + rect.width, rect.y),
                new Point(rect.x + rect.width, rect.y + rect.height),
                new Point(rect.x, rect.y + rect.height)
        );
        return contour;
    }

    private static Mat rotateMat(Mat src, double angle, double borderValue) {
        double normalized = ((angle % 360) + 360) % 360;
        if (normalized == 0) {
            return src.clone();
        }

        Point center = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rotationMatrix = Geometry.getRotationMatrix2D(center, angle, 1.0);

        double cos = Math.abs(rotationMatrix.get(0, 0)[0]);
        double sin = Math.abs(rotationMatrix.get(0, 1)[0]);
        int newWidth = Math.max(1, (int) Math.round(src.rows() * sin + src.cols() * cos));
        int newHeight = Math.max(1, (int) Math.round(src.rows() * cos + src.cols() * sin));

        double[] tx = rotationMatrix.get(0, 2);
        double[] ty = rotationMatrix.get(1, 2);
        tx[0] += newWidth / 2.0 - center.x;
        ty[0] += newHeight / 2.0 - center.y;
        rotationMatrix.put(0, 2, tx);
        rotationMatrix.put(1, 2, ty);

        Mat rotated = new Mat();
        Imgproc.warpAffine(
                src,
                rotated,
                rotationMatrix,
                new Size(newWidth, newHeight),
                Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,
                new Scalar(borderValue)
        );
        rotationMatrix.release();
        return rotated;
    }

    private static double estimateBorderMean(Mat mat) {
        if (mat.empty()) return 255.0;
        int border = Math.max(1, Math.min(mat.cols(), mat.rows()) / 8);
        Mat top = new Mat(mat, new Rect(0, 0, mat.cols(), border));
        Mat bottom = new Mat(mat, new Rect(0, mat.rows() - border, mat.cols(), border));
        Mat left = new Mat(mat, new Rect(0, 0, border, mat.rows()));
        Mat right = new Mat(mat, new Rect(mat.cols() - border, 0, border, mat.rows()));
        double mean = (Core.mean(top).val[0] + Core.mean(bottom).val[0] + Core.mean(left).val[0] + Core.mean(right).val[0]) / 4.0;
        top.release();
        bottom.release();
        left.release();
        right.release();
        return mean;
    }

    private static Rect toSafeRect(RectF rectF, int maxWidth, int maxHeight) {
        int left = clamp(Math.round(Math.min(rectF.left, rectF.right)), 0, maxWidth - 1);
        int top = clamp(Math.round(Math.min(rectF.top, rectF.bottom)), 0, maxHeight - 1);
        int right = clamp(Math.round(Math.max(rectF.left, rectF.right)), left + 1, maxWidth);
        int bottom = clamp(Math.round(Math.max(rectF.top, rectF.bottom)), top + 1, maxHeight);
        return new Rect(left, top, right - left, bottom - top);
    }

    private static Rect clampRect(Rect rect, int maxWidth, int maxHeight) {
        int left = clamp(rect.x, 0, maxWidth - 1);
        int top = clamp(rect.y, 0, maxHeight - 1);
        int right = clamp(rect.x + rect.width, left + 1, maxWidth);
        int bottom = clamp(rect.y + rect.height, top + 1, maxHeight);
        return new Rect(left, top, right - left, bottom - top);
    }

    private static Rect expandRect(Rect rect, int padding, int maxWidth, int maxHeight) {
        return clampRect(
                new Rect(rect.x - padding, rect.y - padding, rect.width + padding * 2, rect.height + padding * 2),
                maxWidth,
                maxHeight
        );
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
        while (collected < MAX_MATCHES_PER_VARIANT) {
            Core.MinMaxLocResult minMax = Core.minMaxLoc(matchResult);
            if (minMax.maxVal < threshold) {
                break;
            }

            int x = (int) Math.round(minMax.maxLoc.x);
            int y = (int) Math.round(minMax.maxLoc.y);
            candidates.add(new DetectionCandidate(new Rect(x, y, templateWidth, templateHeight), minMax.maxVal));
            collected++;

            int suppressWidth = Math.max(6, (int) Math.round(templateWidth * 0.65));
            int suppressHeight = Math.max(6, (int) Math.round(templateHeight * 0.65));
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

    private static List<DetectionCandidate> applyObjectAwareNMS(
            List<DetectionCandidate> candidates,
            ReferenceInfo referenceInfo,
            double overlapThreshold
    ) {
        List<DetectionCandidate> result = new ArrayList<>();
        if (candidates.isEmpty()) return result;

        List<DetectionCandidate> filtered = new ArrayList<>();
        for (DetectionCandidate candidate : candidates) {
            if (candidate.rect.width < MIN_TEMPLATE_SIZE || candidate.rect.height < MIN_TEMPLATE_SIZE) continue;
            if (isUnreasonableSize(candidate.rect, referenceInfo.objectRect)) continue;
            filtered.add(candidate);
        }
        if (filtered.isEmpty()) return result;

        filtered = removeGroupBoxes(filtered, referenceInfo);
        if (filtered.isEmpty()) return result;

        List<DetectionCandidate> sorted = new ArrayList<>(filtered);
        Collections.sort(sorted, (a, b) -> Double.compare(
                getObjectRankScore(b, referenceInfo.objectRect),
                getObjectRankScore(a, referenceInfo.objectRect)
        ));

        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            DetectionCandidate current = sorted.get(i);
            result.add(current);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) continue;
                DetectionCandidate other = sorted.get(j);
                if (shouldSuppress(current.rect, other.rect, overlapThreshold, referenceInfo.objectRect)) {
                    removed[j] = true;
                }
            }
        }

        result.sort(Comparator.comparingInt((DetectionCandidate c) -> c.rect.y).thenComparingInt(c -> c.rect.x));
        return result;
    }

    private static boolean isUnreasonableSize(Rect rect, Rect referenceRect) {
        double refWidth = Math.max(1.0, referenceRect.width);
        double refHeight = Math.max(1.0, referenceRect.height);
        double refArea = Math.max(1.0, referenceRect.width * (double) referenceRect.height);

        double widthRatio = rect.width / refWidth;
        double heightRatio = rect.height / refHeight;
        double areaRatio = (rect.width * (double) rect.height) / refArea;

        return widthRatio < MIN_WIDTH_HEIGHT_RATIO
                || heightRatio < MIN_WIDTH_HEIGHT_RATIO
                || widthRatio > MAX_WIDTH_HEIGHT_RATIO
                || heightRatio > MAX_WIDTH_HEIGHT_RATIO
                || areaRatio < MIN_AREA_RATIO
                || areaRatio > MAX_AREA_RATIO;
    }

    private static List<DetectionCandidate> removeGroupBoxes(List<DetectionCandidate> candidates, ReferenceInfo referenceInfo) {
        List<DetectionCandidate> cleaned = new ArrayList<>();

        for (int i = 0; i < candidates.size(); i++) {
            DetectionCandidate box = candidates.get(i);
            double boxArea = getArea(box.rect);
            int containedSmallerBoxes = 0;
            int foregroundComponents = countForegroundComponents(box.rect, referenceInfo, referenceInfo.contourArea * FOREGROUND_COMPONENT_SPLIT_RATIO);

            for (int j = 0; j < candidates.size(); j++) {
                if (i == j) continue;
                DetectionCandidate other = candidates.get(j);
                double otherArea = getArea(other.rect);

                if (otherArea < boxArea * 0.76
                        && getIntersectionOverSmaller(box.rect, other.rect) >= GROUP_CONTAINMENT_THRESHOLD) {
                    containedSmallerBoxes++;
                    if (containedSmallerBoxes >= 2) break;
                }
            }

            boolean looksLikeGroupByComponents = foregroundComponents >= 2
                    && boxArea > getArea(referenceInfo.objectRect) * LARGE_BOX_COMPONENT_RATIO;

            if (containedSmallerBoxes < 2 && !looksLikeGroupByComponents) {
                cleaned.add(box);
            }
        }

        return cleaned;
    }

    private static int countForegroundComponents(Rect rect, ReferenceInfo referenceInfo, double minArea) {
        Rect safe = clampRect(rect, referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
        Mat roiMask = new Mat(referenceInfo.foregroundMask, safe);
        Mat copy = roiMask.clone();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        int count = 0;
        try {
            Imgproc.findContours(copy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                double area = Geometry.contourArea(contour);
                if (area >= Math.max(MIN_CONTOUR_AREA, minArea)) {
                    count++;
                }
                contour.release();
            }
        } catch (Exception ignored) {
        } finally {
            roiMask.release();
            copy.release();
            hierarchy.release();
        }
        return count;
    }

    private static double getObjectRankScore(DetectionCandidate candidate, Rect referenceRect) {
        double refWidth = Math.max(1.0, referenceRect.width);
        double refHeight = Math.max(1.0, referenceRect.height);

        double widthRatio = Math.max(0.001, candidate.rect.width / refWidth);
        double heightRatio = Math.max(0.001, candidate.rect.height / refHeight);

        double sizePenalty = Math.abs(Math.log(widthRatio)) + Math.abs(Math.log(heightRatio));
        return candidate.score - SIZE_RANKING_WEIGHT * sizePenalty;
    }

    private static boolean shouldSuppress(Rect kept, Rect candidate, double overlapThreshold, Rect referenceRect) {
        double iou = getIntersectionOverUnion(kept, candidate);
        if (iou > overlapThreshold) {
            return true;
        }

        if (getIntersectionOverSmaller(kept, candidate) >= CONTAINMENT_NMS_THRESHOLD) {
            return true;
        }

        double maxDistance = Math.min(
                Math.min(kept.width, kept.height),
                Math.min(candidate.width, candidate.height)
        ) * CENTER_DUPLICATE_DISTANCE_RATIO;

        if (getCenterDistance(kept, candidate) <= maxDistance) {
            double areaRatio = getArea(kept) / Math.max(1.0, getArea(candidate));
            double normalizedAreaRatio = Math.max(areaRatio, 1.0 / Math.max(0.001, areaRatio));
            return normalizedAreaRatio <= 3.10;
        }

        // If both boxes are close to the expected object size and overlap a lot by the
        // smaller box, they are rotated/template duplicates around the same object.
        double smallerOverlap = getIntersectionOverSmaller(kept, candidate);
        if (smallerOverlap >= 0.55) {
            double keptSize = getArea(kept) / Math.max(1.0, getArea(referenceRect));
            double candidateSize = getArea(candidate) / Math.max(1.0, getArea(referenceRect));
            return keptSize <= MAX_AREA_RATIO && candidateSize <= MAX_AREA_RATIO;
        }

        return false;
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

    private static double getIntersectionOverSmaller(Rect r1, Rect r2) {
        int x1 = Math.max(r1.x, r2.x);
        int y1 = Math.max(r1.y, r2.y);
        int x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
        int y2 = Math.min(r1.y + r1.height, r2.y + r2.height);

        if (x1 >= x2 || y1 >= y2) return 0.0;

        double intersectionArea = (double) (x2 - x1) * (y2 - y1);
        double smallerArea = Math.min(getArea(r1), getArea(r2));
        if (smallerArea <= 0) return 0.0;
        return intersectionArea / smallerArea;
    }

    private static double getArea(Rect rect) {
        return Math.max(0.0, rect.width * (double) rect.height);
    }

    private static double getCenterDistance(Rect r1, Rect r2) {
        double c1x = r1.x + r1.width / 2.0;
        double c1y = r1.y + r1.height / 2.0;
        double c2x = r2.x + r2.width / 2.0;
        double c2y = r2.y + r2.height / 2.0;
        double dx = c1x - c2x;
        double dy = c1y - c2y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}
