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
import org.opencv.geometry.Geometry;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class OpenCVDetectionHelper {

    private static final int MIN_TEMPLATE_SIZE = 10;
    private static final int MAX_TEMPLATE_MATCHES = 14;
    private static final double MIN_CONTOUR_AREA = 28.0;

    // Wider than a classical detector, because CountLens must accept smaller/larger versions
    // of the selected object and also partly visible objects at the image border.
    private static final double MIN_BOX_AREA_RATIO = 0.10;
    private static final double MAX_BOX_AREA_RATIO = 7.50;
    private static final double MIN_CONTOUR_AREA_RATIO = 0.06;
    private static final double MAX_CONTOUR_AREA_RATIO = 8.50;
    private static final double MAX_ASPECT_LOG_DISTANCE = 1.35; // orientation-independent aspect tolerance

    // Template matching is used only as a fallback. The main photo path is component based.
    private static final double[] FAST_TEMPLATE_SCALES = {0.55, 0.70, 0.85, 1.00, 1.20, 1.45};
    private static final double[] FAST_TEMPLATE_ANGLES = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330};

    // Dense circular objects such as pipe openings need a detector that looks for one
    // circular/ring instance, not for a big connected blue component.  The normal color
    // component path merges touching pipes into blobs, so CountLens first tries this
    // reference-guided Hough ring detector when the selected object looks like a ring.
    private static final int MAX_RING_CANDIDATES_TO_VALIDATE = 900;
    private static final int MAX_RING_DETECTIONS = 650;

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

    private static class RingReference {
        final boolean valid;
        final double centerX;
        final double centerY;
        final double radius;
        final double innerGray;
        final double ringGray;
        final DominantColor ringColor;

        RingReference(boolean valid, double centerX, double centerY, double radius, double innerGray, double ringGray, DominantColor ringColor) {
            this.valid = valid;
            this.centerX = centerX;
            this.centerY = centerY;
            this.radius = radius;
            this.innerGray = innerGray;
            this.ringGray = ringGray;
            this.ringColor = ringColor;
        }
    }

    private static class ReferenceInfo {
        final Rect selectionRect;
        final Rect objectRect;
        final Mat foregroundMask;
        final MatOfPoint contour;
        final double contourArea;
        final double objectBoxArea;
        final double referenceAspect;
        final double referenceFillRatio;
        final double foregroundGray;
        final double backgroundGray;
        final boolean usesColorMask;
        final DominantColor dominantColor;

        ReferenceInfo(
                Rect selectionRect,
                Rect objectRect,
                Mat foregroundMask,
                MatOfPoint contour,
                double contourArea,
                double foregroundGray,
                double backgroundGray,
                boolean usesColorMask,
                DominantColor dominantColor
        ) {
            this.selectionRect = selectionRect;
            this.objectRect = objectRect;
            this.foregroundMask = foregroundMask;
            this.contour = contour;
            this.contourArea = Math.max(MIN_CONTOUR_AREA, contourArea);
            this.objectBoxArea = Math.max(1.0, objectRect.width * (double) objectRect.height);
            this.referenceAspect = normalizedAspect(objectRect);
            this.referenceFillRatio = this.contourArea / this.objectBoxArea;
            this.foregroundGray = foregroundGray;
            this.backgroundGray = backgroundGray;
            this.usesColorMask = usesColorMask;
            this.dominantColor = dominantColor;
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
            List<DetectionCandidate> candidates = new ArrayList<>();

            // Special but important case: dense circular/ring objects such as blue pipe
            // openings.  Color connected-components merge many touching pipes into one large
            // object, and template matching detects only fragments.  A selected pipe opening is
            // better represented as a ring/circle, so detect circles directly and validate them
            // by the selected ring color and dark inner hole.
            boolean usedRingDetector = addHoughRingCandidates(fullMat, grayFull, referenceInfo, candidates);

            if (!usedRingDetector) {
                // Best path for real photos: segment the selected object's foreground color/contrast,
                // merge broken visual pieces into one object component, then count one box per component.
                // This fixes the old behaviour where a bottle/heart was counted three times internally.
                addMergedObjectComponentCandidates(referenceInfo, candidates);

                // If the selected object has no reliable color mask, use shape/contrast components.
                if (candidates.isEmpty() || (!referenceInfo.usesColorMask && candidates.size() < 2)) {
                    addContrastShapeCandidates(grayFull, referenceInfo, candidates);
                }
            }

            // Expensive rotated template matching is now only a last-resort fallback. It is not used
            // when the component/ring detector already found objects, so phone photos stay fast.
            if (candidates.isEmpty()) {
                Rect objectRoi = clampRect(referenceInfo.objectRect, fullMat.cols(), fullMat.rows());
                Mat grayTemplate = new Mat(grayFull, objectRoi).clone();
                try {
                    if (Math.abs(initialRotation) > 0.5f) {
                        Mat corrected = rotateMat(grayTemplate, -initialRotation, estimateBorderMean(grayTemplate));
                        grayTemplate.release();
                        grayTemplate = corrected;
                    }
                    addTemplateFallbackCandidates(grayFull, grayTemplate, threshold, candidates);
                } finally {
                    grayTemplate.release();
                }
            }

            List<DetectionCandidate> finalDetections = applyObjectAwareNMS(candidates, referenceInfo, nmsThreshold);
            Bitmap resultBitmap = drawDetections(fullMat, finalDetections);
            callback.onDetectionComplete(resultBitmap, finalDetections.size());
        } catch (Exception e) {
            callback.onDetectionComplete(sourceBitmap, 0);
        } finally {
            fullMat.release();
            grayFull.release();
            if (referenceInfo != null) referenceInfo.release();
        }
    }

    private static Bitmap drawDetections(Mat fullMat, List<DetectionCandidate> detections) {
        Mat output = fullMat.clone();
        try {
            int index = 1;
            int total = detections.size();
            int thickness = total > 80 ? 2 : (total > 30 ? 3 : 4);
            double labelScale = total > 120 ? 0.36 : (total > 50 ? 0.48 : 0.85);
            int labelThickness = total > 80 ? 1 : (total > 30 ? 2 : 3);
            boolean drawLabels = total <= 180;
            for (DetectionCandidate candidate : detections) {
                Rect rect = clampRect(candidate.rect, output.cols(), output.rows());
                Imgproc.rectangle(output, rect.tl(), rect.br(), new Scalar(34, 197, 94, 255), thickness);
                if (drawLabels) {
                    Imgproc.putText(
                            output,
                            String.valueOf(index),
                            new Point(rect.x + 3, Math.max(12, rect.y + Math.max(10, (int) Math.round(18 * labelScale)))),
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

    private static ReferenceInfo buildReferenceInfo(Mat fullRgba, Mat grayFull, Rect selectionRoi) {
        Mat localMask = new Mat();
        Mat fullMask = new Mat();
        MatOfPoint referenceContour = null;
        Rect objectRect = selectionRoi;
        double contourArea = Math.max(MIN_CONTOUR_AREA, selectionRoi.width * (double) selectionRoi.height);
        double foregroundGray = 0.0;
        double backgroundGray = 255.0;
        boolean usesColorMask = false;
        DominantColor dominantColor = new DominantColor(false, 0.0, 0.0, 0.0);

        try {
            LocalMaskResult local = createLocalReferenceMask(fullRgba, grayFull, selectionRoi);
            localMask = local.mask;
            dominantColor = local.dominantColor;

            referenceContour = findBestReferenceContour(localMask);
            if (referenceContour != null && Geometry.contourArea(referenceContour) >= MIN_CONTOUR_AREA) {
                Rect localRect = Geometry.boundingRect(referenceContour);
                objectRect = new Rect(
                        selectionRoi.x + localRect.x,
                        selectionRoi.y + localRect.y,
                        localRect.width,
                        localRect.height
                );
                objectRect = expandRect(objectRect, 3, fullRgba.cols(), fullRgba.rows());
                contourArea = Geometry.contourArea(referenceContour);
            } else {
                if (referenceContour != null) referenceContour.release();
                referenceContour = rectangleContour(new Rect(0, 0, selectionRoi.width, selectionRoi.height));
                contourArea = Math.max(MIN_CONTOUR_AREA, selectionRoi.width * (double) selectionRoi.height);
            }

            Mat grayRoi = new Mat(grayFull, selectionRoi);
            try {
                foregroundGray = Core.mean(grayRoi, localMask).val[0];
                backgroundGray = estimateBorderMean(grayRoi);
            } finally {
                grayRoi.release();
            }

            MaskBuildResult full = createFullForegroundMask(fullRgba, grayFull, selectionRoi, objectRect, localMask, dominantColor, foregroundGray, backgroundGray);
            fullMask = full.mask;
            usesColorMask = full.usesColorMask;
        } catch (Exception ignored) {
            fullMask = Mat.zeros(grayFull.rows(), grayFull.cols(), CvType.CV_8UC1);
            Imgproc.rectangle(fullMask, selectionRoi.tl(), selectionRoi.br(), new Scalar(255), -1);
            if (referenceContour != null) referenceContour.release();
            referenceContour = rectangleContour(new Rect(0, 0, selectionRoi.width, selectionRoi.height));
        } finally {
            localMask.release();
        }

        return new ReferenceInfo(
                selectionRoi,
                objectRect,
                fullMask,
                referenceContour,
                contourArea,
                foregroundGray,
                backgroundGray,
                usesColorMask,
                dominantColor
        );
    }

    private static class LocalMaskResult {
        final Mat mask;
        final DominantColor dominantColor;

        LocalMaskResult(Mat mask, DominantColor dominantColor) {
            this.mask = mask;
            this.dominantColor = dominantColor;
        }
    }

    private static class MaskBuildResult {
        final Mat mask;
        final boolean usesColorMask;

        MaskBuildResult(Mat mask, boolean usesColorMask) {
            this.mask = mask;
            this.usesColorMask = usesColorMask;
        }
    }

    private static LocalMaskResult createLocalReferenceMask(Mat fullRgba, Mat grayFull, Rect selectionRoi) {
        Mat grayRoi = new Mat(grayFull, selectionRoi);
        Mat contrastMask = new Mat();
        Mat colorMask = new Mat();
        Mat finalMask = new Mat();
        DominantColor dominantColor = new DominantColor(false, 0.0, 0.0, 0.0);

        try {
            double background = estimateBorderMean(grayRoi);
            Mat bg = new Mat(grayRoi.size(), grayRoi.type(), new Scalar(background));
            Mat diff = new Mat();
            try {
                Core.absdiff(grayRoi, bg, diff);
                Imgproc.threshold(diff, contrastMask, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
                if (Core.countNonZero(contrastMask) < Math.max(MIN_CONTOUR_AREA, grayRoi.rows() * grayRoi.cols() * 0.015)) {
                    Imgproc.threshold(diff, contrastMask, 18, 255, Imgproc.THRESH_BINARY);
                }
            } finally {
                bg.release();
                diff.release();
            }

            Mat hsvRoi = new Mat();
            Mat rgbRoi = new Mat();
            Mat rgbaRoi = new Mat(fullRgba, selectionRoi);
            try {
                Imgproc.cvtColor(rgbaRoi, rgbRoi, Imgproc.COLOR_RGBA2RGB);
                Imgproc.cvtColor(rgbRoi, hsvRoi, Imgproc.COLOR_RGB2HSV);
                dominantColor = estimateDominantColor(hsvRoi, contrastMask);
                if (dominantColor.valid) {
                    colorMask = createHueMask(hsvRoi, dominantColor.hue, dominantColor.saturation, dominantColor.value);
                } else {
                    colorMask = Mat.zeros(grayRoi.rows(), grayRoi.cols(), CvType.CV_8UC1);
                }
            } finally {
                rgbaRoi.release();
                rgbRoi.release();
                hsvRoi.release();
            }

            // Use the colored foreground when reliable; otherwise use contrast. For photos, this
            // makes red bottles/cupcakes independent from the blue/background region.
            if (dominantColor.valid && Core.countNonZero(colorMask) >= MIN_CONTOUR_AREA) {
                Core.bitwise_or(colorMask, contrastMask, finalMask);
            } else {
                finalMask = contrastMask.clone();
            }

            cleanMaskForSingleObject(finalMask, selectionRoi);
            if (Core.countNonZero(finalMask) < MIN_CONTOUR_AREA) {
                finalMask.release();
                finalMask = Mat.ones(grayRoi.rows(), grayRoi.cols(), CvType.CV_8UC1);
                finalMask.setTo(new Scalar(255));
            }
            return new LocalMaskResult(finalMask, dominantColor);
        } finally {
            grayRoi.release();
            contrastMask.release();
            colorMask.release();
        }
    }

    private static MaskBuildResult createFullForegroundMask(
            Mat fullRgba,
            Mat grayFull,
            Rect selectionRoi,
            Rect objectRect,
            Mat localReferenceMask,
            DominantColor dominantColor,
            double foregroundGray,
            double backgroundGray
    ) {
        Mat finalMask = new Mat();
        boolean useColorMask = false;

        try {
            if (dominantColor.valid && dominantColor.saturation >= 38.0 && dominantColor.value >= 25.0) {
                Mat rgbFull = new Mat();
                Mat hsvFull = new Mat();
                try {
                    Imgproc.cvtColor(fullRgba, rgbFull, Imgproc.COLOR_RGBA2RGB);
                    Imgproc.cvtColor(rgbFull, hsvFull, Imgproc.COLOR_RGB2HSV);
                    finalMask = createHueMask(hsvFull, dominantColor.hue, dominantColor.saturation, dominantColor.value);
                    useColorMask = Core.countNonZero(finalMask) > MIN_CONTOUR_AREA;
                } finally {
                    rgbFull.release();
                    hsvFull.release();
                }
            }

            if (!useColorMask) {
                double mid = (foregroundGray + backgroundGray) / 2.0;
                if (foregroundGray < backgroundGray) {
                    Imgproc.threshold(grayFull, finalMask, mid, 255, Imgproc.THRESH_BINARY_INV);
                } else {
                    Imgproc.threshold(grayFull, finalMask, mid, 255, Imgproc.THRESH_BINARY);
                }
            }

            // This is the important robust step: close/dilate the mask using the size of the
            // selected object, not a fixed 3x3 kernel. It merges highlights, black decoration lines,
            // and broken red pieces into one physical object before counting.
            cleanMaskForWholeImage(finalMask, objectRect, useColorMask);
            return new MaskBuildResult(finalMask, useColorMask);
        } catch (Exception ignored) {
            finalMask.release();
            Mat fallback = Mat.zeros(grayFull.rows(), grayFull.cols(), CvType.CV_8UC1);
            Imgproc.rectangle(fallback, selectionRoi.tl(), selectionRoi.br(), new Scalar(255), -1);
            return new MaskBuildResult(fallback, false);
        }
    }

    private static DominantColor estimateDominantColor(Mat hsvRoi, Mat supportMask) {
        int[] hist = new int[180];
        double[] satSum = new double[180];
        double[] valSum = new double[180];
        int total = 0;

        for (int y = 0; y < hsvRoi.rows(); y++) {
            for (int x = 0; x < hsvRoi.cols(); x++) {
                double[] support = supportMask.get(y, x);
                if (support == null || support.length == 0 || support[0] <= 0) continue;
                double[] hsv = hsvRoi.get(y, x);
                if (hsv == null || hsv.length < 3) continue;
                double sat = hsv[1];
                double val = hsv[2];
                if (sat < 40.0 || val < 28.0) continue;
                int h = clamp((int) Math.round(hsv[0]), 0, 179);
                hist[h]++;
                satSum[h] += sat;
                valSum[h] += val;
                total++;
            }
        }

        if (total < 18) return new DominantColor(false, 0.0, 0.0, 0.0);

        int bestHue = 0;
        int bestCount = -1;
        for (int h = 0; h < 180; h++) {
            int smoothed = 0;
            for (int offset = -5; offset <= 5; offset++) {
                smoothed += hist[(h + offset + 180) % 180];
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
            if (dist <= 13 && hist[h] > 0) {
                double angle = (2.0 * Math.PI * h) / 180.0;
                sumCos += Math.cos(angle) * hist[h];
                sumSin += Math.sin(angle) * hist[h];
                sumSat += satSum[h];
                sumVal += valSum[h];
                count += hist[h];
            }
        }

        if (count < 10) return new DominantColor(false, 0.0, 0.0, 0.0);
        double hue = (Math.atan2(sumSin, sumCos) * 180.0) / (2.0 * Math.PI);
        if (hue < 0) hue += 180.0;
        return new DominantColor(true, hue, sumSat / count, sumVal / count);
    }

    private static Mat createHueMask(Mat hsv, double hue, double saturation, double value) {
        int hueTolerance = saturation > 110 ? 16 : 24;
        double minSat = Math.max(55.0, saturation * 0.46);
        double minValue = Math.max(25.0, value * 0.20);
        Mat mask = new Mat();

        double lowHue = hue - hueTolerance;
        double highHue = hue + hueTolerance;
        if (lowHue < 0) {
            Mat a = new Mat();
            Mat b = new Mat();
            Core.inRange(hsv, new Scalar(0, minSat, minValue), new Scalar(highHue, 255, 255), a);
            Core.inRange(hsv, new Scalar(180 + lowHue, minSat, minValue), new Scalar(180, 255, 255), b);
            Core.bitwise_or(a, b, mask);
            a.release();
            b.release();
        } else if (highHue > 180) {
            Mat a = new Mat();
            Mat b = new Mat();
            Core.inRange(hsv, new Scalar(lowHue, minSat, minValue), new Scalar(180, 255, 255), a);
            Core.inRange(hsv, new Scalar(0, minSat, minValue), new Scalar(highHue - 180, 255, 255), b);
            Core.bitwise_or(a, b, mask);
            a.release();
            b.release();
        } else {
            Core.inRange(hsv, new Scalar(lowHue, minSat, minValue), new Scalar(highHue, 255, 255), mask);
        }
        return mask;
    }

    private static void cleanMaskForSingleObject(Mat mask, Rect selectionRoi) {
        int base = Math.max(7, Math.min(selectionRoi.width, selectionRoi.height));
        Mat open = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat close = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(odd(base * 0.09), odd(base * 0.09)));
        try {
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, open);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, close);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, close);
        } finally {
            open.release();
            close.release();
        }
    }

    private static void cleanMaskForWholeImage(Mat mask, Rect objectRect, boolean colorMask) {
        int base = Math.max(7, Math.min(objectRect.width, objectRect.height));
        int openSize = colorMask ? 3 : 5;
        int closeSize = colorMask ? odd(base * 0.13) : odd(base * 0.08);
        int dilateSize = colorMask ? odd(base * 0.035) : 3;

        Mat open = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(openSize, openSize));
        Mat close = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(closeSize, closeSize));
        Mat dilate = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(dilateSize, dilateSize));
        try {
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, open);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, close);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, close);
            if (colorMask && dilateSize > 3) {
                Imgproc.dilate(mask, mask, dilate);
                Imgproc.erode(mask, mask, dilate);
            }
        } finally {
            open.release();
            close.release();
            dilate.release();
        }
    }


    private static boolean addHoughRingCandidates(Mat fullRgba, Mat grayFull, ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        RingReference ringReference = buildRingReference(fullRgba, grayFull, referenceInfo.selectionRect);
        if (!ringReference.valid) return false;

        Mat blurred = new Mat();
        Mat circles = new Mat();
        try {
            Imgproc.medianBlur(grayFull, blurred, 5);

            int minRadius = Math.max(3, (int) Math.round(ringReference.radius * 0.45));
            int maxRadius = Math.max(minRadius + 2, (int) Math.round(ringReference.radius * 1.85));
            double minDistance = Math.max(5.0, ringReference.radius * 0.90);

            // A low param2 is needed for dense pipe stacks, but every Hough proposal is
            // validated afterwards using selected color + dark-hole evidence to remove noise.
            Imgproc.HoughCircles(
                    blurred,
                    circles,
                    Imgproc.HOUGH_GRADIENT,
                    1.18,
                    minDistance,
                    85.0,
                    13.0,
                    minRadius,
                    maxRadius
            );

            int checked = 0;
            for (int i = 0; i < circles.cols() && checked < MAX_RING_CANDIDATES_TO_VALIDATE; i++) {
                double[] c = circles.get(0, i);
                if (c == null || c.length < 3) continue;
                checked++;
                DetectionCandidate candidate = validateRingCandidate(fullRgba, grayFull, c[0], c[1], c[2], ringReference);
                if (candidate != null) {
                    candidates.add(candidate);
                }
            }

            if (candidates.size() < 4) {
                candidates.clear();
                return false;
            }

            // Keep the best candidates before the generic NMS.  This prevents bad Hough noise
            // from dominating runtime on very large/dense industrial photos.
            Collections.sort(candidates, (a, b) -> Double.compare(b.score, a.score));
            if (candidates.size() > MAX_RING_DETECTIONS) {
                candidates.subList(MAX_RING_DETECTIONS, candidates.size()).clear();
            }
            return true;
        } catch (Exception ignored) {
            candidates.clear();
            return false;
        } finally {
            blurred.release();
            circles.release();
        }
    }

    private static RingReference buildRingReference(Mat fullRgba, Mat grayFull, Rect selectionRoi) {
        if (selectionRoi.width < MIN_TEMPLATE_SIZE || selectionRoi.height < MIN_TEMPLATE_SIZE) {
            return new RingReference(false, 0, 0, 0, 0, 0, new DominantColor(false, 0, 0, 0));
        }

        double aspect = normalizedAspect(selectionRoi);
        if (aspect > 2.35) {
            return new RingReference(false, 0, 0, 0, 0, 0, new DominantColor(false, 0, 0, 0));
        }

        Mat grayRoi = new Mat(grayFull, selectionRoi);
        Mat blurred = new Mat();
        Mat circles = new Mat();
        try {
            Imgproc.medianBlur(grayRoi, blurred, 5);
            int minSide = Math.min(selectionRoi.width, selectionRoi.height);
            int minRadius = Math.max(3, (int) Math.round(minSide * 0.16));
            int maxRadius = Math.max(minRadius + 2, (int) Math.round(minSide * 0.62));
            Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.15, Math.max(6.0, minSide * 0.55), 85.0, 9.5, minRadius, maxRadius);

            double cx = selectionRoi.width / 2.0;
            double cy = selectionRoi.height / 2.0;
            double radius = Math.max(3.0, minSide * 0.34);
            double bestDistance = Double.MAX_VALUE;
            boolean hasCircle = false;
            for (int i = 0; i < circles.cols(); i++) {
                double[] c = circles.get(0, i);
                if (c == null || c.length < 3) continue;
                double distance = Math.sqrt(Math.pow(c[0] - cx, 2) + Math.pow(c[1] - cy, 2));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    cx = c[0];
                    cy = c[1];
                    radius = c[2];
                    hasCircle = true;
                }
            }

            Mat innerMask = createCircleMask(grayRoi.rows(), grayRoi.cols(), cx, cy, radius * 0.52);
            Mat ringMask = createAnnulusMask(grayRoi.rows(), grayRoi.cols(), cx, cy, radius * 0.72, radius * 1.35);
            Mat rgbaRoi = new Mat(fullRgba, selectionRoi);
            Mat rgbRoi = new Mat();
            Mat hsvRoi = new Mat();
            try {
                Imgproc.cvtColor(rgbaRoi, rgbRoi, Imgproc.COLOR_RGBA2RGB);
                Imgproc.cvtColor(rgbRoi, hsvRoi, Imgproc.COLOR_RGB2HSV);
                DominantColor ringColor = estimateDominantColor(hsvRoi, ringMask);
                double innerGray = Core.mean(grayRoi, innerMask).val[0];
                double ringGray = Core.mean(grayRoi, ringMask).val[0];
                int ringPixels = Core.countNonZero(ringMask);
                int innerPixels = Core.countNonZero(innerMask);
                double colorRatio = ringColor.valid ? colorSupportRatio(hsvRoi, ringMask, ringColor, 26.0) : 0.0;

                boolean darkCenter = innerPixels > 8 && ringPixels > 16 && (innerGray + 8.0 < ringGray || innerGray < 80.0);
                boolean coloredRing = ringColor.valid && ringColor.saturation >= 45.0 && colorRatio >= 0.14;
                boolean roundSelection = aspect <= 1.70 || hasCircle;
                boolean valid = roundSelection && darkCenter && coloredRing;

                return new RingReference(valid, selectionRoi.x + cx, selectionRoi.y + cy, radius, innerGray, ringGray, ringColor);
            } finally {
                innerMask.release();
                ringMask.release();
                rgbaRoi.release();
                rgbRoi.release();
                hsvRoi.release();
            }
        } catch (Exception ignored) {
            return new RingReference(false, 0, 0, 0, 0, 0, new DominantColor(false, 0, 0, 0));
        } finally {
            grayRoi.release();
            blurred.release();
            circles.release();
        }
    }

    private static DetectionCandidate validateRingCandidate(Mat fullRgba, Mat grayFull, double cx, double cy, double radius, RingReference reference) {
        if (radius < 3.0) return null;
        int outer = (int) Math.ceil(radius * 1.36);
        int left = (int) Math.round(cx - outer);
        int top = (int) Math.round(cy - outer);
        int right = (int) Math.round(cx + outer);
        int bottom = (int) Math.round(cy + outer);
        if (right <= 0 || bottom <= 0 || left >= grayFull.cols() || top >= grayFull.rows()) return null;

        Rect roi = clampRect(new Rect(left, top, right - left, bottom - top), grayFull.cols(), grayFull.rows());
        if (roi.width < MIN_TEMPLATE_SIZE || roi.height < MIN_TEMPLATE_SIZE) return null;

        double localCx = cx - roi.x;
        double localCy = cy - roi.y;
        Mat grayRoi = new Mat(grayFull, roi);
        Mat rgbaRoi = new Mat(fullRgba, roi);
        Mat innerMask = createCircleMask(grayRoi.rows(), grayRoi.cols(), localCx, localCy, radius * 0.52);
        Mat ringMask = createAnnulusMask(grayRoi.rows(), grayRoi.cols(), localCx, localCy, radius * 0.72, radius * 1.34);
        Mat rgbRoi = new Mat();
        Mat hsvRoi = new Mat();
        try {
            int ringPixels = Core.countNonZero(ringMask);
            int innerPixels = Core.countNonZero(innerMask);
            if (ringPixels < 14 || innerPixels < 6) return null;

            Imgproc.cvtColor(rgbaRoi, rgbRoi, Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(rgbRoi, hsvRoi, Imgproc.COLOR_RGB2HSV);
            double colorRatio = colorSupportRatio(hsvRoi, ringMask, reference.ringColor, 30.0);
            if (colorRatio < 0.16) return null;

            double innerGray = Core.mean(grayRoi, innerMask).val[0];
            double ringGray = Core.mean(grayRoi, ringMask).val[0];
            boolean darkCenter = innerGray + 6.0 < ringGray || innerGray < Math.min(85.0, reference.ringGray * 0.72 + 12.0);
            if (!darkCenter) return null;

            double radiusRatio = radius / Math.max(1.0, reference.radius);
            if (radiusRatio < 0.45 || radiusRatio > 1.95) return null;

            double sizePenalty = Math.abs(Math.log(Math.max(0.12, radiusRatio)));
            double darknessScore = Math.max(0.0, Math.min(1.0, (ringGray - innerGray) / 70.0));
            double score = 4.0 + 1.8 * colorRatio + 0.9 * darknessScore - 0.25 * sizePenalty;

            Rect box = expandRect(new Rect(
                    (int) Math.round(cx - radius * 1.08),
                    (int) Math.round(cy - radius * 1.08),
                    Math.max(1, (int) Math.round(radius * 2.16)),
                    Math.max(1, (int) Math.round(radius * 2.16))
            ), 1, grayFull.cols(), grayFull.rows());
            return new DetectionCandidate(box, score);
        } catch (Exception ignored) {
            return null;
        } finally {
            grayRoi.release();
            rgbaRoi.release();
            innerMask.release();
            ringMask.release();
            rgbRoi.release();
            hsvRoi.release();
        }
    }

    private static Mat createCircleMask(int rows, int cols, double cx, double cy, double radius) {
        Mat mask = Mat.zeros(rows, cols, CvType.CV_8UC1);
        Imgproc.circle(mask, new Point(cx, cy), Math.max(1, (int) Math.round(radius)), new Scalar(255), -1);
        return mask;
    }

    private static Mat createAnnulusMask(int rows, int cols, double cx, double cy, double innerRadius, double outerRadius) {
        Mat mask = Mat.zeros(rows, cols, CvType.CV_8UC1);
        Imgproc.circle(mask, new Point(cx, cy), Math.max(1, (int) Math.round(outerRadius)), new Scalar(255), -1);
        Imgproc.circle(mask, new Point(cx, cy), Math.max(1, (int) Math.round(innerRadius)), new Scalar(0), -1);
        return mask;
    }

    private static double colorSupportRatio(Mat hsv, Mat supportMask, DominantColor color, double hueTolerance) {
        if (hsv.empty() || supportMask.empty() || color == null || !color.valid) return 0.0;
        int total = 0;
        int matched = 0;
        double minSat = Math.max(35.0, color.saturation * 0.36);
        double minValue = Math.max(18.0, color.value * 0.16);
        for (int y = 0; y < hsv.rows(); y++) {
            for (int x = 0; x < hsv.cols(); x++) {
                double[] support = supportMask.get(y, x);
                if (support == null || support.length == 0 || support[0] <= 0) continue;
                total++;
                double[] pixel = hsv.get(y, x);
                if (pixel == null || pixel.length < 3) continue;
                if (pixel[1] < minSat || pixel[2] < minValue) continue;
                if (hueDistance(pixel[0], color.hue) <= hueTolerance) matched++;
            }
        }
        if (total <= 0) return 0.0;
        return matched / (double) total;
    }

    private static double hueDistance(double a, double b) {
        double diff = Math.abs(a - b);
        return Math.min(diff, 180.0 - diff);
    }

    private static void addMergedObjectComponentCandidates(ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat copy = referenceInfo.foregroundMask.clone();
        try {
            Imgproc.findContours(copy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                try {
                    double area = Geometry.contourArea(contour);
                    if (area < Math.max(MIN_CONTOUR_AREA, referenceInfo.contourArea * MIN_CONTOUR_AREA_RATIO)) {
                        continue;
                    }
                    Rect rect = Geometry.boundingRect(contour);
                    Rect padded = expandRect(rect, Math.max(2, (int) Math.round(Math.min(rect.width, rect.height) * 0.025)), referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
                    CandidateScore score = validateComponent(padded, area, referenceInfo);
                    if (score.valid) {
                        candidates.add(new DetectionCandidate(padded, score.score));
                    }
                } finally {
                    contour.release();
                }
            }
        } catch (Exception ignored) {
        } finally {
            copy.release();
            hierarchy.release();
        }
    }

    private static class CandidateScore {
        final boolean valid;
        final double score;

        CandidateScore(boolean valid, double score) {
            this.valid = valid;
            this.score = score;
        }
    }

    private static CandidateScore validateComponent(Rect rect, double contourArea, ReferenceInfo referenceInfo) {
        if (rect.width < MIN_TEMPLATE_SIZE || rect.height < MIN_TEMPLATE_SIZE) {
            return new CandidateScore(false, 0.0);
        }

        double boxArea = Math.max(1.0, rect.width * (double) rect.height);
        double boxRatio = boxArea / referenceInfo.objectBoxArea;
        double contourRatio = contourArea / Math.max(1.0, referenceInfo.contourArea);
        if (boxRatio < MIN_BOX_AREA_RATIO || boxRatio > MAX_BOX_AREA_RATIO) {
            return new CandidateScore(false, 0.0);
        }
        if (contourRatio < MIN_CONTOUR_AREA_RATIO || contourRatio > MAX_CONTOUR_AREA_RATIO) {
            return new CandidateScore(false, 0.0);
        }

        double aspect = normalizedAspect(rect);
        double aspectDistance = Math.abs(Math.log(Math.max(0.05, aspect / Math.max(0.05, referenceInfo.referenceAspect))));
        if (aspectDistance > MAX_ASPECT_LOG_DISTANCE) {
            return new CandidateScore(false, 0.0);
        }

        double fillRatio = contourArea / boxArea;
        boolean plausibleFill = fillRatio >= Math.max(0.02, referenceInfo.referenceFillRatio * 0.25)
                && fillRatio <= Math.min(0.98, referenceInfo.referenceFillRatio * 3.20 + 0.15);
        if (!plausibleFill && boxRatio < 0.22) {
            return new CandidateScore(false, 0.0);
        }

        // Border objects are often partly visible, so do not reject them. Just give them lower score.
        double sizePenalty = Math.abs(Math.log(Math.max(0.06, boxRatio)));
        double contourPenalty = Math.abs(Math.log(Math.max(0.06, contourRatio)));
        double fillPenalty = Math.abs(fillRatio - referenceInfo.referenceFillRatio);
        double score = 3.0 - 0.34 * sizePenalty - 0.20 * contourPenalty - 0.35 * aspectDistance - 0.22 * fillPenalty;
        return new CandidateScore(true, score);
    }

    private static void addContrastShapeCandidates(Mat grayFull, ReferenceInfo referenceInfo, List<DetectionCandidate> candidates) {
        Mat binary = new Mat();
        Mat copy = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        try {
            boolean darkObject = referenceInfo.foregroundGray < referenceInfo.backgroundGray;
            int type = darkObject ? Imgproc.THRESH_BINARY_INV : Imgproc.THRESH_BINARY;
            Imgproc.threshold(grayFull, binary, 0, 255, type + Imgproc.THRESH_OTSU);
            cleanMaskForWholeImage(binary, referenceInfo.objectRect, false);
            copy = binary.clone();
            Imgproc.findContours(copy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint contour : contours) {
                try {
                    double area = Geometry.contourArea(contour);
                    Rect rect = Geometry.boundingRect(contour);
                    CandidateScore basic = validateComponent(rect, area, referenceInfo);
                    if (!basic.valid) continue;
                    double shape = Geometry.matchShapes(referenceInfo.contour, contour, Imgproc.CONTOURS_MATCH_I3, 0.0);
                    if (shape <= 0.80) {
                        candidates.add(new DetectionCandidate(rect, basic.score - Math.min(0.9, shape)));
                    }
                } finally {
                    contour.release();
                }
            }
        } catch (Exception ignored) {
        } finally {
            binary.release();
            copy.release();
            hierarchy.release();
        }
    }

    private static void addTemplateFallbackCandidates(Mat grayFull, Mat grayTemplate, double threshold, List<DetectionCandidate> candidates) {
        if (grayTemplate.empty() || grayTemplate.cols() < MIN_TEMPLATE_SIZE || grayTemplate.rows() < MIN_TEMPLATE_SIZE) return;
        double borderValue = estimateBorderMean(grayTemplate);
        double activeThreshold = Math.max(0.62, threshold);
        int totalCollected = 0;

        for (double angle : FAST_TEMPLATE_ANGLES) {
            if (totalCollected >= MAX_TEMPLATE_MATCHES * 3) break;
            Mat rotated = new Mat();
            try {
                rotated = rotateMat(grayTemplate, angle, borderValue);
                if (rotated.cols() < MIN_TEMPLATE_SIZE || rotated.rows() < MIN_TEMPLATE_SIZE) continue;
                for (double scale : FAST_TEMPLATE_SCALES) {
                    if (totalCollected >= MAX_TEMPLATE_MATCHES * 3) break;
                    Mat scaled = new Mat();
                    Mat result = new Mat();
                    try {
                        int w = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(rotated.cols() * scale));
                        int h = Math.max(MIN_TEMPLATE_SIZE, (int) Math.round(rotated.rows() * scale));
                        if (w >= grayFull.cols() || h >= grayFull.rows()) continue;
                        Imgproc.resize(rotated, scaled, new Size(w, h));
                        Imgproc.matchTemplate(grayFull, scaled, result, Imgproc.TM_CCOEFF_NORMED);
                        totalCollected += collectLocalMaxima(result, w, h, activeThreshold, candidates, MAX_TEMPLATE_MATCHES);
                    } finally {
                        scaled.release();
                        result.release();
                    }
                }
            } finally {
                rotated.release();
            }
        }
    }

    private static int collectLocalMaxima(Mat matchResult, int width, int height, double threshold, List<DetectionCandidate> candidates, int maxCount) {
        int collected = 0;
        while (collected < maxCount) {
            Core.MinMaxLocResult mm = Core.minMaxLoc(matchResult);
            if (mm.maxVal < threshold) break;
            int x = (int) Math.round(mm.maxLoc.x);
            int y = (int) Math.round(mm.maxLoc.y);
            candidates.add(new DetectionCandidate(new Rect(x, y, width, height), mm.maxVal));
            collected++;

            int sw = Math.max(8, (int) Math.round(width * 0.70));
            int sh = Math.max(8, (int) Math.round(height * 0.70));
            Imgproc.rectangle(
                    matchResult,
                    new Point(Math.max(0, x - sw / 2.0), Math.max(0, y - sh / 2.0)),
                    new Point(Math.min(matchResult.cols() - 1, x + sw / 2.0), Math.min(matchResult.rows() - 1, y + sh / 2.0)),
                    new Scalar(0),
                    -1
            );
        }
        return collected;
    }

    private static List<DetectionCandidate> applyObjectAwareNMS(List<DetectionCandidate> candidates, ReferenceInfo referenceInfo, double overlapThreshold) {
        List<DetectionCandidate> result = new ArrayList<>();
        if (candidates.isEmpty()) return result;

        List<DetectionCandidate> sorted = new ArrayList<>(candidates);
        Collections.sort(sorted, (a, b) -> Double.compare(b.score, a.score));

        for (DetectionCandidate candidate : sorted) {
            Rect candidateRect = clampRect(candidate.rect, referenceInfo.foregroundMask.cols(), referenceInfo.foregroundMask.rows());
            if (candidateRect.width < MIN_TEMPLATE_SIZE || candidateRect.height < MIN_TEMPLATE_SIZE) continue;

            boolean duplicate = false;
            for (DetectionCandidate kept : result) {
                if (shouldSuppress(kept.rect, candidateRect, overlapThreshold)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                result.add(new DetectionCandidate(candidateRect, candidate.score));
            }
        }

        // A final cleanup removes tiny boxes fully inside larger accepted boxes. This is critical
        // for selected bottles/hearts where specular highlights or decorations can otherwise be counted.
        result = removeContainedFragments(result);
        result.sort(Comparator.comparingInt((DetectionCandidate c) -> c.rect.y).thenComparingInt(c -> c.rect.x));
        return result;
    }

    private static boolean shouldSuppress(Rect kept, Rect candidate, double overlapThreshold) {
        double iou = intersectionOverUnion(kept, candidate);
        if (iou >= Math.max(0.12, overlapThreshold)) return true;

        double smallerOverlap = intersectionOverSmaller(kept, candidate);
        if (smallerOverlap >= 0.70) return true;

        double centerDistance = centerDistance(kept, candidate);
        double minSide = Math.min(Math.min(kept.width, kept.height), Math.min(candidate.width, candidate.height));
        if (centerDistance <= minSide * 0.48) {
            double areaRatio = area(kept) / Math.max(1.0, area(candidate));
            double normalized = Math.max(areaRatio, 1.0 / Math.max(0.001, areaRatio));
            return normalized <= 4.0;
        }
        return false;
    }

    private static List<DetectionCandidate> removeContainedFragments(List<DetectionCandidate> input) {
        List<DetectionCandidate> cleaned = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            DetectionCandidate current = input.get(i);
            boolean insideLarger = false;
            for (int j = 0; j < input.size(); j++) {
                if (i == j) continue;
                DetectionCandidate other = input.get(j);
                if (area(current.rect) < area(other.rect) * 0.72
                        && intersectionOverSmaller(current.rect, other.rect) >= 0.82) {
                    insideLarger = true;
                    break;
                }
            }
            if (!insideLarger) cleaned.add(current);
        }
        return cleaned;
    }

    private static MatOfPoint findBestReferenceContour(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat copy = mask.clone();
        MatOfPoint best = null;
        try {
            Imgproc.findContours(copy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            double bestScore = 0.0;
            double cx = mask.cols() / 2.0;
            double cy = mask.rows() / 2.0;
            double maxDist = Math.sqrt(cx * cx + cy * cy) + 1e-6;
            for (MatOfPoint contour : contours) {
                double area = Geometry.contourArea(contour);
                if (area < MIN_CONTOUR_AREA) {
                    contour.release();
                    continue;
                }
                Rect rect = Geometry.boundingRect(contour);
                double rcx = rect.x + rect.width / 2.0;
                double rcy = rect.y + rect.height / 2.0;
                double distance = Math.sqrt(Math.pow(rcx - cx, 2) + Math.pow(rcy - cy, 2));
                double centerWeight = 1.0 - Math.min(1.0, distance / maxDist);
                double score = area * (1.0 + 0.40 * centerWeight);
                if (score > bestScore) {
                    if (best != null) best.release();
                    best = contour;
                    bestScore = score;
                } else {
                    contour.release();
                }
            }
            return best;
        } finally {
            copy.release();
            hierarchy.release();
        }
    }

    private static Mat rotateMat(Mat src, double angle, double borderValue) {
        double normalized = ((angle % 360) + 360) % 360;
        if (Math.abs(normalized) < 0.001) return src.clone();

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
        Imgproc.warpAffine(src, rotated, rotationMatrix, new Size(newWidth, newHeight), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(borderValue));
        rotationMatrix.release();
        return rotated;
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

    private static double estimateBorderMean(Mat mat) {
        if (mat.empty()) return 255.0;
        int border = Math.max(1, Math.min(mat.cols(), mat.rows()) / 8);
        Mat top = new Mat(mat, new Rect(0, 0, mat.cols(), border));
        Mat bottom = new Mat(mat, new Rect(0, mat.rows() - border, mat.cols(), border));
        Mat left = new Mat(mat, new Rect(0, 0, border, mat.rows()));
        Mat right = new Mat(mat, new Rect(mat.cols() - border, 0, border, mat.rows()));
        try {
            return (Core.mean(top).val[0] + Core.mean(bottom).val[0] + Core.mean(left).val[0] + Core.mean(right).val[0]) / 4.0;
        } finally {
            top.release();
            bottom.release();
            left.release();
            right.release();
        }
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
        return clampRect(new Rect(rect.x - padding, rect.y - padding, rect.width + padding * 2, rect.height + padding * 2), maxWidth, maxHeight);
    }

    private static int odd(double raw) {
        int value = (int) Math.round(raw);
        value = Math.max(3, Math.min(55, value));
        if (value % 2 == 0) value++;
        return value;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double normalizedAspect(Rect rect) {
        double w = Math.max(1.0, rect.width);
        double h = Math.max(1.0, rect.height);
        return Math.max(w, h) / Math.min(w, h);
    }

    private static double area(Rect rect) {
        return Math.max(0.0, rect.width * (double) rect.height);
    }

    private static double intersectionOverUnion(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);
        if (x1 >= x2 || y1 >= y2) return 0.0;
        double intersection = (x2 - x1) * (double) (y2 - y1);
        double union = area(a) + area(b) - intersection;
        if (union <= 0) return 0.0;
        return intersection / union;
    }

    private static double intersectionOverSmaller(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);
        if (x1 >= x2 || y1 >= y2) return 0.0;
        double intersection = (x2 - x1) * (double) (y2 - y1);
        double smaller = Math.min(area(a), area(b));
        if (smaller <= 0) return 0.0;
        return intersection / smaller;
    }

    private static double centerDistance(Rect a, Rect b) {
        double ax = a.x + a.width / 2.0;
        double ay = a.y + a.height / 2.0;
        double bx = b.x + b.width / 2.0;
        double by = b.y + b.height / 2.0;
        double dx = ax - bx;
        double dy = ay - by;
        return Math.sqrt(dx * dx + dy * dy);
    }
}
