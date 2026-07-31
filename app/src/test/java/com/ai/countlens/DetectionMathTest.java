package com.ai.countlens;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DetectionMathTest {
    private static final double EPSILON = 1e-6;

    @Test
    public void identicalBoxesHavePerfectOverlap() {
        assertEquals(1.0, DetectionMath.intersectionOverUnion(0, 0, 10, 10, 0, 0, 10, 10), EPSILON);
        assertEquals(1.0, DetectionMath.containmentRatio(0, 0, 10, 10, 0, 0, 10, 10), EPSILON);
    }

    @Test
    public void disjointBoxesHaveNoOverlap() {
        assertEquals(0.0, DetectionMath.intersectionOverUnion(0, 0, 10, 10, 20, 20, 5, 5), EPSILON);
        assertEquals(0.0, DetectionMath.containmentRatio(0, 0, 10, 10, 20, 20, 5, 5), EPSILON);
    }

    @Test
    public void containmentDetectsNestedDuplicate() {
        assertEquals(1.0, DetectionMath.containmentRatio(0, 0, 20, 20, 5, 5, 5, 5), EPSILON);
        assertTrue(DetectionMath.intersectionOverUnion(0, 0, 20, 20, 5, 5, 5, 5) < 0.1);
    }

    @Test
    public void medianIgnoresInvalidValues() {
        assertEquals(3.0, DetectionMath.median(Arrays.asList(1.0, 3.0, 5.0, Double.NaN, -2.0)), EPSILON);
    }

    @Test
    public void robustVariationIsInsensitiveToOneLargeOutlier() {
        double variation = DetectionMath.robustCoefficientOfVariation(Arrays.asList(10.0, 10.5, 9.5, 10.2, 80.0));
        assertTrue(variation < 0.08);
    }
}
