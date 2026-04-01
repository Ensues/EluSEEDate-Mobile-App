/**
 * Bounding Box Overlay Component
 * 
 * Renders YOLO detection bounding boxes on top of camera view
 * Shows class labels and confidence scores
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Detection } from '../services/yoloInference';

const FULL_FRAME_ROI = {
  x: 0,
  y: 0,
  width: 1,
  height: 1,
};

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

interface BoundingBoxOverlayProps {
  detections: Detection[];
  containerWidth: number;
  containerHeight: number;
}

export default function BoundingBoxOverlay({ 
  detections, 
  containerWidth, 
  containerHeight 
}: BoundingBoxOverlayProps) {
  if (detections.length === 0) {
    return null;
  }

  return (
    <View style={styles.container} pointerEvents="none">
      {detections.map((detection, index) => {
        const { boundingBox, className, confidence } = detection;
        const roi = detection.roi ?? FULL_FRAME_ROI;

        // Re-map ROI-relative YOLO coordinates to full-frame normalized coordinates.
        // x_full = roi.x + (x_roi * roi.width), y_full = roi.y + (y_roi * roi.height)
        // w_full = w_roi * roi.width, h_full = h_roi * roi.height
        const leftNorm = clamp01(roi.x + (boundingBox.x * roi.width));
        const topNorm = clamp01(roi.y + (boundingBox.y * roi.height));
        const rightNorm = clamp01(roi.x + ((boundingBox.x + boundingBox.width) * roi.width));
        const bottomNorm = clamp01(roi.y + ((boundingBox.y + boundingBox.height) * roi.height));

        if (rightNorm <= leftNorm || bottomNorm <= topNorm) {
          return null;
        }
        
        // Convert normalized coordinates to pixel coordinates
        const left = leftNorm * containerWidth;
        const top = topNorm * containerHeight;
        const width = (rightNorm - leftNorm) * containerWidth;
        const height = (bottomNorm - topNorm) * containerHeight;
        
        return (
          <View
            key={`detection-${index}`}
            style={[
              styles.boundingBox,
              {
                left,
                top,
                width,
                height,
              }
            ]}
          >
            {/* Label background */}
            <View style={styles.labelContainer}>
              <Text style={styles.labelText}>
                {className} {(confidence * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 50, // Above camera, below UI overlays
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 3,
    borderColor: '#00FF00', // Green border for detections
    borderRadius: 4,
  },
  labelContainer: {
    position: 'absolute',
    top: -24,
    left: 0,
    backgroundColor: '#00FF00',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  labelText: {
    color: '#000000',
    fontSize: 12,
    fontWeight: 'bold',
  },
});
