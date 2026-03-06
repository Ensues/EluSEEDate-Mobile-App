/**
 * Bounding Box Overlay Component
 * 
 * Renders YOLO detection bounding boxes on top of camera view
 * Shows class labels and confidence scores
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Detection } from '../services/yoloInference';

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
        
        // Convert normalized coordinates to pixel coordinates
        const left = boundingBox.x * containerWidth;
        const top = boundingBox.y * containerHeight;
        const width = boundingBox.width * containerWidth;
        const height = boundingBox.height * containerHeight;
        
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
