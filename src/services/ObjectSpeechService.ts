import * as Speech from 'expo-speech';
import { Detection } from './yoloInference';

export interface ObjectSpeechServiceOptions {
  confidenceThreshold?: number;
  sameClassCooldownMs?: number;
  globalCooldownMs?: number;
  interruptPriorityDelta?: number;
  rate?: number;
  pitch?: number;
  language?: string;
}

type DirectionLabel = 'left' | 'ahead' | 'right';

interface AnnouncementCandidate {
  className: string;
  confidence: number;
  area: number;
  direction: DirectionLabel;
  priority: number;
  message: string;
}

interface ActiveSpeechState {
  className: string;
  priority: number;
}

const DEFAULT_OPTIONS: Required<ObjectSpeechServiceOptions> = {
  confidenceThreshold: 0.45,
  sameClassCooldownMs: 4000,
  globalCooldownMs: 1200,
  interruptPriorityDelta: 0.18,
  rate: 0.98,
  pitch: 1.0,
  language: 'en-US',
};

/**
 * Announces object detections through Expo Speech with anti-spam controls.
 * Candidate ranking emphasizes bounding-box area (proximity proxy) and confidence.
 */
export class ObjectSpeechService {
  private readonly options: Required<ObjectSpeechServiceOptions>;
  // Tracks the last announcement timestamp per class label.
  private readonly lastClassAnnouncementMs = new Map<string, number>();
  // Tracks the last announcement timestamp across all classes.
  private lastAnnouncementMs = 0;
  // Holds currently active speech metadata for interrupt decisions.
  private activeSpeech: ActiveSpeechState | null = null;
  // Guards against stale callback races after speech interruption/restart.
  private activeSpeechToken = 0;
  private enabled = true;

  constructor(options?: ObjectSpeechServiceOptions) {
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
    };
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;

    if (!enabled) {
      void Speech.stop();
      this.activeSpeech = null;
    }
  }

  updateOptions(options: Partial<ObjectSpeechServiceOptions>): void {
    this.options.confidenceThreshold = options.confidenceThreshold ?? this.options.confidenceThreshold;
    this.options.sameClassCooldownMs = options.sameClassCooldownMs ?? this.options.sameClassCooldownMs;
    this.options.globalCooldownMs = options.globalCooldownMs ?? this.options.globalCooldownMs;
    this.options.interruptPriorityDelta = options.interruptPriorityDelta ?? this.options.interruptPriorityDelta;
    this.options.rate = options.rate ?? this.options.rate;
    this.options.pitch = options.pitch ?? this.options.pitch;
    this.options.language = options.language ?? this.options.language;
  }

  async announceDetections(detections: Detection[]): Promise<void> {
    // Fast exit when disabled or when no detections are available.
    if (!this.enabled || detections.length === 0) {
      return;
    }

    // Selects the highest-priority candidate after threshold filtering.
    const candidate = this.pickBestCandidate(detections);
    if (!candidate) {
      return;
    }

    const now = Date.now();
    // Applies per-class and global cooldown windows.
    if (!this.passesCooldowns(candidate.className, now)) {
      return;
    }

    const speaking = await Speech.isSpeakingAsync().catch(() => false);
    if (speaking) {
      // Interrupts only when the incoming candidate materially exceeds current priority.
      if (!this.shouldInterrupt(candidate)) {
        return;
      }

      await Speech.stop().catch(() => undefined);
    }

    this.lastClassAnnouncementMs.set(candidate.className, now);
    this.lastAnnouncementMs = now;
    this.activeSpeech = {
      className: candidate.className,
      priority: candidate.priority,
    };
    const token = ++this.activeSpeechToken;

    Speech.speak(candidate.message, {
      language: this.options.language,
      rate: this.options.rate,
      pitch: this.options.pitch,
      onDone: () => {
        if (this.activeSpeechToken === token) {
          this.activeSpeech = null;
        }
      },
      onStopped: () => {
        if (this.activeSpeechToken === token) {
          this.activeSpeech = null;
        }
      },
      onError: () => {
        if (this.activeSpeechToken === token) {
          this.activeSpeech = null;
        }
      },
    });
  }

  async stop(): Promise<void> {
    await Speech.stop().catch(() => undefined);
    this.activeSpeechToken += 1;
    this.activeSpeech = null;
  }

  async dispose(): Promise<void> {
    await this.stop();
    this.lastClassAnnouncementMs.clear();
    this.lastAnnouncementMs = 0;
  }

  private pickBestCandidate(detections: Detection[]): AnnouncementCandidate | null {
    let bestCandidate: AnnouncementCandidate | null = null;

    for (const detection of detections) {
      const confidence = detection.confidence;
      // Confidence gate suppresses low-certainty detections.
      if (!Number.isFinite(confidence) || confidence <= this.options.confidenceThreshold) {
        continue;
      }

      const { x, width, height } = detection.boundingBox;
      if (!Number.isFinite(x) || !Number.isFinite(width) || !Number.isFinite(height)) {
        continue;
      }

      if (width <= 0 || height <= 0) {
        continue;
      }

      const area = width * height;
      const direction = this.getDirectionFromBounds(x, x + width);
      // Area is normalized against a quarter-frame reference and combined with confidence.
      const proximityScore = Math.max(0, Math.min(1, area / 0.25));
      const priority = (proximityScore * 0.8) + (confidence * 0.2);

      const className = this.toSpokenClassName(detection.className);
      const message = direction === 'ahead'
        ? className + ' ahead'
        : className + ' on the ' + direction;

      if (!bestCandidate || priority > bestCandidate.priority) {
        bestCandidate = {
          className: detection.className,
          confidence,
          area,
          direction,
          priority,
          message,
        };
      }
    }

    return bestCandidate;
  }

  private getDirectionFromBounds(xMin: number, xMax: number): DirectionLabel {
    // Direction is computed from normalized horizontal center point.
    const clampedMin = Math.max(0, Math.min(1, xMin));
    const clampedMax = Math.max(0, Math.min(1, xMax));
    const center = (clampedMin + clampedMax) * 0.5;

    if (center <= 0.33) {
      return 'left';
    }

    if (center >= 0.67) {
      return 'right';
    }

    return 'ahead';
  }

  private passesCooldowns(className: string, now: number): boolean {
    const classLastSpokenAt = this.lastClassAnnouncementMs.get(className) ?? 0;

    // Per-class cooldown prevents repeated announcements of the same object.
    const classCooldownPassed = (now - classLastSpokenAt) >= this.options.sameClassCooldownMs;
    if (!classCooldownPassed) {
      return false;
    }

    // Global cooldown prevents rapid speech churn across different classes.
    const globalCooldownPassed = (now - this.lastAnnouncementMs) >= this.options.globalCooldownMs;
    if (!globalCooldownPassed) {
      return false;
    }

    return true;
  }

  private shouldInterrupt(candidate: AnnouncementCandidate): boolean {
    if (!this.activeSpeech) {
      return true;
    }

    // Same-class interruptions are skipped to preserve stable speech cadence.
    if (candidate.className === this.activeSpeech.className) {
      return false;
    }

    return candidate.priority >= (this.activeSpeech.priority + this.options.interruptPriorityDelta);
  }

  private toSpokenClassName(rawClassName: string): string {
    const cleaned = rawClassName
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (!cleaned) {
      return 'Object';
    }

    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
  }
}
