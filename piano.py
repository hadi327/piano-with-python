import cv2
import pygame
import numpy as np
import time
import math
from collections import deque


class ElegantVirtualPiano:
    def __init__(self):
        # Initialize Pygame for audio with optimal settings
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

        # Two octaves of piano notes
        self.octave_notes = {
            'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
            'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
            'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
        }

        # Create a more limited set for cleaner interface
        self.active_notes = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4']

        # Audio management
        self.note_sounds = {}
        self.active_channels = {}
        self.create_high_quality_sounds()

        # Visual settings
        self.window_width = 1200
        self.window_height = 800
        self.key_width = self.window_width // len(self.active_notes)
        self.key_height = 200

        # Colors (elegant palette)
        self.colors = {
            'background': (25, 25, 35),
            'white_key': (245, 245, 245),
            'white_key_active': (100, 200, 255),
            'black_key': (40, 40, 50),
            'black_key_active': (80, 160, 255),
            'text': (220, 220, 220),
            'text_active': (255, 255, 255),
            'ui_accent': (70, 130, 180),
            'finger_trail': (255, 215, 0)
        }

        # Finger tracking
        self.finger_trails = deque(maxlen=20)
        self.note_cooldown = 0.15

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.setup_camera()

        # Visual effects
        self.wave_effects = []

        print("🎹 Elegant Virtual Piano Initialized")
        print("=" * 50)

    def setup_camera(self):
        """Configure camera for better performance"""
        if not self.cap.isOpened():
            print("❌ Error: Could not access webcam")
            return False

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Background subtractor with tuned parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=True
        )

        return True

    def create_high_quality_sounds(self):
        """Create professional-quality piano sounds"""
        sample_rate = 44100

        for note_name in self.active_notes:
            freq = self.octave_notes[note_name]
            duration = 3.0  # Longer duration for sustain

            # Generate time array
            t = np.linspace(0, duration, int(duration * sample_rate), False)

            # Complex harmonic structure for rich piano sound
            harmonics = [
                (1.00, 1.0),  # Fundamental
                (2.00, 0.6),  # Octave
                (3.00, 0.4),  # Fifth
                (4.00, 0.2),  # Double octave
                (5.00, 0.1),  # Major third
                (6.00, 0.05),  # Fifth
            ]

            # Generate sound wave
            sound_wave = np.zeros_like(t)
            for harmonic, amplitude in harmonics:
                harmonic_freq = freq * harmonic
                wave = amplitude * np.sin(2 * np.pi * harmonic_freq * t)

                # Add slight detune for realism
                if harmonic > 1:
                    detune = 1 + (np.random.random() - 0.5) * 0.001
                    wave = amplitude * np.sin(2 * np.pi * harmonic_freq * detune * t)

                sound_wave += wave

            # Apply sophisticated envelope
            envelope = self.create_piano_envelope(len(t), sample_rate)
            sound_wave *= envelope

            # Add subtle chorus effect
            chorus = 0.05 * np.sin(2 * np.pi * 0.5 * t)
            sound_wave += chorus

            # Normalize and convert to stereo
            sound_wave = sound_wave / np.max(np.abs(sound_wave))
            stereo_sound = np.column_stack((sound_wave, sound_wave))

            # Convert to pygame sound
            sound_16bit = (stereo_sound * 32767).astype(np.int16)
            self.note_sounds[note_name] = pygame.sndarray.make_sound(sound_16bit)

    def create_piano_envelope(self, num_frames, sample_rate):
        """Create a realistic piano ADSR envelope"""
        envelope = np.ones(num_frames)

        # Time parameters (in seconds)
        attack = 0.01  # Quick attack
        decay = 0.1  # Fast decay
        sustain = 2.0  # Long sustain
        release = 0.9  # Smooth release

        attack_frames = int(attack * sample_rate)
        decay_frames = int(decay * sample_rate)
        sustain_frames = int(sustain * sample_rate)
        release_frames = int(release * sample_rate)

        # Ensure we don't exceed array bounds
        total_frames = min(num_frames, attack_frames + decay_frames + sustain_frames + release_frames)

        # Attack phase (exponential)
        if attack_frames > 0:
            envelope[:attack_frames] = 1 - np.exp(-5 * np.linspace(0, 1, attack_frames))

        # Decay to sustain level
        if attack_frames + decay_frames <= total_frames:
            sustain_level = 0.7
            decay_end = attack_frames + decay_frames
            envelope[attack_frames:decay_end] = sustain_level + (1 - sustain_level) * np.exp(
                -2 * np.linspace(0, 1, decay_frames))

        # Sustain phase
        sustain_start = attack_frames + decay_frames
        sustain_end = min(sustain_start + sustain_frames, total_frames)
        envelope[sustain_start:sustain_end] = sustain_level

        # Release phase
        release_start = sustain_end
        release_end = min(release_start + release_frames, total_frames)
        if release_end > release_start:
            release_env = np.linspace(sustain_level, 0, release_end - release_start)
            envelope[release_start:release_end] = release_env

        return envelope

    def detect_elegant_hands(self, frame):
        """Sophisticated hand detection with smooth tracking"""
        # Resize frame for processing
        processed_frame = cv2.resize(frame, (320, 240))

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(processed_frame)

        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        finger_positions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 20000:  # Reasonable hand size range
                # Get convex hull for better finger detection
                hull = cv2.convexHull(contour)

                # Find extreme points
                if len(hull) >= 5:
                    # Get top points (potential fingers)
                    for i in range(min(5, len(hull))):
                        point = hull[i][0]
                        # Scale back to original coordinates
                        x = int(point[0] * frame.shape[1] / 320)
                        y = int(point[1] * frame.shape[0] / 240)
                        finger_positions.append((x, y))

        return finger_positions, fg_mask

    def draw_sleek_piano(self, frame, active_notes):
        """Draw an elegant piano interface"""
        height, width = frame.shape[:2]

        # Draw piano background
        cv2.rectangle(frame, (0, height - self.key_height),
                      (width, height), self.colors['background'], -1)

        # Draw individual keys
        for i, note in enumerate(self.active_notes):
            x_start = i * self.key_width
            x_end = (i + 1) * self.key_width
            y_start = height - self.key_height
            y_end = height

            # Determine key color
            if note in active_notes:
                key_color = self.colors['white_key_active']
                border_color = self.colors['ui_accent']
            else:
                key_color = self.colors['white_key']
                border_color = self.colors['black_key']

            # Draw key with shadow effect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), key_color, -1)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), border_color, 3)

            # Draw note label with elegant typography
            label = note.replace('3', '₃').replace('4', '₄')
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            text_x = x_start + (self.key_width - text_size[0]) // 2
            text_y = height - self.key_height // 2 + 10

            text_color = self.colors['text_active'] if note in active_notes else self.colors['text']
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2)

    def draw_finger_elegance(self, frame, finger_positions, active_notes):
        """Draw elegant finger visualization with trails"""
        height, width = frame.shape[:2]

        # Add current positions to trails
        current_time = time.time()
        for pos in finger_positions:
            self.finger_trails.append((pos[0], pos[1], current_time))

        # Draw fading trails
        for i, (x, y, trail_time) in enumerate(self.finger_trails):
            alpha = 1.0 - (current_time - trail_time) * 2
            if alpha > 0:
                # Calculate trail color (fades from gold to transparent)
                trail_color = tuple(int(c * alpha) for c in self.colors['finger_trail'])
                radius = max(3, int(8 * alpha))

                cv2.circle(frame, (x, y), radius, trail_color, -1)

        # Draw current finger positions
        for x, y in finger_positions:
            # Draw elegant finger circles
            cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)  # White core
            cv2.circle(frame, (x, y), 12, self.colors['ui_accent'], 2)  # Blue border
            cv2.circle(frame, (x, y), 8, self.colors['finger_trail'], -1)  # Gold center

            # Draw connection line if near piano
            if y > height - self.key_height - 50:
                piano_y = height - self.key_height
                cv2.line(frame, (x, y), (x, piano_y), self.colors['ui_accent'], 2)

                # Show note being played
                note = self.get_note_from_position(x, width)
                if note and note in active_notes:
                    cv2.putText(frame, f"♪ {note}", (x - 25, y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['ui_accent'], 2)

    def draw_elegant_ui(self, frame, finger_count, active_note_count):
        """Draw sophisticated user interface"""
        height, width = frame.shape[:2]

        # Semi-transparent header
        header = np.zeros((80, width, 3), dtype=np.uint8)
        header[:, :] = self.colors['background']
        frame[0:80, 0:width] = cv2.addWeighted(frame[0:80, 0:width], 0.3, header, 0.7, 0)

        # Application title
        title = "Elegant Virtual Piano"
        cv2.putText(frame, title, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, self.colors['text_active'], 2)

        # Status information
        status_y = 70
        status_items = [
            f"Fingers: {finger_count}",
            f"Active Notes: {active_note_count}",
            f"Octave: 3-4"
        ]

        for i, item in enumerate(status_items):
            cv2.putText(frame, item, (width - 200, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)

        # Instructions footer
        instructions = [
            "Move hands above keys to play • Multiple fingers create chords",
            "Q: Quit • R: Reset • C: Clear notes • Space: Toggle view"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, height - 20 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

    def get_note_from_position(self, x, width):
        """Map finger position to piano note"""
        key_index = x // self.key_width
        if 0 <= key_index < len(self.active_notes):
            return self.active_notes[key_index]
        return None

    def play_note_elegantly(self, note):
        """Play note with professional audio management"""
        if note in self.note_sounds:
            # Stop previous instance if playing
            if note in self.active_channels:
                self.active_channels[note].stop()

            # Play with fade-in for smoothness
            channel = self.note_sounds[note].play()
            channel.set_volume(0.8)  # Slightly reduced volume for elegance
            self.active_channels[note] = channel
            return True
        return False

    def stop_note_gracefully(self, note):
        """Stop note with smooth fade-out"""
        if note in self.active_channels:
            self.active_channels[note].fadeout(300)  # 300ms fadeout
            return True
        return False

    def run_elegant_piano(self):
        """Main application loop"""
        if not self.setup_camera():
            return

        print("🎹 Starting Elegant Virtual Piano...")
        print("✨ Features:")
        print("   • Professional piano sound quality")
        print("   • Elegant visual interface")
        print("   • Smooth finger tracking with trails")
        print("   • Chord support with multiple fingers")
        print("   • Real-time audio synthesis")
        print("=" * 50)

        active_notes = set()
        last_note_time = {}
        show_debug = False

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera feed lost")
                    break

                # Flip and resize frame
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (self.window_width, self.window_height))

                # Detect hands
                finger_positions, debug_mask = self.detect_elegant_hands(frame)

                # Update active notes
                current_notes = set()
                for x, y in finger_positions:
                    note = self.get_note_from_position(x, self.window_width)
                    if note and y > self.window_height - self.key_height - 30:
                        current_notes.add(note)

                # Handle note transitions
                for note in current_notes - active_notes:
                    current_time = time.time()
                    if note not in last_note_time or current_time - last_note_time[note] > self.note_cooldown:
                        if self.play_note_elegantly(note):
                            print(f"🎵 {note} ({self.octave_notes[note]:.1f} Hz)")
                            last_note_time[note] = current_time

                for note in active_notes - current_notes:
                    self.stop_note_gracefully(note)

                active_notes = current_notes

                # Draw interface
                self.draw_sleek_piano(frame, active_notes)
                self.draw_finger_elegance(frame, finger_positions, active_notes)
                self.draw_elegant_ui(frame, len(finger_positions), len(active_notes))

                # Display window
                cv2.imshow('Elegant Virtual Piano 🎹', frame)
                if show_debug:
                    cv2.imshow('Debug View', debug_mask)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
                    print("🔄 Background reset")
                elif key == ord('c'):
                    for note in active_notes.copy():
                        self.stop_note_gracefully(note)
                    active_notes.clear()
                    print("🧹 All notes cleared")
                elif key == ord(' '):
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow('Debug View')

        except KeyboardInterrupt:
            print("\n🛑 Piano stopped gracefully")

        finally:
            # Graceful shutdown
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("🎹 Thank you for playing!")


def main():
    """Application entry point"""
    piano = ElegantVirtualPiano()
    piano.run_elegant_piano()


if __name__ == "__main__":
    main()