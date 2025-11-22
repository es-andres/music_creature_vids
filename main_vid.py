import os
import cv2
import numpy as np
import random
from scipy.signal import find_peaks
# MoviePy 2.0 imports
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================

# Directory Structure
DIR_SRC = "data/music_vid_src"
DIR_PICKED = "data/music_vid_picked"
DIR_ALTERED = "data/music_vid_altered"
DIR_BG = "data/img/backgrounds"

# STEP 1: Snippet Extraction Parameters
SNIPPET_LENGTH_SEC = 60.0  # Length of each cut
MAX_SNIPPETS = 9  # Max number of clips to extract
SMOOTHING_WINDOW = 15  # Window size to smooth movement data (Higher = less twitchy)

# STEP 2: Object Detection & Random Walk Parameters
WALK_EVENTS_LO = 5  # Min number of "glitch events" per video
WALK_EVENTS_HI = 8  # Max number of "glitch events" per video

OBJECTS_PER_EVENT_LO = 1  # Min concurrent objects moving
OBJECTS_PER_EVENT_HI = 5  # Max concurrent objects moving

MIN_OBJ_SIZE_PERCENT = 0.01  # Object must take up 15% of screen

# [NEW] Event Speed Range (Pixels per frame)
# The code will pick a random 'Speed Limit' for EACH event between these two values.
EVENT_SPEED_MIN = 5  # "Chill" event (slow drift)
EVENT_SPEED_MAX = 20  # "Chaotic" event (fast glitch)

# Target Classes for YOLO (COCO dataset indices)
TARGET_CLASSES = [0, 66, 63]  # Person, Keyboard, Remote (Sampler)


class SnippetExtractor:
    def __init__(self):
        self.ensure_dirs()

    def ensure_dirs(self):
        """Ensure output directories exist."""
        os.makedirs(DIR_PICKED, exist_ok=True)

    def get_video_files(self):
        """Get valid video files from source."""
        valid_exts = ('.mov', '.mp4', '.avi')
        return [f for f in os.listdir(DIR_SRC) if f.lower().endswith(valid_exts)]

    def analyze_movement(self, video_path):
        """
        Calculates movement using Frame Differencing (Optimized for speed).
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, prev_frame = cap.read()
        if not ret: return []

        # Aggressive resize for analysis
        analyze_width = 320
        h, w = prev_frame.shape[:2]
        aspect = h / w
        analyze_height = int(analyze_width * aspect)

        prev_frame_resized = cv2.resize(prev_frame, (analyze_width, analyze_height))
        prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        movement_scores = []
        print(f"Analyzing movement in: {os.path.basename(video_path)}...")

        step = 2

        with tqdm(total=total_frames, unit="fr", desc="Scanning") as pbar:
            for _ in range(0, total_frames, step):
                for _ in range(step):
                    ret, frame = cap.read()

                if not ret: break

                frame_resized = cv2.resize(frame, (analyze_width, analyze_height))
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

                score = np.sum(thresh) / (analyze_width * analyze_height * 255)

                for _ in range(step):
                    movement_scores.append(score)

                prev_gray = gray
                pbar.update(step)

        cap.release()
        return np.array(movement_scores[:total_frames])

    def process(self):
        """Main execution for Step 1."""
        files = self.get_video_files()

        for filename in files:
            full_path = os.path.join(DIR_SRC, filename)
            scores = self.analyze_movement(full_path)

            if len(scores) == 0: continue

            # Smooth the signal
            box_pts = SMOOTHING_WINDOW
            box = np.ones(box_pts) / box_pts
            smooth_scores = np.convolve(scores, box, mode='same')

            # Find Peaks
            fps = 30
            distance_frames = int(SNIPPET_LENGTH_SEC * fps * 0.9)

            peaks, _ = find_peaks(smooth_scores, distance=distance_frames)

            # --- INTENSITY RANKING LOGIC ---
            peak_values = [smooth_scores[p] for p in peaks]
            peak_data = list(zip(peaks, peak_values))

            peak_data.sort(key=lambda x: x[1], reverse=True)
            top_peaks = peak_data[:MAX_SNIPPETS]

            # Keep them in Intensity Order (Index 0 = Most Intense)
            final_peaks_ordered = [p[0] for p in top_peaks]

            print(f"Selected Top {len(final_peaks_ordered)} moments (Ranked by Intensity).")

            clip = VideoFileClip(full_path)
            duration = clip.duration

            for i, peak_frame in enumerate(final_peaks_ordered):
                center_time = peak_frame / fps
                start_time = max(0, center_time - (SNIPPET_LENGTH_SEC / 2))
                end_time = min(duration, start_time + SNIPPET_LENGTH_SEC)

                if end_time - start_time < SNIPPET_LENGTH_SEC:
                    start_time = max(0, end_time - SNIPPET_LENGTH_SEC)

                subclip = clip.subclipped(start_time, end_time)
                base_name = os.path.splitext(filename)[0]

                out_name = f"{base_name}_{i + 1}.mov"
                out_path = os.path.join(DIR_PICKED, out_name)

                if not os.path.exists(out_path):
                    print(f"Saving {out_name} (Intensity Rank: {i + 1})...")
                    subclip.write_videofile(out_path, codec="libx264", audio_codec="aac",
                                            logger=None)

            clip.close()


class GlitchProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')  # Load segmentation model
        self.ensure_dirs()
        self.bg_image = self.load_background()

    def ensure_dirs(self):
        os.makedirs(DIR_ALTERED, exist_ok=True)

    def load_background(self):
        """Loads the single background image ."""
        try:
            files = [f for f in os.listdir(DIR_BG) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not files:
                print("No background image found in data/backgrounds")
                return None
            return cv2.imread(os.path.join(DIR_BG, files[0]))
        except Exception as e:
            print(f"Error loading background: {e}")
            return None

    def get_random_crop(self, shape):
        """Returns a random crop of the background image ."""
        if self.bg_image is None:
            return np.zeros(shape, dtype=np.uint8)

        h_req, w_req = shape[:2]
        h_bg, w_bg = self.bg_image.shape[:2]

        if h_bg < h_req or w_bg < w_req:
            scale = max(h_req / h_bg, w_req / w_bg)
            bg_resized = cv2.resize(self.bg_image, (0, 0), fx=scale, fy=scale)
            h_bg, w_bg = bg_resized.shape[:2]
        else:
            bg_resized = self.bg_image

        x = random.randint(0, w_bg - w_req)
        y = random.randint(0, h_bg - h_req)
        return bg_resized[y:y + h_req, x:x + w_req]

    def split_body_parts(self, mask, box):
        """Segments body into parts ."""
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1

        head_limit = y1 + int(h * 0.2)
        torso_limit = y1 + int(h * 0.6)

        parts = []

        # Head
        head_mask = np.zeros_like(mask)
        head_mask[y1:head_limit, x1:x2] = mask[y1:head_limit, x1:x2]
        if np.sum(head_mask) > 0: parts.append(head_mask)

        # Torso
        torso_mask = np.zeros_like(mask)
        torso_mask[head_limit:torso_limit, x1:x2] = mask[head_limit:torso_limit, x1:x2]
        if np.sum(torso_mask) > 0: parts.append(torso_mask)

        # Hands/Lower
        hands_mask = np.zeros_like(mask)
        hands_mask[torso_limit:y2, x1:x2] = mask[torso_limit:y2, x1:x2]
        if np.sum(hands_mask) > 0: parts.append(hands_mask)

        return parts

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(DIR_ALTERED, f"{base_name}_alt.mov")

        temp_out = os.path.join("cache", "temp_video.mp4")
        if not os.path.exists("cache"): os.makedirs("cache")

        writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # --- SCHEDULING EVENTS ---
        num_events = random.randint(WALK_EVENTS_LO, WALK_EVENTS_HI)

        event_starts = []

        # [NEW LOGIC] Force one event in the first 3 seconds
        first_3_sec_frames = int(fps * 3)
        if first_3_sec_frames > total_frames - 30:
            first_3_sec_frames = max(0, total_frames - 30)

        # Add the mandatory early event
        event_starts.append(random.randint(0, max(1, first_3_sec_frames)))

        # Generate remaining events randomly throughout the video
        remaining_events = num_events - 1
        if remaining_events > 0:
            for _ in range(remaining_events):
                event_starts.append(random.randint(0, max(1, total_frames - 30)))

        event_starts.sort()

        active_walkers = []

        print(
            f"Processing {base_name} with {num_events} glitch events (First one at frame {event_starts[0]})...")

        for frame_idx in tqdm(range(total_frames), desc="Rendering"):
            ret, frame = cap.read()
            if not ret: break

            # --- Start new event ---
            if frame_idx in event_starts:
                num_objs = random.randint(OBJECTS_PER_EVENT_LO, OBJECTS_PER_EVENT_HI)

                # Determine the "Vibe" (Max Speed) for THIS specific event
                this_event_max_speed = random.randint(EVENT_SPEED_MIN, EVENT_SPEED_MAX)

                results = self.model(frame, classes=TARGET_CLASSES, verbose=False)[0]

                current_candidates = []

                if results.masks:
                    for i, mask_data in enumerate(results.masks.data):
                        mask_np = mask_data.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (width, height))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        # Size check
                        if (np.sum(mask_binary) / (width * height)) < MIN_OBJ_SIZE_PERCENT:
                            continue

                        cls_id = int(results.boxes.cls[i])

                        # Body parts exception
                        if cls_id == 0:
                            parts = self.split_body_parts(mask_binary, results.boxes.xyxy[i])
                            for p in parts:
                                current_candidates.append(p)
                        else:
                            current_candidates.append(mask_binary)

                if current_candidates:
                    selected_masks = random.sample(current_candidates,
                                                   min(num_objs, len(current_candidates)))

                    for m in selected_masks:
                        # Clipping from image file
                        bg_crop = self.get_random_crop((height, width))

                        # Set velocity based on THIS event's speed cap
                        safe_speed = max(2, this_event_max_speed)
                        dx = random.choice([-1, 1]) * random.randint(2, safe_speed)
                        dy = random.choice([-1, 1]) * random.randint(2, safe_speed)

                        walker = {
                            'mask_full': m,
                            'orig_crop': frame.copy(),
                            'bg_texture': bg_crop,
                            'curr_x': 0,
                            'curr_y': 0,
                            'dx': dx,
                            'dy': dy,
                            'remaining': random.randint(30, 240),  # Walk length
                            'speed_cap': safe_speed
                        }
                        active_walkers.append(walker)

            # --- Render Frame ---
            final_frame = frame.copy()

            for i in range(len(active_walkers) - 1, -1, -1):
                w_obj = active_walkers[i]

                # Random walk logic
                w_obj['curr_x'] += w_obj['dx']
                w_obj['curr_y'] += w_obj['dy']

                if random.random() > 0.8:
                    w_obj['dx'] = random.choice([-1, 1]) * random.randint(2, w_obj['speed_cap'])
                if random.random() > 0.8:
                    w_obj['dy'] = random.choice([-1, 1]) * random.randint(2, w_obj['speed_cap'])

                # Place random clipping in ORIGINAL position
                mask_indices = w_obj['mask_full'] > 0
                final_frame[mask_indices] = w_obj['bg_texture'][mask_indices]

                # Displace object but show changes (using captured frame)
                # Appear from other edge (wrapping)
                shift_x = int(w_obj['curr_x'])
                shift_y = int(w_obj['curr_y'])

                shifted_mask = np.roll(w_obj['mask_full'], shift_y, axis=0)
                shifted_mask = np.roll(shifted_mask, shift_x, axis=1)

                moved_content = np.roll(w_obj['orig_crop'], shift_y, axis=0)
                moved_content = np.roll(moved_content, shift_x, axis=1)

                moved_indices = shifted_mask > 0
                final_frame[moved_indices] = moved_content[moved_indices]

                w_obj['remaining'] -= 1
                if w_obj['remaining'] <= 0:
                    # Objects come back to origin (pop removes the effect)
                    active_walkers.pop(i)

            writer.write(final_frame)

        cap.release()
        writer.release()

        print("Merging audio...")
        original_clip = VideoFileClip(video_path)
        processed_clip = VideoFileClip(temp_out)

        processed_clip = processed_clip.with_audio(original_clip.audio)
        processed_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)

        if os.path.exists(temp_out):
            os.remove(temp_out)
        print(f"Finished {out_path}")

    def process_all(self):
        files = [f for f in os.listdir(DIR_PICKED) if f.lower().endswith('.mov')]
        for f in files:
            self.process_video(os.path.join(DIR_PICKED, f))


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting Music Video Automator ---")

    # Step 1: Extract Snippets
    print("\n[Step 1] Extracting interesting snippets based on movement...")
    extractor = SnippetExtractor()
    extractor.process()

    # Step 2: Apply Glitch/Walk Effects
    print("\n[Step 2] Applying Object Detection and Random Walks...")
    processor = GlitchProcessor()
    processor.process_all()

    print("\nDone.")