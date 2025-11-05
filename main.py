import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from moviepy import ImageClip, CompositeVideoClip, AudioFileClip
from tqdm import tqdm

DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "img"
BCKGRND_DIR = IMG_DIR / "backgrounds"
STICKERS = IMG_DIR / "stickers"
VID_OUT = DATA_DIR / "vid_out"
AUDIO_DIR = DATA_DIR / "sound"
DURATION = 15

# --- NEW PARAMETER TO TWEAK ---
# This is the minimum size (in pixels) for the creature's height and width.
# The code will scale them up if their random size is smaller than this.
MIN_CREATURE_DIMENSION = 500
# --- END NEW PARAMETER ---

FUNCTIONS = [
    np.sin,
    np.cos,
    np.tan,
]
COLORS = [
    # "#03255",
    "#c73d0f",
    "#c42f20",
    "#cea25b",
    "#e8d6b3",
    "#a0a7b6",
    "#d8baa5",
    "#e7ca9e",
]

# list of matplotlib fonts rendered many of them as squares. whatever:
# /System/Library/Fonts/
FONTS = [
    # cursive
    "Apple Chancery.ttf",
    "MarkerFelt.ttc",
    "Noteworthy.ttc",
    "Comic Sans MS.ttf",
    "Trattatello.ttf",
    # not
    "Courier New Bold.ttf",
    "Avenir Next Condensed.ttc",
    "Geneva.ttf",
    "Keyboard.ttf",
    "Menlo.ttc",
]


class Creature:
    def __init__(self,
                 img,
                 x,
                 y,
                 back_clip,
                 audio,  # Added audio
                 wackiness,  # Added wackiness
                 x_add=None,
                 y_add=None,
                 ):
        # --- Original Movement Logic ---
        self.y_f = random.choice(FUNCTIONS)
        self.y_f_min = np.abs(self.y_f(0))
        self.y_f_max = np.abs(self.y_f(DURATION))
        self.x_f = random.choice(FUNCTIONS)
        self.x_f_min = np.abs(self.x_f(0))
        self.x_f_max = np.abs(self.x_f(DURATION))
        self.x_f_sub = np.random.uniform(0, DURATION)
        self.y_f_sub = np.random.uniform(0, DURATION)
        self.img = img  # This img clip has the *base* size
        self.x = x
        self.y = y
        self.back_clip = back_clip
        s = 300
        self.x_delta = back_clip.size[0] / np.random.normal(s, s * .15)
        self.y_delta = back_clip.size[1] / np.random.normal(s, s * .15)
        if not x_add:
            self.x_move = random.choice([np.add, np.subtract])
        if not y_add:
            self.y_move = random.choice([np.add, np.subtract])
        barrier_nums = [0]
        for draws in range(1, int(np.random.uniform(1, 1))):
            barrier_nums += [np.random.randint(0, 10)]
        barrier_nums = list(sorted(set(barrier_nums)))
        delta_nums = [0]
        for draws in range(1, int(np.random.uniform(5, 10))):
            delta_nums += [np.random.randint(0, 10)]
        delta_nums = list(sorted(set(delta_nums)))
        buffer = 0.05
        self.deltas = {
            0: {
                "x": self.x_delta,
                "y": self.y_delta
            }
        }
        self.barriers = {
            0: {
                "left": -self.back_clip.size[0] * buffer,
                "right": self.back_clip.size[0] * (1 - buffer),
                "top": -self.back_clip.size[1] * buffer,
                "bottom": self.back_clip.size[1] * (1 - buffer),
            }
        }
        for i in range(1, 11):
            if i in barrier_nums:
                x_barrier = self.get_barrier(side=0, buffer=0)
                y_barrier = self.get_barrier(side=1, buffer=0)
                self.barriers[i] = {
                    "left": x_barrier["low"],
                    "right": x_barrier["high"],
                    "top": y_barrier["low"],
                    "bottom": y_barrier["high"]
                }
            else:
                self.barriers[i] = self.barriers[0]
        for i in range(1, 11):
            if i in delta_nums:
                x_denom = np.random.normal(s, s * .25)
                y_denom = np.random.normal(s, s * .25)
                self.deltas[i] = {
                    "x": self.back_clip.size[0] / x_denom,
                    "y": self.back_clip.size[1] / y_denom
                }
            else:
                self.deltas[i] = self.deltas[0]

        # --- New Dancing & Audio Setup ---
        self.audio = audio
        self.wackiness = wackiness  # Store wackiness

        # New: Set a dance speed divisor (1=on beat, 2=half speed, 3=third speed)
        self.dance_speed_divisor = random.choice([2, 4, 8])  # User changed this

        # 1. Process audio to get an amplitude array
        sound_array = self.audio.to_soundarray(fps=48000)

        if sound_array.ndim == 2:  # Stereo
            amplitude = np.abs(sound_array.mean(axis=1))
        else:  # Mono
            amplitude = np.abs(sound_array)

        self.audio_fps = 48000

        max_amp = np.max(amplitude)
        if max_amp > 0:
            self.normalized_amplitude = amplitude / max_amp
        else:
            self.normalized_amplitude = amplitude  # All zeros

        self.audio_len = len(self.normalized_amplitude)

        # 2. Define dance parameters based on wackiness
        # Scale (pulse and audio reaction)
        self.scale_pulse_speed = random.uniform(3, 10) * (1 + self.wackiness * 2)
        self.scale_pulse_amount = random.uniform(0.05, 0.2) * self.wackiness
        self.scale_audio_sensitivity = random.uniform(0.1, 0.4) * (1 + self.wackiness)

        # Rotation (base speed and audio jiggle)
        self.base_rotation_speed = random.uniform(-60, 60) * self.wackiness
        self.rot_jiggle_speed = random.uniform(10, 30)
        self.rot_audio_sensitivity = random.uniform(10, 60) * self.wackiness  # Max jiggle degrees

        # Position (audio jiggle)
        self.pos_jitter_freq = random.uniform(5, 20)
        self.pos_jitter_amount = random.uniform(10, 50) * self.wackiness * (1 + self.wackiness)

        # --- NEW ROBUST CLAMPING FIX ---
        # Pre-calculate the maximum possible size of the creature
        # 1. Max scale from pulsing and audio reaction
        max_scale_factor = 1.0 + self.scale_pulse_amount + self.scale_audio_sensitivity
        # 2. Max expansion from rotation (sqrt(2) is ~1.414)
        max_rotation_expansion = 1.415  # Use a safe margin

        # Get the base size (e.g., 500x500)
        base_w, base_h = self.img.size

        # Calculate max possible width and height
        self.max_w = base_w * max_scale_factor * max_rotation_expansion
        self.max_h = base_h * max_scale_factor * max_rotation_expansion
        # --- END NEW FIX ---

    def get_amplitude_at(self, t):
        """Gets normalized amplitude (0-1) at time t."""
        idx = int(t * self.audio_fps)
        # Clamp index to be within bounds
        idx = max(0, min(idx, self.audio_len - 1))
        return self.normalized_amplitude[idx]

    def get_barrier(self, side, buffer):
        barrier_start = (
                self.img.size[side] +
                np.random.uniform(self.img.size[side],
                                  self.back_clip.size[side] - self.img.size[side]
                                  )
        )
        return {
            "low": barrier_start,
            "high": min(
                (
                        barrier_start +
                        np.random.uniform(self.img.size[side], self.back_clip.size[side] * .9)
                ),
                self.back_clip.size[side] * (1 + buffer)
            )
        }

    def scale_x_f(self, t):
        if self.x_f not in [np.cos, np.tan, np.sin]:
            res = (np.abs(self.x_f_sub - self.x_f(t)) - self.x_f_min) / (
                    self.x_f_max - self.x_f_min)
        else:
            res = min(np.abs(self.x_f_sub - self.x_f(t)), 1)
        return res

    def scale_y_f(self, t):
        if self.x_f not in [np.cos, np.tan, np.sin]:
            res = (np.abs(self.y_f_sub - self.y_f(t)) - self.y_f_min) / (
                    self.y_f_max - self.y_f_min)
        else:
            res = min(np.abs(self.y_f_sub - self.y_f(t)), 1)
        return res

    def x_shift(self, t):
        t_lookup = int(math.floor(t / DURATION * 100) / 10)
        res = self.x_move(self.x, self.scale_x_f(t) * self.deltas[t_lookup]["x"])
        left, right = self.barriers[t_lookup]["left"], self.barriers[t_lookup]["right"]
        if self.x_move == np.subtract:
            if res < left:
                self.x_move = np.add
        else:
            if res > right:
                self.x_move = np.subtract
        self.x = self.x_move(self.x, self.scale_x_f(t) * self.deltas[t_lookup]["x"])

    def y_shift(self, t):
        t_lookup = int(math.floor(t / DURATION * 100) / 10)
        res = self.y_move(self.y, self.scale_y_f(t) * self.deltas[t_lookup]["y"])
        top, bottom = self.barriers[t_lookup]["top"], self.barriers[t_lookup]["bottom"]
        if self.y_move == np.add:
            if res > bottom:
                self.y_move = np.subtract
        else:
            if res < top:
                self.y_move = np.add
        self.y = self.y_move(self.y, self.scale_y_f(t) * self.deltas[t_lookup]["y"])


def move_creature(creature, t):
    """
    Calculates the creature's position at time t.
    This now includes the base "wandering" movement and an audio-reactive "jiggle".
    """
    # 1. Get base "wandering" position by updating internal state
    creature.x_shift(t=t)
    creature.y_shift(t=t)
    base_x, base_y = creature.x, creature.y

    # 2. Get audio amplitude (at real time)
    amp = creature.get_amplitude_at(t)

    # 3. Get dance time (slower for some creatures)
    dance_t = t / creature.dance_speed_divisor

    # 4. Add position-based "dancing" (jiggle/back-and-forth)
    jitter_x = (np.cos(dance_t * creature.pos_jitter_freq) *
                amp *
                creature.pos_jitter_amount)
    jitter_y = (np.sin(dance_t * creature.pos_jitter_freq) *
                amp *
                creature.pos_jitter_amount)

    final_x = base_x + jitter_x
    final_y = base_y + jitter_y

    # --- ERROR FIX: Clamp the final coordinates ---
    screen_w, screen_h = creature.back_clip.size

    # --- CHANGE: Use the new pre-calculated *max* size for a safe clamp ---
    creature_w, creature_h = creature.max_w, creature.max_h

    final_x = np.clip(final_x, -creature_w + 1, screen_w - 1)
    final_y = np.clip(final_y, -creature_h + 1, screen_h - 1)
    # --- End of Error Fix ---

    return final_x, final_y


def get_creature(back_clip, audio, wackiness):
    f = get_random_file(STICKERS)
    size_scale = np.random.uniform(.5, 1)
    img = ImageClip(
        str(STICKERS / f),  # Use str() for path compatibility
        duration=DURATION
    )

    # --- EDITED: Minimum Size Logic ---
    # Calculate the original target dimension (based on user's square logic)
    target_dim = round(img.size[0] * size_scale)

    # Enforce the minimum dimension
    final_dim = max(target_dim, MIN_CREATURE_DIMENSION)

    # Use .resized for static resizing with the new final dimension
    img = img.resized(height=final_dim, width=final_dim)
    # --- End of Edited Logic ---

    creature = Creature(img=img,  # This 'img' is the base-sized clip
                        x=np.random.uniform(back_clip.size[0]),
                        y=np.random.uniform(back_clip.size[1]),
                        back_clip=back_clip,
                        audio=audio,  # Pass audio
                        wackiness=wackiness  # Pass wackiness
                        )

    # --- Apply all creature movements ---

    # 1. Position
    img = img.with_position(lambda t: move_creature(
        creature=creature,
        t=t,
    )
                            )

    # 2. Scale
    def scale_func(t):
        amp = creature.get_amplitude_at(t)
        dance_t = t / creature.dance_speed_divisor

        pulse = np.sin(dance_t * creature.scale_pulse_speed) * creature.scale_pulse_amount
        audio_react = amp * creature.scale_audio_sensitivity
        return 1.0 + pulse + audio_react

    img = img.resized(scale_func)

    # 3. Rotation
    def rotate_func(t):
        amp = creature.get_amplitude_at(t)
        dance_t = t / creature.dance_speed_divisor

        base_rot = dance_t * creature.base_rotation_speed
        jiggle = np.sin(dance_t * creature.rot_jiggle_speed) * amp * creature.rot_audio_sensitivity
        return base_rot + jiggle

    # --- FIX: Add expand=True to prevent clipping ---
    img = img.rotated(rotate_func, resample='bicubic', expand=True)

    return img


def get_txt(aux_txt=["SOME", "I", "MADE", ""]):
    done = False
    font_path = ""
    while not done:
        try:
            font_name = random.choice(FONTS)
            font_path = f"/System/Library/Fonts/{font_name}"

            if not os.path.isfile(font_path):
                font_path = font_name

            ImageFont.truetype(font_path, index=0)
            done = True
        except (OSError, AttributeError):
            print(f"Warning: Could not load font {font_path}. Retrying...")

    txt = np.random.choice(
        [
            "SOUNDS",
            # "SONIDOS",
            # "RUMORI",
            # "NOISE",
            # "RUIDO",
            # "CHISME",
            # "JODA",
            # "RELAJO",
            "SANDUNGUEO",
            "PERREO"
        ], 1
    )

    if aux_txt:
        num_aux_words = random.randint(0, 3)
        t = np.random.choice(aux_txt, num_aux_words, replace=False).tolist() + txt
        random.shuffle(t)
    else:
        t = txt

    final_text = "\n".join(t)

    # --- NEW MANUAL PIL-BASED RENDERING LOGIC ---

    font_size = 100
    margin = 20  # Our desired padding

    pil_font = ImageFont.truetype(font_path, font_size, index=0)

    if not final_text:
        # If no text, create a dummy 1x1 transparent clip
        img_array = np.array(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        return ImageClip(img_array, duration=DURATION).with_position("center")

    # 1. Get the text's precise visual bounding box (left, top, right, bottom)
    bbox = pil_font.getbbox(final_text)

    # 2. Calculate the text's *visual* width and height
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 3. Calculate final *background* size with margin
    final_box_w = text_w + (margin * 2)
    final_box_h = text_h + (margin * 2)

    # 4. Get the background color (as hex)
    hex_color = random.choice(COLORS)

    # 5. Create the background image using PIL
    background_image = Image.new("RGBA", (final_box_w, final_box_h), hex_color)

    # 6. Create a drawing context
    draw = ImageDraw.Draw(background_image)

    # 7. Calculate the top-left (x, y) position to draw the text
    #    This calculation centers the *visual* part of the text
    draw_x = margin - bbox[0]
    draw_y = margin - bbox[1]

    # 8. Draw the text onto the background image
    draw.text((draw_x, draw_y), final_text, font=pil_font, fill="black")

    # 9. Convert the final PIL Image to a NumPy array for MoviePy
    final_image_array = np.array(background_image)

    # 10. Create the clip from the array
    final_clip = ImageClip(final_image_array, duration=DURATION)

    # 11. Return the final clip, positioned in the center of the screen
    return final_clip.with_position("center")


def get_random_file(directory):
    files = os.listdir(directory)
    return random.choice([f for f in files if not f.startswith(".")])


def get_clip(audio, add_text=True, aux_text=["SOME", "I", "MADE", ""]):
    f = get_random_file(BCKGRND_DIR)

    # --- Robust Background Crop Logic ---
    img = Image.open(BCKGRND_DIR / f)

    target_w, target_h = 1080, 1920
    target_aspect = target_w / target_h  # 9:16

    orig_w, orig_h = img.size
    orig_aspect = orig_w / orig_h

    if orig_aspect > target_aspect:
        # Image is WIDER than 9:16
        new_h = orig_h
        new_w = int(new_h * target_aspect)
        x_top = random.uniform(0, orig_w - new_w)
        y_top = 0
        crop_box = (x_top, y_top, x_top + new_w, y_top + new_h)
    else:
        # Image is TALLER than 9:16
        new_w = orig_w
        new_h = int(new_w / target_aspect)
        x_top = 0
        y_top = random.uniform(0, orig_h - new_h)
        crop_box = (x_top, y_top, x_top + new_w, y_top + new_h)

    img = img.crop(crop_box)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    # --- End New Logic ---

    back_clip = ImageClip(np.array(img), duration=DURATION,
                          is_mask=False)

    creatures = []
    for i in range(0, np.random.randint(2, 5)):
        creature_wackiness = random.uniform(0, 1)
        creature = get_creature(back_clip, audio, creature_wackiness)
        creatures += [creature]
    if add_text:
        txt = get_txt(aux_txt=aux_text)
        clip = CompositeVideoClip([back_clip] + creatures + [txt])
    else:
        clip = CompositeVideoClip([back_clip] + creatures)
    return clip


def get_audio():
    sound_dirs = ["sets", "sets"]
    d = np.random.choice(sound_dirs, 1, p=[0.3, 0.7])[0]
    f = get_random_file(AUDIO_DIR / d)

    # Fix for 24/32-bit audio files
    audio = AudioFileClip(str(AUDIO_DIR / d / f), fps=48000, nbytes=4)

    lo = np.random.uniform(0, np.max([0, audio.duration - DURATION]))
    hi = lo + np.min([audio.duration, DURATION])
    audio = audio.subclipped(lo, hi)
    return audio


def main():
    n_start = 1
    n_clips = 25
    for hr in tqdm(range(n_start, n_start + n_clips)):
        audio = get_audio()
        clip = get_clip(audio, add_text=True, aux_text=[])
        clip = clip.with_audio(audio)

        clip.write_videofile(
            str(VID_OUT / f"{hr}.mp4"),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=24
        )


if __name__ == "__main__":
    main()
