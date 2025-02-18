import math
import os

from moviepy import ImageClip, TextClip, CompositeVideoClip, AudioFileClip
from pathlib import Path
import numpy as np
import random
from PIL import Image, ImageFont
from datetime import datetime

from tqdm import tqdm

DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "img"
BCKGRND_DIR = IMG_DIR / "backgrounds"
STICKERS = IMG_DIR / "stickers"
VID_OUT = DATA_DIR / "vid_out"
CACHE = DATA_DIR / "cache"
AUDIO_DIR = DATA_DIR / "sound"
DURATION = 5
FUNCTIONS = [
    np.sin,
    np.cos,
    np.tan,
    # lambda x: np.pow(x, 2),
    # lambda x: np.pow(x, 3),
]
COLORS = [
    # "#032e55",
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
                 x_add=None,
                 y_add=None,
                 ):
        self.y_f = random.choice(FUNCTIONS)
        self.y_f_min = np.abs(self.y_f(0))
        self.y_f_max = np.abs(self.y_f(DURATION))
        self.x_f = random.choice(FUNCTIONS)
        self.x_f_min = np.abs(self.x_f(0))
        self.x_f_max = np.abs(self.x_f(DURATION))
        self.x_f_sub = np.random.uniform(0, DURATION)
        self.y_f_sub = np.random.uniform(0, DURATION)
        self.img = img
        self.x = x
        self.y = y
        self.back_clip = back_clip
        s = 300
        self.x_delta = back_clip.size[0] / np.random.normal(s, s*.15)
        self.y_delta = back_clip.size[1] / np.random.normal(s, s*.15)
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
                    "right": self.back_clip.size[0] * (1-buffer),
                    "top": -self.back_clip.size[1] * buffer,
                    "bottom": self.back_clip.size[1] * (1-buffer),
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
                x_denom = np.random.normal(s, s*.25)
                y_denom = np.random.normal(s, s*.25)
                self.deltas[i] = {
                    "x": self.back_clip.size[0] / x_denom,
                    "y": self.back_clip.size[1] / y_denom
                }
            else:
                self.deltas[i] = self.deltas[0]

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
                                np.random.uniform(self.img.size[side], self.back_clip.size[side]*.9)
                        ),
                        self.back_clip.size[side]*(1+buffer)
                    )
        }

    def scale_x_f(self, t):
        if self.x_f not in [np.cos, np.tan, np.sin]:
            res = (np.abs(self.x_f_sub - self.x_f(t)) - self.x_f_min) / (self.x_f_max - self.x_f_min)
        else:
            res = min(np.abs(self.x_f_sub - self.x_f(t)), 1)
        return res

    def scale_y_f(self, t):
        if self.x_f not in [np.cos, np.tan, np.sin]:
            res = (np.abs(self.y_f_sub - self.y_f(t)) - self.y_f_min) / (self.y_f_max - self.y_f_min)
        else:
            res = min(np.abs(self.y_f_sub - self.y_f(t)), 1)
        return res

    def x_shift(self, t):
        t_lookup = int(math.floor(t/DURATION*100)/10)
        res = self.x_move(self.x, self.scale_x_f(t)*self.deltas[t_lookup]["x"])
        left, right = self.barriers[t_lookup]["left"], self.barriers[t_lookup]["right"]
        if self.x_move == np.subtract:
            if res < left:
                self.x_move = np.add
        else:
            if res > right:
                self.x_move = np.subtract
        res = self.x_move(self.x, self.scale_x_f(t)*self.deltas[t_lookup]["x"])
        # if not left < res < right:
        #     print("x out of bounds", res)
        #     print(self.x_move, self.x, self.scale_x_f(t), self.deltas[t_lookup]["x"])
        #     print("left", left, "right", right)
        self.x = self.x_move(self.x, self.scale_x_f(t)*self.deltas[t_lookup]["x"])

    def y_shift(self, t):
        t_lookup = int(math.floor(t / DURATION * 100) / 10)
        res = self.y_move(self.y, self.scale_y_f(t)*self.deltas[t_lookup]["y"])
        top, bottom = self.barriers[t_lookup]["top"], self.barriers[t_lookup]["bottom"]
        if self.y_move == np.add:
            if res > bottom:
                self.y_move = np.subtract
        else:
            if res < top:
                self.y_move = np.add
        res = self.y_move(self.y, self.scale_y_f(t)*self.deltas[t_lookup]["y"])
        # if not top < res < bottom:
        #     print("y out of bounds", res)
        #     print(self.y_move, self.y, self.scale_y_f(t), self.deltas[t_lookup]["y"])
        #     print("top", top, "bottom", bottom)
        self.y = self.y_move(self.y, self.scale_y_f(t)*self.deltas[t_lookup]["y"])


def move_creature(creature, t):
    creature.x_shift(t=t)
    creature.y_shift(t=t)
    return creature.x, creature.y


def get_creature(back_clip):
    f = get_random_file(STICKERS)
    img = ImageClip(
        STICKERS / f,
        duration=DURATION
    )
    creature = Creature(img=img,
                        x=np.random.uniform(back_clip.size[0]),
                        y=np.random.uniform(back_clip.size[1]),
                        back_clip=back_clip
                        )

    x_c, y_c = np.random.uniform(0, back_clip.size[0]), np.random.uniform(0, back_clip.size[1])
    x_target = np.random.uniform(0, back_clip.size[0])
    x_0 = (back_clip.size[0] / 2) - (img.size[0] / 2)
    y_0 = (back_clip.size[1] / 2) - (img.size[1] / 2)
    img = img.with_position(lambda t: move_creature(
        creature=creature,
        t=t,
        )
    )
    return img


def get_txt():
    done = False
    font = ""
    while not done:
        try:
            font = random.choice(FONTS)
            ImageFont.truetype(font)
            done = True
        except OSError:
            print(font)
    aux_txt = ["SOME", "I", "MADE", ""]
    txt = ["SOUNDS"]
    t = np.random.choice(aux_txt, 3).tolist() + txt
    random.shuffle(t)
    text_clip = TextClip(
        text="\n".join(t),
        margin=(20, 20),
        font_size=100,
        text_align="center",
        font=font,
        color="black",
        bg_color=random.choice(COLORS),
        duration=DURATION
     ).with_position("center")
    return text_clip


def get_random_file(directory):
    files = os.listdir(directory)
    return random.choice([f for f in files if not f.startswith(".")])


def get_clip():
    f = get_random_file(BCKGRND_DIR)
    img = Image.open(BCKGRND_DIR / f)
    x_top = np.random.uniform(0, img.size[0] - 1080)
    y_top = np.random.uniform(0, img.size[1] - 1920)
    img = img.crop((x_top, y_top, x_top + 1080, y_top + 1920))
    back_clip = ImageClip(np.array(img), duration=DURATION,
                          is_mask=False)
    txt = get_txt()
    creature = get_creature(back_clip)
    clip = CompositeVideoClip([back_clip, creature, txt])
    return clip


def get_audio():
    sound_dirs = ["beats", "sets"]
    d = np.random.choice(sound_dirs, 1, [0.2, 0.8])[0]
    f = get_random_file(AUDIO_DIR / d)
    audio = AudioFileClip(AUDIO_DIR / d / f)
    lo = np.random.uniform(0, np.max([0, audio.duration - DURATION]))
    hi = lo + np.min([audio.duration, DURATION])
    audio = audio[lo:hi]
    return audio


def main():
    n_clips = 10
    for hr in tqdm(range(1, n_clips + 1)):
        clip = get_clip()
        audio = get_audio()
        clip = clip.with_audio(audio)
        clip.write_videofile(
            VID_OUT / f"{hr}.mp4",
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=24
        )


if __name__ == "__main__":
    main()
