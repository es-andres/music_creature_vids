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
DURATION = 4
FUNCTIONS = [
    np.sin,
    np.cos,
    np.tan,
]
COLORS = [
    "#f77866",
    "#fd909c",
    "#d11b20",
    # "#692e29",
    "#c7e67b",
    "#96b2de",
    "#f7b819",
    # "#a98a2e",
    "#763374",
    "#00b1d1"
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
        self.img = img
        self.x = x
        self.y = y
        self.back_clip = back_clip
        self.x_delta = back_clip.size[0] / 30
        self.y_delta = back_clip.size[1] / 30
        if not x_add:
            self.x_add = random.choice([np.add, np.subtract])
        if not y_add:
            self.y_add = random.choice([np.add, np.subtract])
        break_nums = []
        for draws in range(1, int(np.random.uniform(2, 5))):
            break_nums += [np.random.randint(0, 10)]
        break_nums = list(sorted(set([0] + break_nums)))
        x_barrier = self.get_barrier(0)
        y_barrier = self.get_barrier(1)
        self.barriers = {
            0: {
                    "left": 0,
                    "right": self.back_clip.size[0],
                    "top": 0,
                    "bottom": self.back_clip.size[1]
                }
        }
        break_nums = [0]
        for i in range(1, 11):
            if i in break_nums:
                x_barrier = self.get_barrier(0)
                y_barrier = self.get_barrier(1)
                self.barriers[i] = {
                    "left": x_barrier["low"],
                    "right": x_barrier["high"],
                    "top": y_barrier["low"],
                    "bottom": y_barrier["high"]
                }
            else:
                self.barriers[i] = self.barriers[0]

    def get_barrier(self, side):
        barrier_start = np.random.uniform(
            0,
            self.back_clip.size[side]
        )
        return {
            "low": barrier_start,
            "high": min(
                        (
                                barrier_start +
                                np.random.uniform(
                                    0, self.back_clip.size[side]
                                )
                        ),
                        self.back_clip.size[side]
                    )
        }

    def x_move(self, t):
        f = random.choice(FUNCTIONS)
        f = lambda x: 1
        res = self.x_add(self.x, f(t / DURATION)*self.x_delta)
        if self.x_add == np.subtract:
            if res < self.barriers[int(math.floor(t/DURATION*100)/10)]["left"]:
                self.x_add = np.add
        else:
            if res > self.barriers[int(math.floor(t/DURATION*100)/10)]["right"]:
                self.x_add = np.subtract
        self.x = self.x_add(self.x, f(t / DURATION)*self.x_delta)

    def y_move(self, t):
        f = random.choice(FUNCTIONS)
        f = lambda x: 1
        res = self.y_add(self.y, f(t / DURATION)*self.y_delta)
        if self.y_add == np.add:
            if res > self.barriers[int(math.floor(t/DURATION*100)/10)]["bottom"]:
                self.y_add = np.subtract
        else:
            if res < self.barriers[int(math.floor(t/DURATION*100)/10)]["top"]:
                self.y_add = np.add
        self.y = self.y_add(self.y, f(t / DURATION)*self.y_delta)


def move_creature(creature, t):
    creature.x_move(t=t)
    creature.y_move(t=t)
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
    f = get_random_file(AUDIO_DIR)
    print(f)
    audio = AudioFileClip(AUDIO_DIR / f)
    lo = np.random.uniform(0, np.max([0, audio.duration - DURATION]))
    hi = lo + np.min([audio.duration, DURATION])
    audio = audio[lo:hi]
    return audio


def main():
    n = np.random.randint(4, 8)
    n = 5
    for hr in tqdm(range(1,2)):
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
