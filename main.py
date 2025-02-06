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
DURATION = 15
FUNCTIONS = [
    np.sin,
    np.cos,
    np.tan,
]
COLORS = [
    "#ff0000",
    "#ffac00",
    "#fff100",
    "#0bff00",
    "#00f6ff"
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


def get_creature(back_clip):
    f = random.choice(os.listdir(STICKERS))
    img = ImageClip(
        STICKERS / f,
        duration=DURATION
    )
    x_f, y_f = random.choice(FUNCTIONS), random.choice(FUNCTIONS)

    def move_creature(t, s, back_clip, x_c, y_c):

        x, y = np.abs(x_f(t)*s), np.abs(y_f(t)*s)
        x += x_c
        y += y_c
        if x > back_clip.size[0]:
            x = x - back_clip.size[0]*np.floor(x / back_clip.size[0])
        if y > back_clip.size[1]:
            y = y - back_clip.size[1]*np.floor(y / back_clip.size[1])
        return x, y

    s = np.random.uniform(100, 500)
    x_c, y_c = np.random.uniform(0, back_clip.size[0]), np.random.uniform(0, back_clip.size[1])
    img = img.with_position(lambda t: move_creature(t, s, back_clip, x_c, y_c))
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


def get_clip():
    f = random.choice(os.listdir(BCKGRND_DIR))
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
    f = random.choice(os.listdir(AUDIO_DIR))
    print(f)
    audio = AudioFileClip(AUDIO_DIR / f)
    lo = np.random.uniform(0, np.max([0, audio.duration - DURATION]))
    hi = lo + np.min([audio.duration, DURATION])
    audio = audio[lo:hi]
    return audio


def get_times():
    """
    Monday: 7 am, 11 am, 2 pm
    Tuesday: 7 am, 10 am, 1 pm
    Wednesday: 7 am, 9 am, 4 pm
    Thursday: 7 am, 9 am, 2 pm
    Friday: 7 am, 11 am, 3 pm
    Saturday:  7 am, 12 pm, 3 pm
    Sunday: 7 am, 11 am, 7 pm
    :return:
    """
    today = datetime.now().strftime("%A")
    times = {
        "Monday": [
            "7", "11", "14"
        ],
        "Tuesday": [
            "7", "10", "13"
        ],
        "Wednesday": [
            "7", "9", "16"
        ],
        "Thursday": [
            "7", "9", "14"
        ],
        "Friday": [
            "7", "11", "15"
        ],
        "Saturday": [
            "7", "12", "15"
        ],
        "Sunday": [
            "7", "11", "19"
        ],
    }
    return times[today]


def main():
    today = datetime.today().strftime('%Y_%m_%d')
    n = np.random.randint(4, 8)
    n = 5
    for hr in tqdm(range(3, 4)):
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
