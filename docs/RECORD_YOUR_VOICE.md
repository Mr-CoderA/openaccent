# Record your voice profile (one-time, ~5 minutes of effort)

A single good recording (`me_profile.wav`) becomes your reusable cloning
reference for EVERY conversion via `--voice-ref`. You never re-record per clip.

## Setup

| factor | what to do |
|---|---|
| Room | smallest quiet room; curtains/bed/couch absorb echo; fan & AC OFF |
| Phone/mic | hold 15–20 cm from mouth, slightly off-axis (reduces puffs) |
| Level | speak at normal conversation volume — don't whisper, don't project |
| Pace | your natural pace; pause briefly at commas/periods |
| Takes | mistakes? just pause 2 s and repeat the sentence — we can't edit, so keep going cleanly |
| Format | any (wav/m4a/mp3) — pipeline normalizes to 16 kHz mono |

## Reading script (~60–90 seconds at natural pace)

Read this aloud in one recording. It covers all major English sounds, plus
numbers and question intonation:

```
Please call Stella. Ask her to bring these things with her from the store.
Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a
snack for her brother Bob.

We also need a small plastic snake and a big toy frog for the kids.
She can scoop these things into three red bags, and we will go meet her
Wednesday at the train station.

When the sunlight strikes raindrops in the air, they act as a prism and
form a rainbow. The rainbow is a division of white light into many beautiful
colors. These are red, orange, yellow, green, blue, indigo, and violet.

Take sixty-five minutes to finish thirty pages of the third chapter.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
She sells seashells by the seashore on clear summer mornings.

My name is [your name]. I was born in [your city], and I have lived there
for [number] years. Today I feel great, and this is my real voice.
```

(The last paragraph matters most — your own name/habits in your natural voice.)

## Use it

```bash
# convert ANY audio with YOUR voice preserved:
./.venv/bin/python anycast.py --input some_clip.wav --target pk \
    --voice-ref me_profile.wav --output out.wav

# max accuracy:
./.venv/bin/python anycast.py --input some_clip.wav --target us \
    --voice-ref me_profile.wav --best-of 5 --preset max --output out.wav
```

## What to expect

- Profile quality dominates cloning similarity. Clean 60–90 s enrollment:
  expect clearly higher similarity than the ~0.80 you saw with short clips.
- The accent reference stays separate (`samples/ref_*.wav` or `--reference`).
- Without `--voice-ref`, the system falls back to using the input clip itself
  as enrollment (short clips = weaker cloning).
