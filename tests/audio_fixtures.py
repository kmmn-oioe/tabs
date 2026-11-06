"""Small generated, labeled multitrack scenes; these are not real-song data."""
import numpy as np

SAMPLE_RATE = 22050


def guitar_scene(*, lead=False, stereo=True, moving=False, spread=.7,
                 lead_harmonics=(1, .35, .2, .1, .07), lead_antiphase=False,
                 varying_timbre=False):
    duration = 4.
    audio = np.zeros((int(SAMPLE_RATE*duration), 2))
    notes, labels = [], []

    def add(pitch, start, length, pan, level, harmonics, is_lead):
        t = np.arange(round(SAMPLE_RATE*length)) / SAMPLE_RATE
        phase = 2*np.pi*440*2**((pitch-69)/12)*t
        wave = sum(a*np.sin(h*phase)*np.exp(-.4*h*t) for h,a in enumerate(harmonics, 1))
        envelope = np.minimum(1, t/.008) * np.minimum(1, (length-t)/.03)
        if not is_lead:
            envelope *= np.exp(-2*t)
        wave *= level*envelope
        gains = np.sqrt(np.array([1-pan, 1+pan])/2)
        if is_lead and lead_antiphase:
            gains[1] *= -1
        first = round(start*SAMPLE_RATE)
        stop = min(len(audio), first+len(t))
        audio[first:stop] += wave[:stop-first, None]*gains
        notes.append(dict(pitch=pitch, start=start, end=min(duration, start+length), confidence=.9))
        labels.append(is_lead)

    for i,start in enumerate(np.arange(0, 3.7, .25)):
        pan = -spread if not moving or start < 2 else spread
        harmonics = [.55,1,.7,.5,.3] if varying_timbre and i % 2 else [1,.45,.25,.15,.08]
        add([45,52,57,60][i % 4], float(start), .55, pan, .22, harmonics, False)
    if lead:
        for i,pitch in enumerate([66,64,62,64,66]):
            add(pitch, 1.6+i*.4, .32, spread, .22, lead_harmonics, True)
    return (audio if stereo else audio.mean(axis=1)), notes, labels
