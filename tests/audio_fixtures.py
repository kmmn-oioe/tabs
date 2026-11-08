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


def recurring_melody_scene(*, lower_lead=False, mirror=False):
    """A frequent melody over a ringing ostinato, with a silent lead intro.

    Both voices use the same harmonic envelope. The melody has more attacks
    than the accompaniment; the lower-register variant rules out pitch-height
    selection, and mirroring rules out hardcoded left/right assignments.
    """
    duration = 4.
    audio = np.zeros((round(SAMPLE_RATE*duration), 2))
    notes, labels = [], []

    def add(pitch, start, length, pan, is_lead):
        t = np.arange(round(SAMPLE_RATE*length))/SAMPLE_RATE
        phase = 2*np.pi*440*2**((pitch-69)/12)*t
        wave = sum(a*np.sin(h*phase)*np.exp(-.5*h*t)
                   for h,a in enumerate((1,.35,.18,.1,.06),1))
        wave *= .18*np.minimum(1,t/.008)*np.minimum(1,(length-t)/.02)
        gains = np.sqrt(np.array([1-pan,1+pan])/2)
        first = round(start*SAMPLE_RATE)
        stop = min(len(audio),first+len(t))
        audio[first:stop] += wave[:stop-first,None]*gains
        notes.append(dict(pitch=pitch,start=start,end=min(duration,start+length),confidence=.7))
        labels.append(is_lead)

    direction = -1 if mirror else 1
    backing = [69,81] if lower_lead else [45,57]
    melody = [52,54,61,59] if lower_lead else [64,66,73,71]
    for i in range(10):
        add(backing[i%2],i*.4,.55,-.7*direction,False)
    for i in range(15):
        add(melody[i%4],.8+i*.2,.18,.7*direction,True)
    return audio,notes,labels
