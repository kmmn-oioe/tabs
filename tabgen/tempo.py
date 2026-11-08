"""Tempo and grid origin from onset evidence and competing pulse grids.

A single autocorrelation peak can follow a repeated accent every three eighths
instead of the beat. Compare quarter/eighth/sixteenth pulse templates, refine
fractional BPM, and estimate the origin using the same template. No song data
or reference TAB is used by this estimator.
"""
from __future__ import annotations

import numpy as np

from .audio import load_audio


# Quarter-note attacks are strongest, with weaker binary subdivisions. The
# normalization penalizes grids which add many empty pulses (double time).
_PULSES = ((0., 1.), (.5, .35), (.25, .1), (.75, .1))


def _onset_envelope(mono: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Spectral flux at 256-sample intervals, including safe short inputs."""
    import scipy.signal
    n_fft, hop = 1024, 256
    dt = hop / sr
    if len(mono) < n_fft:
        return np.zeros(len(mono)//hop), dt
    frames = scipy.signal.stft(mono, nperseg=n_fft, noverlap=n_fft-hop,
                               window='hann', padded=False)[2]
    diff = np.maximum(np.diff(np.abs(frames), axis=1), 0.)
    env = diff.mean(axis=0)
    env /= env.std() + 1e-9
    env = np.convolve(env, [.25, .5, .25], mode='same')
    return env, dt


def _prepare(env, dt):
    env = np.asarray(env, dtype=float)
    if env.ndim != 1 or not np.isfinite(env).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid onset envelope or frame period')
    if len(env)*dt < 2 or len(env) < 8 or np.ptp(env) < 1e-8:
        return None
    # Remove background and limit isolated loud hits; silence contributes zero
    # before centering, so a long silent pre-roll cannot dominate phase fitting.
    signal = np.maximum(env-np.percentile(env,40), 0.)
    signal = np.minimum(signal, np.percentile(signal,99.5))
    signal -= signal.mean()
    scale = signal.std()
    return signal/scale if scale > 1e-8 else None


def _pulse_values(signal, dt, bpm, phases):
    beat = 60./bpm
    duration = len(signal)*dt
    ticks = phases[:,None] + np.arange(0.,duration,beat)[None,:]
    sample_times = np.arange(len(signal))*dt
    summed, energy = np.zeros(len(phases)), np.zeros(len(phases))
    for fraction, weight in _PULSES:
        times = ticks+fraction*beat
        valid = (times >= 0) & (times < duration)
        summed += weight*np.sum(np.interp(times,sample_times,signal,left=0.,right=0.)*valid,axis=1)
        energy += weight*weight*np.sum(valid,axis=1)
    values = summed/np.sqrt(np.maximum(energy,1.))
    # A weak preference breaks almost equivalent metrical interpretations;
    # it neither excludes fast/slow tempos nor folds them into another range.
    return values*np.exp(-.3*np.log(bpm/120.)**2)


def _pulse_fit(signal, dt, bpm, phase_count=48):
    phases = np.arange(phase_count)*(60./bpm)/phase_count
    values = _pulse_values(signal,dt,bpm,phases)
    best = int(np.argmax(values))
    return float(values[best]), float(phases[best])


def _candidates(env, dt, min_bpm=60., max_bpm=220.):
    if not (np.isfinite(min_bpm) and np.isfinite(max_bpm) and 0 < min_bpm < max_bpm):
        raise ValueError('Invalid BPM range')
    signal = _prepare(env,dt)
    if signal is None:
        return []
    # Local windows avoid missing a narrow fractional-BPM peak on a long song
    # during the coarse search. Final scoring uses the complete analysis span.
    window = max(8, round(20./dt))
    positions = sorted({0,max(0,(len(signal)-window)//2),max(0,len(signal)-window)})
    windows = [signal[start:start+window] for start in positions]
    coarse = []
    bpms = np.unique(np.r_[np.arange(min_bpm,max_bpm,1.),max_bpm])
    for bpm in bpms:
        score = np.mean([_pulse_fit(w,dt,bpm,32)[0]/np.sqrt(len(w)*dt) for w in windows])
        coarse.append((float(score),float(bpm)))
    seeds = []
    for score,bpm in sorted(coarse,reverse=True):
        if not any(abs(bpm-other)<2 for other in seeds):
            seeds.append(bpm)
        if len(seeds) == 8:
            break
    refined = {}
    for seed in seeds:
        for bpm in np.arange(max(min_bpm,seed-1),min(max_bpm,seed+1)+.001,.05):
            bpm = round(float(bpm),6)
            if bpm in refined:
                continue
            score,phase = _pulse_fit(signal,dt,bpm)
            refined[bpm] = dict(bpm=bpm,score=score,phase=phase)
    # Expose distinct alternatives, rather than multiple adjacent refinements
    # of a single candidate, for reproducible offline diagnostics.
    candidates = []
    for c in sorted(refined.values(),key=lambda c:c['score'],reverse=True):
        if not any(abs(c['bpm']-p['bpm'])<1 for p in candidates):
            candidates.append(c)
        if len(candidates) == 8:
            break
    return candidates


def estimate_bpm(env: np.ndarray, frame_period: float,
                 min_bpm: float = 60., max_bpm: float = 220.) -> float:
    """Select the best pulse grid; return 120 for uninformative short/silent input."""
    candidates = _candidates(env,frame_period,min_bpm,max_bpm)
    return candidates[0]['bpm'] if candidates else 120.


def estimate_offset(env: np.ndarray, frame_period: float, bpm: float) -> float:
    """Grid origin in [0, quarter-note duration), refined below frame spacing."""
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError('BPM must be positive and finite')
    signal = _prepare(env,frame_period)
    if signal is None:
        return 0.
    beat = 60./bpm
    _,phase = _pulse_fit(signal,frame_period,bpm)
    phases = (phase+np.linspace(-beat/48,beat/48,41)) % beat
    return float(phases[int(np.argmax(_pulse_values(signal,frame_period,bpm,phases)))])


def estimate_bpm_and_offset(wav_path: str, *, diagnostics: dict | None = None) -> tuple[float, float]:
    """Estimate tempo on up to 120 seconds and fit phase at the returned BPM."""
    mono,sr = load_audio(wav_path,sr=22050)
    mono = mono[:sr*120]
    env,dt = _onset_envelope(mono,sr)
    candidates = _candidates(env,dt)
    # The app stores one decimal place. Fit the origin at this exact tempo so
    # rounding BPM later cannot silently change the clock used for the fit.
    bpm = round(candidates[0]['bpm'],1) if candidates else 120.
    offset = estimate_offset(env,dt,bpm) if candidates else 0.
    if diagnostics is not None:
        diagnostics.update(method='pulse_grid',analysis_seconds=len(mono)/sr,
                           candidates=candidates,bpm=bpm,grid_offset=offset,
                           fallback=not bool(candidates))
    return bpm,offset
