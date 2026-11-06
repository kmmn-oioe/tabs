import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tabgen.split import split_notes
from tabgen.harmony import ChromaFeatures, chords_from_chroma, audio_chord_changes
from tabgen.tab import TabOptions


def note(pitch, start=0., end=None, confidence=.9):
    return dict(pitch=pitch, start=start, end=start+.4 if end is None else end, confidence=confidence)


def features(segments):
    times = np.arange(0, segments[-1][1], .025)
    chroma = np.zeros((12, len(times)))
    bass = np.zeros_like(chroma)
    rms = np.zeros(len(times))
    for start, end, pitches in segments:
        selected = (times >= start) & (times < end)
        for pitch in pitches:
            chroma[pitch % 12, selected] += 1
        if pitches:
            bass[min(pitches) % 12, selected] = 1
            rms[selected] = .1
    return ChromaFeatures(times, chroma, bass, rms, segments[-1][1])


class MelodyTests(unittest.TestCase):
    def test_low_melody_is_not_discarded(self):
        lead, rhythm = split_notes([note(50), note(52, .5), note(54, 1)], lead_threshold=76)
        self.assertEqual([n['pitch'] for n in lead], [50, 52, 54])
        self.assertEqual(rhythm, [])

    def test_lookahead_tracks_low_line_under_high_accompaniment(self):
        notes = [note(84, 0, 2, .95), note(55, 0, .7, .85), note(57, .5), note(59, 1)]
        original = [dict(n) for n in notes]
        lead, rhythm = split_notes(notes)
        self.assertEqual([n['pitch'] for n in lead], [55, 57, 59])
        self.assertEqual([n['pitch'] for n in rhythm], [84])
        self.assertLessEqual(lead[0]['end'], lead[1]['start'])
        self.assertEqual(notes, original)

    def test_skip_weak_octave_artifact_and_allow_rest(self):
        lead, rhythm = split_notes([note(60), note(96, .25, confidence=.31), note(62, .5), note(84, 4)])
        self.assertEqual([n['pitch'] for n in lead], [60, 62, 84])
        self.assertEqual([n['pitch'] for n in rhythm], [96])

    def test_chord_attacks_are_not_forced_into_melody(self):
        notes = [note(p) for p in [48, 52, 55, 60]] + [note(62, .5)]
        lead, rhythm = split_notes(notes)
        self.assertEqual([n['pitch'] for n in lead], [62])
        self.assertEqual(len(rhythm), 4)
        self.assertEqual(len(split_notes(notes, suppress_chords=False)[0]), 2)
        self.assertEqual(split_notes([]), ([], []))
        self.assertEqual(split_notes([note(60, confidence=.1)])[0], [])


class AudioHarmonyTests(unittest.TestCase):
    def test_change_inside_old_four_beat_block(self):
        data = features([(0, 1.5, [48, 52, 55]), (1.5, 3, [43, 47, 50])])
        changes = chords_from_chroma(data, TabOptions(chord_beats=4))
        self.assertEqual([c.name for c in changes], ['C', 'G'])
        self.assertAlmostEqual(changes[1].start, 1.5)

    def test_passing_tone_does_not_flicker_and_modulation_allowed(self):
        data = features([(0, 2, [48, 52, 55]), (2, 4, [42, 46, 49])])
        data.chroma[6, (data.times > .9) & (data.times < 1)] += 1
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C', 'F#'])
        self.assertAlmostEqual(changes[1].start, 2.)

    def test_silence_and_single_pitch_do_not_invent_chords(self):
        data = features([(0, 1, [48, 52, 55]), (1, 2, []), (2, 3, [60])])
        changes = chords_from_chroma(data, TabOptions())
        self.assertEqual([c.name for c in changes], ['C', 'N.C.'])
        self.assertAlmostEqual(changes[1].start, 1)
        self.assertAlmostEqual(changes[-1].end, 3)

    def test_actual_audio_not_detected_note_input(self):
        # Unequal chord lengths force a change outside a fixed two-beat window.
        sr = 22050
        waves = []
        for pitches, duration in [([48, 52, 55], 1.5), ([43, 47, 50], 2.)]:
            t = np.arange(round(sr*duration))/sr
            ramp = np.minimum(1, t*100) * np.minimum(1, (duration-t)*100)
            waves.append(sum(np.sin(2*np.pi*440*2**((p-69)/12)*t) for p in pitches)*.15*ramp)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'chords.wav'
            sf.write(path, np.concatenate(waves), sr)
            changes = audio_chord_changes(str(path), TabOptions())
        self.assertEqual([c.name for c in changes], ['C', 'G'])
        self.assertAlmostEqual(changes[1].start, 1.5)

    def test_short_and_silent_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'silence.wav'
            for length in (0, 100, 22050):
                sf.write(path, np.zeros(length), 22050)
                changes = audio_chord_changes(str(path), TabOptions())
                self.assertTrue(all(c.name == 'N.C.' for c in changes))

    def test_pipeline_reports_note_fallback(self):
        from tabgen.pipeline import generate
        logs = []
        notes = [note(p) for p in (48, 52, 55)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audio.wav'
            path.touch()
            with patch('tabgen.transcribe.transcribe', return_value=notes), \
                 patch('tabgen.harmony.audio_chord_changes', side_effect=RuntimeError('decode error')):
                text, mid, xml, _ = generate(str(path), split_lead_rhythm=True, auto_bpm=False,
                                            progress=lambda pct,msg: logs.append(msg))
            self.assertIn('C', text.split('コード進行')[1])
            self.assertTrue(any('検出ノートから推定' in msg for msg in logs))
            self.assertTrue(all(Path(p).is_file() for p in mid.split(';') + xml.split(';')))


class SeparationTests(unittest.TestCase):
    def test_selects_guitar_not_no_guitar(self):
        from tabgen.separate import separate_guitar
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp)/'htdemucs_6s'/'audio'/'guitar.wav'
            expected.parent.mkdir(parents=True)
            expected.touch()
            proc = unittest.mock.Mock(stderr=io.BytesIO(), returncode=0)
            with patch('tabgen.separate._device_args', return_value=[]), \
                 patch('tabgen.separate.subprocess.Popen', return_value=proc):
                self.assertEqual(separate_guitar('audio.wav', tmp), str(expected))


if __name__ == '__main__':
    unittest.main()
