import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tabgen import player as module
from tabgen.playhead import PlaybackTiming, parse_blocks, cursor_cells
from tabgen.tab import TabOptions, notes_to_tab


class CallbackStop(Exception):
    pass


class FakeStream:
    instances = []
    peak_active = 0
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.time = 10.
        self.latency = .1
        self.closed = False
        self.active = False
        self.instances.append(self)

    def start(self):
        self.active = True
        type(self).peak_active = max(type(self).peak_active, sum(s.active for s in self.instances))

    def stop(self):
        self.active = False
        self.kwargs['finished_callback']()

    def abort(self):
        self.stop()

    def close(self):
        self.active = False
        self.closed = True

    def emit(self, count, dac=10.1):
        output = np.full((count, self.kwargs.get('channels',1)), 999., dtype=np.float32)
        try:
            self.kwargs['callback'](output, count, SimpleNamespace(outputBufferDacTime=dac, currentTime=dac-.1), None)
        except CallbackStop:
            pass
        return output


class PlayerClockTests(unittest.TestCase):
    def setUp(self):
        self.sd = patch.object(module, 'sd', SimpleNamespace(OutputStream=FakeStream, CallbackStop=CallbackStop), create=True)
        self.ready = patch.object(module, 'SD_OK', True)
        self.sd.start()
        self.ready.start()
        FakeStream.instances = []
        FakeStream.peak_active = 0
        self.player = module.AudioPlayer()
        self.player.sr = 1000
        self.player.data = np.ones((4000, 1), dtype=np.float32)

    def tearDown(self):
        self.player.stop()
        self.sd.stop()
        self.ready.stop()

    def test_clock_tracks_audible_time_not_submitted_buffers(self):
        self.assertTrue(self.player.play())
        stream = FakeStream.instances[-1]
        stream.emit(1000)
        self.assertEqual(self.player.position, 0.)  # DAC scheduled in the future.
        stream.time = 10.35
        self.assertAlmostEqual(self.player.position, .25)
        stream.time = 50
        self.assertEqual(self.player.position, 1.)  # Never beyond queued data.

    def test_pause_freezes_position_and_old_finished_callback_is_ignored(self):
        self.player.play()
        stream = FakeStream.instances[-1]
        stream.emit(1000)
        stream.time = 10.6
        self.player.pause()
        self.assertAlmostEqual(self.player.position, .5)
        stream.kwargs['finished_callback']()
        self.assertAlmostEqual(self.player.position, .5)
        self.assertTrue(stream.closed)
        self.assertFalse(self.player.playing)

    def test_seek_restarts_playing_stream_and_rejects_old_completion(self):
        self.player.play()
        old = FakeStream.instances[-1]
        old.emit(1000)
        self.player.seek(2)
        current = FakeStream.instances[-1]
        self.assertIsNot(current, old)
        self.assertTrue(self.player.playing)
        old.kwargs['finished_callback']()
        current.emit(1000)
        current.time = 10.35
        self.assertAlmostEqual(self.player.position, 2.25)

    def test_final_partial_buffer_is_zero_filled_and_finishes_at_end(self):
        self.player.data = np.arange(3, dtype=np.float32).reshape(-1, 1)
        self.player.play()
        stream = FakeStream.instances[-1]
        np.testing.assert_array_equal(stream.emit(8).ravel(), [0,1,2,0,0,0,0,0])
        stream.kwargs['finished_callback']()
        self.assertFalse(self.player.playing)
        self.assertEqual(self.player.position, self.player.duration)

    def test_seek_end_stops_and_stop_preserves_position(self):
        self.player.play()
        self.player.seek(self.player.duration)
        self.assertFalse(self.player.playing)
        self.assertEqual(self.player.position, 4.)
        self.player.stop()
        self.assertEqual(self.player.position, 4.)
        self.player.seek(0)
        self.assertEqual(self.player.position, 0.)

    def test_failed_play_reports_error_instead_of_looking_like_playback(self):
        with patch.object(module.sd, 'OutputStream', side_effect=RuntimeError('No output device')):
            self.assertFalse(self.player.play())
        self.assertFalse(self.player.playing)
        self.assertIn('No output device', self.player.last_error)

    def test_missing_dac_timestamp_uses_monotonic_clock_and_output_latency(self):
        with patch.object(module.time, 'perf_counter', return_value=20.):
            self.player.play()
            FakeStream.instances[-1].emit(1000, dac=0.)
        with patch.object(module.time, 'perf_counter', return_value=20.35):
            self.assertAlmostEqual(self.player.position,.25)

    def test_loading_new_audio_silences_old_callback_and_keeps_one_stream(self):
        self.player.play()
        old = FakeStream.instances[-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'separated.wav'
            sf.write(path,np.full(4000,.25),1000,subtype='FLOAT')
            self.assertTrue(self.player.load(str(path)))
            self.assertFalse(self.player.playing)
            self.assertTrue(old.closed)
            self.assertTrue(self.player.play())
            current = FakeStream.instances[-1]
            np.testing.assert_array_equal(old.emit(8).ravel(),np.zeros(8))
            np.testing.assert_array_equal(current.emit(8).ravel(),np.full(8,.25))
        self.assertEqual(FakeStream.peak_active,1)


class TabTimelineTests(unittest.TestCase):
    def setUp(self):
        self.opts = TabOptions(bpm=120, grid_offset=.25)
        self.timing = PlaybackTiming.from_options(self.opts)
        self.tab = notes_to_tab([dict(pitch=78,string=0,fret=14,start=.25,end=18.,confidence=.9)],
                                self.opts, arranged=True, duration=18.)
        self.blocks = parse_blocks(self.tab,self.timing)

    def test_offset_and_smooth_motion_with_two_character_frets(self):
        self.assertEqual(cursor_cells(self.blocks,self.timing,.2), [])
        a = cursor_cells(self.blocks,self.timing,.25)[0]
        half = cursor_cells(self.blocks,self.timing,.3125)[0]
        self.assertEqual(a.column,2)
        self.assertEqual(self.tab.splitlines()[a.block.rows[0]-1][2:4], '14')
        self.assertEqual(half.column,3)
        self.assertEqual(half.cell_column,2)

    def test_bar_separator_and_next_block_positions_match_renderer(self):
        # At the second bar the renderer skips a two-character separator slot.
        bar2 = cursor_cells(self.blocks,self.timing,2.25)[0]
        self.assertEqual(bar2.column,2+2*(16+1))
        next_block = cursor_cells(self.blocks,self.timing,8.25)[0]
        self.assertEqual(next_block.block.first_bar,5)
        self.assertEqual(next_block.column,2)

    def test_parts_share_the_clock_and_headers_do_not_affect_timing(self):
        text = '====== リードギター ======\n説明\n'+self.tab+'\n====== リズムギター ======\nBPM 120\n'+self.tab
        blocks = parse_blocks(text,self.timing)
        cells = cursor_cells(blocks,self.timing,3.25)
        self.assertEqual([c.block.part for c in cells], ['リードギター','リズムギター'])
        self.assertEqual(cells[0].column,cells[1].column)

    def test_generation_snapshot_does_not_follow_later_form_edits(self):
        self.opts.bpm = 200
        self.opts.grid_offset = 7
        self.assertEqual(self.timing.bpm,120)
        self.assertEqual(self.timing.offset,.25)

    def test_broken_edited_rows_disable_mapping_instead_of_pointing_at_wrong_text(self):
        broken = self.tab.replace('1|','X|')
        self.assertEqual(parse_blocks(broken,self.timing), [])

    def test_rest_cells_and_custom_quantization_are_mapped(self):
        opts = TabOptions(bpm=90,quantize=.5)
        tab = notes_to_tab([dict(pitch=52,start=10,end=10.2,confidence=.9)],opts)
        blocks = parse_blocks(tab,PlaybackTiming.from_options(opts))
        self.assertTrue(cursor_cells(blocks,PlaybackTiming.from_options(opts),0))
        self.assertEqual(len(blocks[0].rows),6)


if __name__ == '__main__':
    unittest.main()
