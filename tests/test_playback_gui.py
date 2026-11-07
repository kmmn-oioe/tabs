import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from main import App
from tabgen import player as module
from tabgen.tab import TabOptions, notes_to_tab
from tabgen.playhead import cursor_cells
from test_playback import FakeStream, CallbackStop


@unittest.skipUnless(os.environ.get('DISPLAY') or sys.platform == 'win32', 'A graphical display is required')
class PlaybackGuiTests(unittest.TestCase):
    def setUp(self):
        self.sd = patch.object(module, 'sd', SimpleNamespace(OutputStream=FakeStream, CallbackStop=CallbackStop), create=True)
        self.ready = patch.object(module, 'SD_OK', True)
        self.sd.start()
        self.ready.start()
        FakeStream.instances = []
        FakeStream.peak_active = 0
        self.app = App()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'song.wav'
        sf.write(self.path, np.zeros(24000), 1000)
        self.opts = TabOptions(bpm=120, grid_offset=.25)
        tab = notes_to_tab([dict(pitch=64,start=.25,end=24.,confidence=.9)],self.opts,duration=24.)
        self.content = '====== リードギター ======\n'+tab+'\n====== リズムギター ======\n説明\n'+tab
        self.app._done(self.content,'','',str(self.path),self.opts)
        self.app.update()

    def tearDown(self):
        self.app.destroy()
        self.temp.cleanup()
        self.ready.stop()
        self.sd.stop()

    def test_cursor_moves_without_editing_saved_text_and_holds_when_paused(self):
        app = self.app
        app.player.seek(.25)
        app._refresh_playback()
        app.update()
        bar = app.playhead._bars[0]
        first_x = bar.winfo_x()
        self.assertTrue(bar.winfo_ismapped())
        self.assertEqual(bar.winfo_y(), app.text.dlineinfo(f'{app.playhead.blocks[0].rows[0]}.0')[1])
        app.player.seek(.3125)
        app._refresh_playback()
        app.update()
        self.assertGreater(bar.winfo_x(),first_x)
        held_x = bar.winfo_x()
        app._refresh_playback()
        app.update()
        self.assertEqual(bar.winfo_x(),held_x)
        self.assertEqual(app._current_text(),self.content)

    def test_seek_and_part_choice_follow_correct_rows(self):
        app = self.app
        app.player.seek(8.25)
        app._refresh_playback()
        app.update()
        active = [b for b in app.playhead.blocks if b.first_bar==5]
        self.assertIsNotNone(app.text.dlineinfo(f'{active[0].rows[-1]}.0'))
        app.follow_part.set('リズムギター')
        app.playhead.select_part('リズムギター')
        app.update()
        self.assertIsNotNone(app.text.dlineinfo(f'{active[1].rows[-1]}.0'))
        self.assertTrue(app.playhead._bars[1].winfo_ismapped())
        self.assertEqual(app._current_text(),self.content)

    def test_font_change_and_horizontal_scroll_keep_cursor_aligned(self):
        app = self.app
        app.font_size.set(20)
        app._apply_font()
        app.player.seek(6.25)
        app._refresh_playback()
        app.update()
        self.assertGreater(app.text.xview()[0],0)
        cell = next(c for c in cursor_cells(
            app.playhead.blocks,app.playhead.timing,6.25) if c.block.part==app.playhead.part)
        row = next(row for row in cell.block.rows if app.text.bbox(f'{row}.{int(cell.column)}'))
        box = app.text.bbox(f'{row}.{int(cell.column)}')
        bar = app.playhead._bars[0]
        self.assertAlmostEqual(bar.winfo_x(),box[0],delta=1)
        self.assertTrue(bar.winfo_ismapped())

    def test_stop_returns_to_beginning_and_updates_slider_while_not_playing(self):
        app = self.app
        app.player.seek(9.)
        app._refresh_playback()
        self.assertAlmostEqual(app.pos_var.get(),9.)
        app._stop_play()
        self.assertEqual(app.pos_var.get(),0.)
        self.assertIn('0:00',app.time_label.cget('text'))
        self.assertEqual(app.play_btn.cget('text'),'⏵')

    def test_clock_poll_advances_bar_and_freezes_after_pause(self):
        app = self.app
        self.assertTrue(app.player.play())
        stream = FakeStream.instances[-1]
        stream.emit(16000)
        stream.time = 10.1+2.25
        app.after(80,app.quit)
        app.mainloop()  # Exercises the actual scheduled UI poll.
        self.assertAlmostEqual(app.pos_var.get(),2.25)
        self.assertTrue(app.playhead._bars[0].winfo_ismapped())
        app._toggle_play()
        frozen = app.player.position
        stream.time += 2
        app._refresh_playback()
        self.assertAlmostEqual(app.player.position,frozen)
        self.assertEqual(app.play_btn.cget('text'),'⏵')

    def test_new_audio_hides_stale_tab_cursor(self):
        app = self.app
        other = Path(self.temp.name)/'other.wav'
        sf.write(other,np.zeros(2000),1000)
        app._load_for_playback(str(other))
        app.update()
        self.assertIsNone(app.playhead.timing)
        self.assertFalse(any(b.winfo_ismapped() for b in app.playhead._bars))
        self.assertEqual(app._current_text(),self.content)

    def test_generation_tempo_remains_when_form_bpm_is_edited(self):
        app = self.app
        app.bpm_var.set(200)
        self.assertEqual(app.playhead.timing.bpm,120)
        app.player.seek(8.25)
        app._refresh_playback()
        app.update()
        self.assertIsNotNone(app.text.dlineinfo(f'{app.playhead.blocks[1].rows[-1]}.0'))

    def setup_sources(self, stereo=False):
        original = Path(self.temp.name)/'original.wav'
        separated = Path(self.temp.name)/'guitar.wav'
        a = np.tile([.125,.25],(24000,1)) if stereo else np.full(24000,.125)
        b = np.tile([.375,.5],(24000,1)) if stereo else np.full(24000,.375)
        sf.write(original,a,1000,subtype='FLOAT')
        sf.write(separated,b,1000,subtype='FLOAT')
        self.app._done(self.content,'','',str(separated),self.opts,
                       {'original':str(original),'separated':str(separated)})
        self.app.update()
        return original,separated

    def test_generation_defaults_to_original_not_separated_and_does_not_autoplay(self):
        self.app.player.play()
        previous = FakeStream.instances[-1]
        original,_ = self.setup_sources()
        self.assertTrue(previous.closed)
        self.assertFalse(self.app.player.playing)
        self.assertEqual(self.app.audio_path,str(original))
        self.assertEqual(self.app.audio_source_var.get(),'元の音源')
        self.assertEqual(self.app.audio_file_var.get(),'original.wav')
        np.testing.assert_array_equal(self.app.player.data[:,0],np.full(24000,.125))

    def test_switch_source_while_playing_closes_previous_and_preserves_time(self):
        app = self.app
        self.setup_sources()
        app.player.play()
        original_stream = FakeStream.instances[-1]
        original_stream.emit(16000)
        original_stream.time = 12.35
        app.audio_source_var.set('分離後の音源')
        app._switch_audio_source()
        current = FakeStream.instances[-1]
        self.assertTrue(original_stream.closed)
        self.assertTrue(app.player.playing)
        self.assertAlmostEqual(app.player.position,2.25,delta=.001)
        self.assertIsNotNone(app.playhead.timing)
        np.testing.assert_array_equal(current.emit(8).ravel(),np.full(8,.375))
        np.testing.assert_array_equal(original_stream.emit(8).ravel(),np.zeros(8))
        self.assertEqual(FakeStream.peak_active,1)

    def test_paused_source_switch_stays_paused_and_both_paths_keep_tab_sync(self):
        app = self.app
        original,_ = self.setup_sources()
        app.player.seek(3)
        for name in ['分離後の音源','元の音源']:
            app.audio_source_var.set(name)
            app._switch_audio_source()
            self.assertFalse(app.player.playing)
            self.assertAlmostEqual(app.player.position,3)
            self.assertIsNotNone(app.playhead.timing)
        self.assertEqual(app.audio_path,str(original))
        self.assertFalse(FakeStream.instances)

    def test_stop_after_switch_leaves_no_output_active(self):
        app = self.app
        self.setup_sources()
        app.player.play()
        for name in ['分離後の音源','元の音源','分離後の音源']:
            app.audio_source_var.set(name)
            app._switch_audio_source()
        app._stop_play()
        self.assertFalse(any(s.active for s in FakeStream.instances))
        self.assertEqual(app.player.position,0)
        self.assertEqual(FakeStream.peak_active,1)

    def test_selecting_same_source_does_not_restart_it(self):
        app = self.app
        self.setup_sources()
        app.player.play()
        app._switch_audio_source()
        self.assertEqual(len(FakeStream.instances),1)

    def test_stereo_output_contains_only_the_selected_source(self):
        app = self.app
        self.setup_sources(stereo=True)
        app.player.play()
        np.testing.assert_array_equal(FakeStream.instances[-1].emit(8),np.tile([.125,.25],(8,1)))
        app.audio_source_var.set('分離後の音源')
        app._switch_audio_source()
        np.testing.assert_array_equal(FakeStream.instances[-1].emit(8),np.tile([.375,.5],(8,1)))
        self.assertEqual(FakeStream.peak_active,1)

    def test_generation_start_stops_preview(self):
        self.app.player.play()
        self.app.path_var.set(str(self.path))
        with patch('main.threading.Thread') as thread:
            self.app._run()
            thread.return_value.start.assert_called_once()
        self.assertFalse(self.app.player.playing)
        self.assertFalse(any(s.active for s in FakeStream.instances))


if __name__ == '__main__':
    unittest.main()
