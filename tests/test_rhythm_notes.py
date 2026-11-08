import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pretty_midi
import soundfile as sf

from audio_fixtures import SAMPLE_RATE, guitar_scene
from tabgen.rhythm import ChordChange, arrange_rhythm
from tabgen.tab import TabOptions, notes_to_tab
from tabgen.tabpdf import data_path, _layout, tab_to_pdf


def note(pitch,start,end=None,confidence=.9):
    return dict(pitch=pitch,start=start,end=start+.2 if end is None else end,confidence=confidence)


class DetectedRhythmTests(unittest.TestCase):
    def test_arpeggio_keeps_attacks_and_pitches_instead_of_a_held_voicing(self):
        notes=[note(p,i*.25) for i,p in enumerate([48,52,55,60]*2)]
        change=[ChordChange(0,2,0,'',(0,4,7))]
        arranged,_=arrange_rhythm(notes,TabOptions(rhythm_mode='notes'),changes=change)
        self.assertEqual([(n['start'],n['pitch']) for n in arranged],
                         [(n['start'],n['pitch']) for n in notes])
        self.assertTrue(all('chord' not in n for n in arranged))
        held,_=arrange_rhythm(notes,TabOptions(rhythm_mode='chords'),changes=change)
        self.assertEqual({n['start'] for n in held},{0})
        self.assertEqual({n['end'] for n in held},{2})

    def test_repeated_picking_and_rest_are_not_replaced_or_filled(self):
        raw=[note(52,t) for t in [0,.25,.5,2,2.25]]
        opts=TabOptions(rhythm_mode='notes')
        arranged,_=arrange_rhythm(raw,opts,changes=[ChordChange(0,3,0,'',(0,4,7))])
        self.assertEqual([n['start'] for n in arranged],[0,.25,.5,2,2.25])
        self.assertTrue(all(n['end']<=n['start']+.2+1e-9 for n in arranged))
        self.assertNotIn('~~',notes_to_tab(arranged,opts,arranged=True))

    def test_unknown_or_missing_harmony_does_not_suppress_detected_notes(self):
        notes=[note(49,.25),note(54,.5)]
        for changes in [[],[ChordChange(0,1)],None]:
            with self.subTest(changes=changes):
                arranged,_=arrange_rhythm(notes,TabOptions(rhythm_mode='notes'),changes=changes)
                self.assertEqual([n['pitch'] for n in arranged],[49,54])

    def test_chord_metadata_does_not_turn_attacks_into_a_held_chord(self):
        original=dict(note(52,0,1),chord='C')
        arranged,_=arrange_rhythm([original],TabOptions(rhythm_mode='notes'),changes=[])
        self.assertNotIn('chord',arranged[0])
        self.assertEqual(original['chord'],'C')

    def test_playability_quantization_and_confidence_still_apply(self):
        raw=[note(42,.28),note(54,.53),note(127,.8),note(60,1.,confidence=.1)]
        opts=TabOptions(rhythm_mode='notes',capo=2,tuning=[64,59,55,50,45,38],grid_offset=.03)
        before=[dict(n) for n in raw]
        arranged,_=arrange_rhythm(raw,opts,changes=[])
        self.assertEqual([n['pitch'] for n in arranged],[42,54])
        self.assertEqual([n['start'] for n in arranged],[.28,.53])
        for n in arranged:
            self.assertEqual(n['pitch'],opts.tuning[n['string']]+opts.capo+n['fret'])
        self.assertEqual(raw,before)

    def test_invalid_mode_fails_before_audio_processing(self):
        from tabgen.pipeline import generate
        with self.assertRaisesRegex(ValueError,'rhythm mode'):
            generate('missing.wav',TabOptions(rhythm_mode='bad'))

    def test_pipeline_excludes_selected_lead_and_preserves_rhythm_in_all_exports(self):
        from tabgen.pipeline import generate
        audio,notes,labels=guitar_scene(lead=True)
        backing=[n for n,l in zip(notes,labels) if not l]
        changes=[ChordChange(0,1.5,0,'',(0,4,7)),ChordChange(1.5,4)]
        opts=TabOptions(rhythm_mode='notes')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'scene.wav';sf.write(source,audio,SAMPLE_RATE)
            diagnostics={}
            with patch('tabgen.transcribe.transcribe',return_value=notes), \
                 patch('tabgen.harmony.audio_chord_changes',return_value=changes):
                text,midi,sheets,_=generate(str(source),opts,split_lead_rhythm=True,
                                           auto_bpm=False,diagnostics=diagnostics)
            expected=[(n['pitch'],n['start']) for n in backing]
            self.assertEqual([(n['pitch'],n['start']) for n in diagnostics['rhythm']],expected)
            self.assertEqual(diagnostics['rhythm_candidates'],backing)
            rhythm_midi=midi.split(';')[1]
            midi_notes=[n for part in pretty_midi.PrettyMIDI(rhythm_midi).instruments for n in part.notes]
            self.assertEqual([(n.pitch,n.start) for n in midi_notes],expected)
            sheet=sheets.split(';')[1]
            xml=ET.parse(sheet)
            self.assertEqual(len(xml.findall('.//note/pitch')),len(backing))
            self.assertEqual(len(xml.findall('.//harmony')),2)
            data=json.loads(data_path(sheet).read_text())
            self.assertEqual(data['rhythm_mode'],'notes')
            self.assertEqual(len(data['notes']),len(backing))
            self.assertEqual([c['name'] for c in data['chords']],['C','N.C.'])
            steps,_,attacks,holds,_=_layout(data)
            self.assertEqual(steps,16)
            self.assertEqual(len(attacks),len(backing))
            self.assertEqual(holds,{})
            for n,saved in zip(diagnostics['rhythm'],data['notes']):
                self.assertEqual((saved['start'],saved['string'],saved['fret']),
                                 (n['start'],n['string'],n['fret']))
            pdf=Path(tmp)/'rhythm.pdf';tab_to_pdf(sheet,str(pdf))
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
        rhythm_text=text.split('====== リズムギター ======')[1]
        self.assertIn('検出した発音を保持',rhythm_text)
        self.assertNotIn('~~',rhythm_text)
        self.assertEqual([n['pitch'] for n in diagnostics['lead']],[66,64,62,64,66])

    def test_chord_annotations_alone_do_not_create_sounding_notes(self):
        from tabgen.sheet import export_musicxml
        from tabgen.tabpdf import export_tab_data
        opts=TabOptions(rhythm_mode='notes')
        changes=[ChordChange(0,1,0,'',(0,4,7)),ChordChange(1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            xml=Path(tmp)/'empty.musicxml'
            export_musicxml([],str(xml),opts,duration=2,chord_changes=changes)
            tree=ET.parse(xml)
            self.assertEqual(tree.findall('.//note/pitch'),[])
            self.assertTrue(tree.findall('.//note/rest'))
            export_tab_data([],str(xml),opts,'rhythm',changes,2)
            self.assertEqual(_layout(json.loads(data_path(xml).read_text()))[2],{})


if __name__ == '__main__':
    unittest.main()
