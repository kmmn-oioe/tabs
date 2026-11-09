from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tabgen.lead_audio import NoteAudioFeatures
from tabgen.tab import TabOptions, arrange_melody, arrange_notes, playable
from tabgen.unison import recover_unisons
from benchmarks.audit_rhythm import audit_rhythm


class UnisonTests(unittest.TestCase):
    def scene(self, *, unison=True, mirror=False, phase=0., mono=False, same_pan=False):
        sr=22050
        audio=np.zeros((8*sr,2))
        raw,features,rhythm,lead=[],[],[],[]
        opts=TabOptions(bpm=120,rhythm_mode='notes',capo=2,tuning=[64,59,55,50,45,38])
        def add(pitch,start,part):
            pan=(.8 if same_pan else -.8) if part=='rhythm' else .8
            if mirror:
                pan=-pan
            n=dict(pitch=pitch,start=start,end=start+.18,confidence=.8+len(raw)*.0001)
            raw.append(n)
            features.append(NoteAudioFeatures(pan,np.array([1.,0.,0.,0.,0.]),1.,1.,True,spatial_rank=.5))
            (rhythm if part=='rhythm' else lead).append(n)
            sound(pitch,start,pan,phase if part=='lead' else 0.)
            return n
        def sound(pitch,start,pan,channel_phase):
            t=np.arange(round(.18*sr))/sr
            angle=2*np.pi*440*2**((pitch-69)/12)*t
            envelope=.15*np.minimum(1,t/.005)*np.minimum(1,(.18-t)/.01)
            gains=np.sqrt(np.array([1-pan,1+pan])/2)
            for ch in range(2):
                offset=channel_phase if ch==1 else 0.
                wave=sum(level*np.sin(h*(angle+offset)) for h,level in ((1,1.),(2,.25),(3,.1)))
                first=round(start*sr)
                audio[first:first+len(t),ch]+=wave*envelope*gains[ch]
        for cycle in range(7):
            start=.5+cycle
            if cycle<3:
                add(64,start,'rhythm')
            add(57,start+.25,'rhythm')
            add(60,start+.5,'rhythm')
        target=add(64,3.5,'lead')
        if unison:
            sound(64,3.5,-.8 if not mirror else .8,0.)
        for start in (4.625,5.625,6.625):
            add(64,start,'lead')
        return (audio.mean(axis=1) if mono else audio),sr,raw,features,arrange_melody(lead,opts),arrange_notes(rhythm,opts),opts,target

    def run_scene(self, **kwargs):
        audio,sr,raw,features,lead,rhythm,opts,target=self.scene(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav'
            sf.write(path,audio,sr,subtype='FLOAT')
            result,added,checks=recover_unisons(str(path),lead,rhythm,raw,features,opts)
        return result,added,checks,lead,rhythm,opts,target

    def test_two_sources_restore_unison_with_phase_and_channel_variations(self):
        for mirror in (False,True):
            for phase in (0.,np.pi,np.pi/2):
                with self.subTest(mirror=mirror,phase=phase):
                    result,added,checks,lead,rhythm,opts,target=self.run_scene(mirror=mirror,phase=phase)
                    self.assertEqual([(n['pitch'],n['start']) for n in added],[(64,3.5)])
                    self.assertEqual([n for n in result if n not in added],rhythm)
                    self.assertTrue(playable(tuple((n['string'],n['fret']) for n in added),opts))
                    self.assertEqual(opts.tuning[added[0]['string']]+opts.capo+added[0]['fret'],64)

    def test_motif_alone_cannot_copy_lead_into_a_real_accompaniment_rest(self):
        for phase in (0.,np.pi,np.pi/2):
            with self.subTest(phase=phase):
                result,added,checks,lead,rhythm,_,_=self.run_scene(unison=False,phase=phase)
                self.assertEqual(added,[])
                self.assertEqual(result,rhythm)
                self.assertTrue(any(c.get('reason')=='only_one_source_supported' for c in checks))

    def test_mono_or_indistinguishable_sources_are_withheld(self):
        for options in (dict(mono=True),dict(same_pan=True)):
            with self.subTest(options=options):
                result,added,_,_,rhythm,_,_=self.run_scene(**options)
                self.assertEqual(added,[])
                self.assertEqual(result,rhythm)

    def test_no_lead_or_no_repeated_motif_does_not_create_a_second_part(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        self.assertEqual(recover_unisons('unused.wav',[],rhythm,raw,features,opts)[1],[])
        # Drop accompaniment target anchors, so only the following bass notes remain.
        rhythm=[n for n in rhythm if n['pitch']!=64]
        self.assertEqual(recover_unisons('unused.wav',lead,rhythm,raw,features,opts)[1],[])

    def test_existing_unison_is_not_duplicated_or_used_recursively(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            first,added,_=recover_unisons(str(path),lead,rhythm,raw,features,opts)
            second,again,_=recover_unisons(str(path),lead,first,raw,features,opts)
        self.assertEqual(again,[])
        self.assertEqual(first,second)

    def test_unplayable_restoration_does_not_remove_existing_notes(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        # Force a full six-string shape at the candidate cell.
        blockers=[dict(pitch=p+opts.capo,start=3.5,end=3.68,confidence=.7,
                       string=s,fret=0) for s,p in enumerate(opts.tuning)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            result,added,checks=recover_unisons(str(path),lead,rhythm+blockers,raw,features,opts)
        self.assertEqual(added,[])
        self.assertTrue(all(n in result for n in rhythm+blockers))
        self.assertTrue(any(c.get('reason')=='no_playable_position' for c in checks))

    def test_audit_traces_shared_note_to_original_detection_with_unique_proof(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            result,added,checks=recover_unisons(str(path),lead,rhythm,raw,features,opts)
        lead_ids={(n['pitch'],n['confidence']) for n in lead}
        input_notes=[n for n in raw if (n['pitch'],n['confidence']) not in lead_ids]
        diagnostics=dict(options=asdict(opts),transcribed=raw,rhythm_candidates=input_notes,
                         rhythm=result,rhythm_unison_restored=added,unison_checks=checks)
        case=dict(id='fixture',evaluation_window=[0.,8.],notes=[])
        audited=audit_rhythm(case,diagnostics)
        self.assertTrue(audited['provenance_replay_identical'])
        self.assertEqual(len(audited['extras']),len(result))
        self.assertEqual(audited['confidence_sweep'][0]['predicted_notes'],len(result))
        diagnostics['unison_checks']=[]
        with self.assertRaisesRegex(ValueError,'unique proof'):
            audit_rhythm(case,diagnostics)
