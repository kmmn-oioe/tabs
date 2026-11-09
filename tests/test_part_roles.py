from pathlib import Path
from dataclasses import asdict
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tabgen.lead_audio import NoteAudioFeatures
from tabgen.part_roles import refine_part_roles
from tabgen.tab import TabOptions, arrange_melody, arrange_notes
from benchmarks.audit_rhythm import audit_rhythm


class PartRoleTests(unittest.TestCase):
    def scene(self, *, signal=True, mirror=False, phase=0., shift=0,
              replacement=False, represented=True, stable=False, same_pan=False):
        sr=22050
        audio=np.zeros((8*sr,2))
        raw,features,lead,rhythm=[],[],[],[]
        opts=TabOptions(bpm=120,rhythm_mode='notes')
        def sound(pitch,start,pan,channel_phase):
            t=np.arange(round(.18*sr))/sr
            angle=2*np.pi*440*2**((pitch-69)/12)*t
            envelope=.15*np.minimum(1,t/.005)*np.minimum(1,(.18-t)/.01)
            gains=np.sqrt(np.array([1-pan,1+pan])/2)
            for ch in range(2):
                offset=channel_phase if ch else 0.
                wave=sum(level*np.sin(h*(angle+offset)) for h,level in ((1,1.),(2,.35),(3,.1)))
                first=round(start*sr)
                audio[first:first+len(t),ch]+=wave*envelope*gains[ch]
        def add(pitch,start,part, audible=True):
            pitch+=shift
            pan=.8 if part=='lead' or same_pan else -.8
            if mirror:pan=-pan
            n=dict(pitch=pitch,start=start,end=start+.18,confidence=.8+len(raw)*.0001)
            raw.append(n)
            features.append(NoteAudioFeatures(pan,np.array([1.,.3,.1,0.,0.])/np.sqrt(1.1),
                                              1.,.9,True,spatial_rank=.5))
            if part=='lead':lead.append(n)
            elif part=='rhythm':rhythm.append(n)
            if audible:sound(pitch,start,pan,phase if part=='lead' else 0.)
            return n
        for cycle in range(7):
            start=.5+cycle
            add(69,start,'lead')
            add(55,start,'rhythm')  # clean samples of the overlapping harmonic
            if cycle!=3:
                add(64 if stable and cycle<3 else 67,start+.25,'lead')
            add(62,start+.5,'lead')
        target=add(67,3.75,'candidate',False)
        rhythm.append(target)  # the transcriber heard the carrier's second harmonic
        features[-1].pan=.2*(-1 if mirror else 1)
        if signal:sound(67+shift,3.75,.8*(-1 if mirror else 1),phase)
        add(55,3.75,'rhythm')
        if replacement or stable:
            old=add(64,3.75,'lead')
            if represented:rhythm.append(old)
        return audio,sr,raw,features,arrange_melody(lead,opts),arrange_notes(rhythm,opts),opts,target

    def run_scene(self, **kwargs):
        audio,sr,raw,features,lead,rhythm,opts,target=self.scene(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            result=refine_part_roles(str(path),lead,rhythm,raw,features,opts)
        return result,lead,rhythm,target

    def test_actual_lead_at_overlapping_harmonic_moves_to_the_correct_part(self):
        for mirror in (False,True):
            for phase in (0.,np.pi,np.pi/2):
                with self.subTest(mirror=mirror,phase=phase):
                    (lead,rhythm,checks),before,backing,target=self.run_scene(mirror=mirror,phase=phase)
                    self.assertTrue(any(c['accepted'] for c in checks))
                    self.assertTrue(any(n['pitch']==67 and n['start']==3.75 for n in lead))
                    self.assertFalse(any(n['pitch']==67 and n['start']==3.75 for n in rhythm))
                    self.assertTrue(all(n in lead for n in before))
                    self.assertEqual(len(rhythm),len(backing)-1)

    def test_carrier_harmonic_alone_cannot_create_a_lead_even_when_the_motif_fits(self):
        for phase in (0.,np.pi/2):
            with self.subTest(phase=phase):
                (lead,rhythm,checks),before,backing,_=self.run_scene(signal=False,phase=phase)
                self.assertEqual(lead,before)
                self.assertEqual(rhythm,backing)
                self.assertFalse(any(c['accepted'] for c in checks))

    def test_replacement_requires_old_pitch_to_remain_in_backing(self):
        (lead,rhythm,checks),before,backing,_=self.run_scene(replacement=True)
        self.assertTrue(any(c['accepted'] for c in checks))
        self.assertTrue(any(n['pitch']==67 and n['start']==3.75 for n in lead))
        self.assertTrue(any(n['pitch']==64 and n['start']==3.75 for n in rhythm))
        (lead,rhythm,checks),before,backing,_=self.run_scene(replacement=True,represented=False)
        self.assertEqual(lead,before)
        self.assertEqual(rhythm,backing)

    def test_stable_existing_melody_is_protected_against_an_alternative_motif(self):
        (lead,rhythm,checks),before,backing,_=self.run_scene(stable=True)
        self.assertEqual(lead,before)
        self.assertEqual(rhythm,backing)

    def test_lower_register_and_repeated_execution_do_not_change_the_role_rule(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene(shift=-12)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            first,backing,checks=refine_part_roles(str(path),lead,rhythm,raw,features,opts)
            self.assertTrue(any(n['pitch']==55 and n['start']==3.75 for n in first))
            second,again,_=refine_part_roles(str(path),first,backing,raw,features,opts)
        self.assertEqual(first,second)
        self.assertEqual(backing,again)

    def test_mono_and_identical_spatial_sources_remain_ambiguous(self):
        for mono,same_pan in ((True,False),(False,True)):
            audio,sr,raw,features,lead,rhythm,opts,_=self.scene(same_pan=same_pan)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'scene.wav';sf.write(path,audio.mean(axis=1) if mono else audio,sr,subtype='FLOAT')
                after,backing,checks=refine_part_roles(str(path),lead,rhythm,raw,features,opts)
            self.assertEqual(after,lead)
            self.assertEqual(backing,rhythm)

    def test_no_lead_anchors_or_manual_entry_boundary_cannot_be_bypassed(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        self.assertEqual(refine_part_roles('unused.wav',[],rhythm,raw,features,opts),([],rhythm,[]))
        opts.lead_start_seconds=4.
        after,backing,_=refine_part_roles('unused.wav',lead,rhythm,raw,features,opts)
        self.assertEqual(after,lead)
        self.assertEqual(backing,rhythm)

    def test_audit_replays_role_removal_without_losing_original_note_provenance(self):
        audio,sr,raw,features,lead,rhythm,opts,_=self.scene()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.wav';sf.write(path,audio,sr,subtype='FLOAT')
            after,backing,checks=refine_part_roles(str(path),lead,rhythm,raw,features,opts)
        keys={(n['pitch'],n['confidence']) for n in rhythm}
        candidates=[n for n in raw if (n['pitch'],n['confidence']) in keys]
        diagnostics=dict(options=asdict(opts),transcribed=raw,rhythm_candidates=candidates,
                         rhythm=backing,part_role_checks=checks,
                         rhythm_role_removed=[n for c in checks if c['accepted'] for n in c['removed_rhythm']])
        case=dict(id='fixture',evaluation_window=[0.,8.],notes=[])
        result=audit_rhythm(case,diagnostics)
        self.assertTrue(result['provenance_replay_identical'])
        self.assertEqual(result['confidence_sweep'][0]['predicted_notes'],len(backing))
        diagnostics['part_role_checks']=[]
        with self.assertRaisesRegex(ValueError,'unique proof'):
            audit_rhythm(case,diagnostics)
