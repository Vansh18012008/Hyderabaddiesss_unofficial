import unittest
import numpy as np
from peak_grouping import PeakGrouper,GroupedTracker
from walking_confidence_tracker import ConfidenceTracker
A=np.arange(-60,61);P=np.ones(256)
def d(r,a=0):return dict(range_bin=r,angle_deg=a,strong=True,score_db=0.)
def track(i,r):return dict(id=i,pos=np.array([r,0.]),vel=np.zeros(2),last=0,confirmed=True)
class GroupTests(unittest.TestCase):
 def test_connected_weighted_center(self):
  p=P.copy();p[30]=1;p[38]=3;g=PeakGrouper().group([d(30),d(38)],np.zeros((121,256)),p,A,0,[])
  self.assertEqual(len(g),1);self.assertEqual(g[0]['member_count'],2);self.assertEqual(g[0]['range_bin'],36)
 def test_complete_linkage_prevents_chain(self):
  g=PeakGrouper().group([d(30),d(45),d(60)],np.zeros((121,256)),P,A,0,[]);self.assertEqual(len(g),2)
 def test_persistent_valley_separates(self):
  gr=PeakGrouper();z=np.full((121,256),-5.);z[60,30]=z[60,42]=0
  self.assertEqual(len(gr.group([d(30),d(42)],z,P,A,0,[])),1)
  self.assertEqual(len(gr.group([d(30),d(42)],z,P,A,1,[])),2)
 def test_different_motion_separates(self):
  gr=PeakGrouper();gr.next_id=3;gr.previous=[dict(id=1,pos=np.array([30.,0]),vel=np.array([10.,0]),age=3,last=0),dict(id=2,pos=np.array([40.,0]),vel=np.array([-10.,0]),age=3,last=0)]
  self.assertEqual(len(gr.group([d(30),d(40)],np.zeros((121,256)),P,A,1,[])),2);self.assertEqual(gr.audit[0]['reason'],'different_motion')
 def test_established_ids_stay_separate(self):
  gr=PeakGrouper();g=gr.group([d(30),d(40)],np.zeros((121,256)),P,A,1,[track(1,30),track(2,40)]);self.assertEqual(len(g),2);self.assertEqual({q['owner'] for q in g},{1,2})
 def test_unresolved_shared_peak_is_ambiguous(self):
  g=PeakGrouper().group([d(35)],np.zeros((121,256)),P,A,1,[track(1,30),track(2,40)]);self.assertTrue(g[0]['ambiguous']);self.assertEqual(g[0]['possible_track_ids'],[1,2])
 def test_owner_blocks_identity_reassignment(self):
  t=ConfidenceTracker()
  for k in range(3):t.step([d(30)],k)
  v,l=t.step([dict(d(31),owner=99)],3);self.assertIn(1,l);self.assertFalse(any(q['id']==1 for q in v))
 def test_strong_turn_retains_id_but_far_jump_does_not(self):
  t=ConfidenceTracker(allow_turns=True)
  for k,r in enumerate([10,25,40]):t.step([d(r)],k)
  v,_=t.step([d(25)],3);self.assertEqual(v[0]['id'],1)
  v,l=t.step([d(200)],4);self.assertIn(1,l);self.assertFalse(any(q['id']==1 for q in v))
 def test_single_person_view_and_no_forced_position(self):
  t=GroupedTracker(1)
  for k in range(5):
   q=t.step([d(30+4*k),d(38+4*k)],np.zeros((121,256)),P,A,k);self.assertLessEqual(len(q['person_candidates']),1)
  self.assertEqual(len(q['person_candidates']),1)
  q=t.step([],np.zeros((121,256)),P,A,5);self.assertFalse(q['person_candidates'])
if __name__=='__main__':unittest.main()
