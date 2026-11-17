"""Meaningful dynamics checks with synthetic moving vehicles; not paper results."""
import math,unittest
from target_attacks import TargetAttackEngine,select_coalition

class Tests(unittest.TestCase):
    def engine(self,attack):return TargetAttackEngine(attack,2047,[10,20,30],{10,20},50,5)
    def state(self,t):return dict(x=10*t,y=0.,speed=10.,heading=0.)
    def test_honest(self):
        e=self.engine('a1_constant_position');s=self.state(3)
        self.assertEqual(e.claim(30,1,3,s),s)
    def test_a1_freezes_position_not_physical_vehicle(self):
        e=self.engine('a1_constant_position');e.claim(10,1,0,self.state(0))
        self.assertEqual(e.claim(10,2,2,self.state(2))['x'],0)
    def test_a2_constant_and_reproducible(self):
        e=self.engine('a2_constant_offset');f=self.engine('a2_constant_offset')
        a=e.claim(10,1,0,self.state(0));b=e.claim(10,2,2,self.state(2))
        self.assertAlmostEqual(a['x'],b['x']-20);self.assertEqual(a,f.claim(10,1,0,self.state(0)))
    def test_a3_changes(self):
        e=self.engine('a3_random_offset');a=e.claim(10,1,0,self.state(0));b=e.claim(10,2,1,self.state(1))
        self.assertNotEqual(a['x'],b['x']-10)
        self.assertLessEqual(abs(a['x']),50);self.assertLessEqual(abs(a['y']),50)
    def test_a4_integrated_stop(self):
        e=self.engine('a4_eventual_stop');e.claim(10,1,0,self.state(0))
        mid=e.claim(10,2,2.5,self.state(2.5));end=e.claim(10,3,8,self.state(8))
        self.assertEqual(mid['speed'],5);self.assertEqual(mid['x'],18.75)
        self.assertEqual(end['speed'],0);self.assertEqual(end['x'],25)
    def test_a5_preserves_relative_positions(self):
        e=self.engine('a5_collusive_offset')
        a=e.claim(10,1,0,self.state(0));b=e.claim(20,1,0,self.state(3))
        self.assertAlmostEqual(b['x']-a['x'],30);self.assertAlmostEqual(b['y']-a['y'],0)
    def test_coalition_and_fail_closed(self):
        members,info=select_coalition({1:(0,0),2:(10,0),3:(20,0),4:(30,0)},.5,4,500)
        self.assertEqual(len(members),2)
        with self.assertRaises(ValueError):select_coalition({1:(0,0),2:(1000,0)},1,4,1)
    def test_no_backwards_time(self):
        e=self.engine('a2_constant_offset');e.claim(10,2,2,self.state(2))
        with self.assertRaises(ValueError):e.claim(10,1,1,self.state(1))

if __name__=='__main__':unittest.main()
