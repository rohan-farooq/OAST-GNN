"""Simulator-independent target-claim interventions; truth stays in evaluation logs."""
import math
import random

ATTACKS = ('a1_constant_position', 'a2_constant_offset', 'a3_random_offset',
           'a4_eventual_stop', 'a5_collusive_offset')


def select_coalition(positions, fraction, seed, radius):
    ids = sorted(positions)
    count = int(round(len(ids) * fraction))
    if fraction > 0 and count == 0:
        count = 1
    if count < 2:
        raise ValueError('A5 requires at least two malicious targets.')
    rng = random.Random(f'coalition:{seed}')
    anchor = rng.choice(ids)
    x, y = positions[anchor]
    candidates = [i for i in ids if i != anchor and
                  math.hypot(positions[i][0]-x, positions[i][1]-y) <= radius]
    if len(candidates) < count-1:
        raise ValueError('Insufficient vehicles inside coalition radius; do not silently change fraction.')
    members = {anchor, *rng.sample(candidates, count-1)}
    return members, dict(anchor_id=anchor, radius_m=radius,
                        selection='uniform anchor then uniform subset within anchor radius',
                        reference_positions={str(i): positions[i] for i in ids})


class TargetAttackEngine:
    def __init__(self, attack, seed, vehicle_ids, members, offset_bound=50.0, stop_seconds=5.0):
        if attack not in ATTACKS or offset_bound <= 0 or stop_seconds <= 0:
            raise ValueError('Invalid target attack settings')
        self.attack, self.seed = attack, seed
        self.ids, self.members = list(sorted(vehicle_ids)), set(members)
        self.bound, self.stop_seconds = offset_bound, stop_seconds
        self.initial = {}
        self.last = {}
        self.shared = self.offset('shared')

    def offset(self, key):
        rng = random.Random(f'target-attack:{self.seed}:{key}')
        return [rng.uniform(-self.bound, self.bound) for _ in range(2)]

    def claim(self, actor_id, sequence, now, true):
        if actor_id not in self.ids or not all(math.isfinite(float(true[k])) for k in ['x','y','speed','heading']):
            raise ValueError('Invalid true state')
        if not math.isfinite(now) or true['speed'] < 0:
            raise ValueError('Invalid time/speed')
        previous = self.last.get(actor_id)
        if previous and (sequence <= previous[0] or now < previous[1]):
            raise ValueError('Nonchronological target beacon')
        self.last[actor_id] = (sequence, now)
        claimed = dict(true)
        if actor_id not in self.members:
            return claimed
        index = self.ids.index(actor_id)
        initial = self.initial.setdefault(actor_id, dict(state=dict(true), time_s=now))
        first = initial['state']
        if self.attack == 'a1_constant_position':
            # Position frozen; speed/heading left as reported by original state.
            claimed.update(x=first['x'], y=first['y'])
        elif self.attack in ['a2_constant_offset','a3_random_offset','a5_collusive_offset']:
            if self.attack == 'a5_collusive_offset':
                dx, dy = self.shared
            else:
                key = f'actor:{index}' if self.attack == 'a2_constant_offset' else f'actor:{index}:seq:{sequence}'
                dx, dy = self.offset(key)
            claimed.update(x=true['x']+dx, y=true['y']+dy)
        elif self.attack == 'a4_eventual_stop':
            # Explicit operationalization: constant-heading linear deceleration.
            t = min(max(0., now-initial['time_s']), self.stop_seconds)
            distance = first['speed']*(t-t*t/(2*self.stop_seconds))
            angle = math.radians(first['heading'])
            claimed.update(x=first['x']+distance*math.cos(angle),
                           y=first['y']+distance*math.sin(angle),
                           speed=first['speed']*(1-t/self.stop_seconds), heading=first['heading'])
        return claimed

    def describe(self):
        return dict(schema='tre-inspired-target-attacks-v1', attack=self.attack, seed=self.seed,
                    vehicle_ids=self.ids, malicious_target_ids=sorted(self.members),
                    actual_target_fraction=len(self.members)/len(self.ids),
                    offset_bound_m=self.bound, stop_seconds=self.stop_seconds,
                    shared_offset_xy_m=self.shared if self.attack=='a5_collusive_offset' else None,
                    onset='first attempted beacon per malicious target after warmup',
                    initial_states={str(k):v for k,v in self.initial.items()},
                    label_policy='assigned malicious identity throughout episode, including initial zero-displacement frames',
                    a4_profile='linear speed reduction, integrated fixed initial heading; 5s default is a study choice',
                    scope='CARLA implementation inspired by TRE attack definitions, not exact dataset/protocol replication')
