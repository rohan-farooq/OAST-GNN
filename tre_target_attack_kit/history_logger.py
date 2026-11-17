"""Append-only operational histories and separate attack instrumentation."""
import json
from pathlib import Path

class HistoryLogger:
    def __init__(self, output, seed):
        self.seed=seed;self.handles={};self.counts={};self.last_frame=None
        self.paths={k:Path(str(output)+suffix) for k,suffix in {
            'beacons':'.beacons.jsonl','self':'.self.jsonl','attacks':'.attacks.evaluation.jsonl'}.items()}
        for p in self.paths.values():
            if p.exists():raise FileExistsError(f'History already exists: {p}')
        try:
            for k,p in self.paths.items():
                p.parent.mkdir(parents=True,exist_ok=True);self.handles[k]=p.open('x',encoding='utf-8');self.counts[k]=0
        except Exception:
            self.close();raise
    def emit(self,kind,record):
        self.handles[kind].write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
        self.counts[kind]+=1
        if self.counts[kind]%64==0:self.handles[kind].flush()
    def beacon(self,received,now):
        # Only decoded wire data, delivery time and channel power; no target truth.
        self.emit('beacons',dict(schema='a6-observer-beacon-v1',seed=self.seed,
            observer_id=int(received.receiver_id),received_time_s=float(received.timestamp),
            processed_time_s=float(now),received_power_dbm=float(received.power_dbm),payload=dict(received.payload)))
    def self_states(self,snapshot,vehicles):
        if snapshot.frame==self.last_frame:return
        self.last_frame=snapshot.frame
        for vehicle in vehicles:
            state=snapshot.find(vehicle.id)
            if state is None:continue
            tr=state.get_transform();v=state.get_velocity()
            self.emit('self',dict(schema='a6-observer-self-v1',seed=self.seed,observer_id=vehicle.id,
                frame=snapshot.frame,time_s=float(snapshot.timestamp.elapsed_seconds),
                x_m=tr.location.x,y_m=tr.location.y,vx_mps=v.x,vy_mps=v.y,vz_mps=v.z,
                heading_deg=tr.rotation.yaw,
                source='simulated ideal ego localization; not independent target truth'))
    def attack(self,report_id,reporter,target,sequence,now,elapsed,attack,byzantine,malicious,period,local,submitted):
        active=bool(byzantine and attack!='none')
        if attack=='on_off':active=active and int(elapsed/max(period,.1))%2==1
        if attack=='false_accusation':active=active and not malicious
        if attack=='false_reassurance':active=active and malicious
        self.emit('attacks',dict(schema='a6-attack-instrumentation-v1',seed=self.seed,
            report_id=report_id,reporter_id=reporter,target_id=target,beacon_sequence=sequence,
            created_time_s=now,elapsed_time_s=elapsed,reporter_attack=attack,
            reporter_is_byzantine=bool(byzantine),target_is_malicious=bool(malicious),
            attack_active=bool(active),score_changed=abs(local-submitted)>1e-10,
            local_score=local,submitted_score=submitted))
    def close(self):
        for f in self.handles.values():
            if not f.closed:f.flush();f.close()
