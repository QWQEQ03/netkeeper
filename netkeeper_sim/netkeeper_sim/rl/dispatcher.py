from __future__ import annotations
import hashlib
import torch
from netkeeper_sim.api.models import OptimizationRequest
from netkeeper_sim.schemas import JointAction, NetworkSnapshot
from netkeeper_sim.rl.config import GraphNetworkConfig
from netkeeper_sim.rl.networks.multi_agent_actor import MultiAgentActor
from netkeeper_sim.rl.schema_adapter import action_masks,candidate_to_joint_action,snapshot_to_graph

class TrainedPolicyDispatcher:
    def __init__(self, checkpoint:str, *, expected_dataset_manifest:str|None=None):
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if data.get('model_version')!='rl-coma-v2': raise ValueError({'code':'MODEL_INCOMPATIBLE','message':'unsupported checkpoint'})
        if expected_dataset_manifest is not None:
            digest=hashlib.sha256(open(expected_dataset_manifest,'rb').read()).hexdigest()
            saved=(data.get('provenance') or {}).get('dataset_manifest_sha256')
            if saved != digest: raise ValueError({'code':'DATASET_INCOMPATIBLE','message':'dataset manifest hash mismatch'})
        c=data['config']; self.actor=MultiAgentActor(GraphNetworkConfig(17,11,c['hidden_dim'],c['gnn_layers'],c['transformer_layers'],c['heads'],c['dropout']))
        self.actor.load_state_dict(data['actor'],strict=True); self.actor.eval()
    def dispatch(self,snapshot:NetworkSnapshot,request:OptimizationRequest)->JointAction:
        if request.snapshot_id!=snapshot.snapshot_id: raise ValueError({'code':'STALE_SNAPSHOT','message':'request does not match snapshot'})
        graph=snapshot_to_graph(snapshot); masks=action_masks(snapshot,graph)
        with torch.no_grad(): out=self.actor.forward_graph(graph,masks)
        choices={agent:int(torch.where(out.masks[agent],out.logits[agent],torch.tensor(float('-inf'))).argmax()) for agent in out.logits}
        return candidate_to_joint_action(snapshot,graph,choices)
