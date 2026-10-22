"""Equation, author-kernel, full-parameter and resume regressions for block EBD.

Run without pytest: PYTHONPATH=src python -m unittest discover -s tests -p test_ebd_vit.py
"""
import ast
import copy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from methods.ebd_vit.objectives import EBDObjectiveConfig, EBDActivityObjective
from methods.ebd_vit.learner import EBDConfig, EBDLearner, supervised_objective, example_error
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.configs import ViTConfig

ROOT = Path(__file__).resolve().parents[1]


def scalar_reference(h, e, c_before, r_before, cfg):
    """Independent scalar objective; all gradients in tests come from autograd."""
    b, _, d = h.shape
    c = cfg.correlation_beta*c_before + (1-cfg.correlation_beta)*torch.einsum("bc,bnd->cnd",e,h)/b
    r = cfg.covariance_beta*r_before + (1-cfg.covariance_beta)*torch.einsum("bni,bnj->nij",h,h)/b
    decorrelation = .5*c.square().sum()/(1-cfg.correlation_beta)
    entropy = -torch.logdet(r+cfg.entropy_epsilon*torch.eye(d,dtype=h.dtype,device=h.device)[None]).sum()/d/(1-cfg.covariance_beta)
    if cfg.power_mode == "author_sample":
        power = .25*(h.square()-cfg.power_target).square().sum()/b
    else:
        power = .25*(h.square().mean(0)-cfg.power_target).square().sum()
    sparse = h.abs().sum()/(b*d)
    return (cfg.error_scale*decorrelation + cfg.entropy_scale*entropy
            + cfg.power_scale*power + cfg.l1_scale*sparse)


def make_model(seed=0, patch_size=4):
    torch.manual_seed(193+seed)
    cfg = ViTConfig(image_size=8,patch_size=patch_size,num_classes=3,embed_dim=8,
                    depth=2,num_heads=2,ffn_dim=12)
    model = AuditedViT(cfg,"post").double()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer,torch.nn.LayerNorm):
                layer.weight.uniform_(.6,1.4)
                layer.bias.uniform_(-.2,.2)
    x = torch.randn(3,3,8,8,dtype=torch.float64)
    labels = torch.tensor([0,1,2])
    return model,x,labels


class EBDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def close(self, a, b, tolerance=1e-10):
        value = float((a-b).norm()/b.norm().clamp_min(1e-15))
        self.assertLess(value,tolerance)
        return value

    def test_activity_gradients_both_power_objectives(self):
        for seed in range(3):
            for mode in ("author_sample","paper_batch"):
                for beta in (.8,.999999):
                    torch.manual_seed(seed)
                    h = torch.randn(5,4,7,dtype=torch.float64,requires_grad=True)
                    e = torch.randn(5,3,dtype=torch.float64)
                    cfg = EBDObjectiveConfig(correlation_beta=beta,covariance_beta=.91,power_mode=mode)
                    state = EBDActivityObjective(3,4,7,cfg,generator=torch.Generator().manual_seed(seed),reference=h)
                    before = copy.deepcopy(state.state_dict())
                    state.update(e,h.detach())
                    actual,_ = state.credit(e,h.detach())
                    target = torch.autograd.grad(scalar_reference(h,e,before["correlation"],before["covariance"],cfg),h)[0]
                    self.close(actual,target)

    def test_batch_reduction_and_no_unit_normalization(self):
        torch.manual_seed(17)
        h = torch.randn(4,5,8,dtype=torch.float64)
        e = torch.randn(4,3,dtype=torch.float64)
        cfg = EBDObjectiveConfig(entropy_scale=0,l1_scale=0,power_scale=0)
        state = EBDActivityObjective(3,5,8,cfg,generator=torch.Generator().manual_seed(41),reference=h)
        duplicated = copy.deepcopy(state)
        state.update(e,h)
        duplicated.update(e.repeat(2,1),h.repeat(2,1,1))
        credit,_ = state.credit(e,h)
        doubled,_ = duplicated.credit(e.repeat(2,1),h.repeat(2,1,1))
        self.close(doubled[:4],credit/2)
        state.correlation.mul_(1e-6)
        tiny,_ = state.credit(e,h)
        self.close(tiny,credit*1e-6)

    def test_token_reduction_has_no_extra_divisor(self):
        torch.manual_seed(18)
        h = torch.randn(4,5,8,dtype=torch.float64)
        e = torch.randn(4,3,dtype=torch.float64)
        cfg = EBDObjectiveConfig()
        one = EBDActivityObjective(3,5,8,cfg,generator=torch.Generator().manual_seed(41),reference=h)
        two = EBDActivityObjective(3,10,8,cfg,generator=torch.Generator().manual_seed(41),reference=h)
        two.correlation.copy_(one.correlation.repeat(1,2,1))
        two.covariance.copy_(one.covariance.repeat(2,1,1))
        one.update(e,h)
        two.update(e,h.repeat(1,2,1))
        a,_ = one.credit(e,h)
        b,_ = two.credit(e,h.repeat(1,2,1))
        self.close(b[:,:5],a)

    def test_full_parameter_gradients_ce_mse_and_no_autograd_in_step(self):
        for seed,patch_size in enumerate((4,2,1)):
            for loss in ("cross_entropy","mse"):
                with self.subTest(seed=seed,loss=loss,tokens=(8//patch_size)**2+1):
                    model,x,labels = make_model(seed,patch_size)
                    cfg = EBDConfig(loss=loss,optimizer="sgd",learning_rate=.001,momentum=0,
                                    weight_decay=0,feedback_seed=100+seed)
                    learner = EBDLearner(model,cfg)
                    reference = copy.deepcopy(model)
                    with torch.no_grad():
                        cache = reference.forward_with_cache(x)
                        error = example_error(cache.logits,labels,loss)
                    z = cache.logits.detach().requires_grad_()
                    if loss == "cross_entropy":
                        supervised = F.cross_entropy(z,labels)
                    else:
                        supervised = .5*(z-F.one_hot(labels,3)).square().sum()/len(labels)
                    self.close(torch.autograd.grad(supervised,z)[0],error/len(labels))
                    gradients = {}
                    parameters = dict(reference.named_parameters())
                    for i,block in enumerate(reference.blocks):
                        tokens = reference.tokenize(x) if i == 0 else cache.block_caches[i].block_input.detach()
                        output,_ = block(tokens)
                        state = learner.objectives[i]
                        local = scalar_reference(output,error,state.correlation,state.covariance,cfg.objective)
                        names = [n for n in parameters if n.startswith(f"blocks.{i}.") or
                            (i == 0 and (n.startswith("patch_embed.") or n in ("cls_token","position")))]
                        values = torch.autograd.grad(local,[parameters[n] for n in names])
                        gradients.update(zip(names,values))
                    z = reference.head(reference.final_norm(cache.block_caches[-1].block_output.detach())[:,0])
                    names = [n for n in parameters if n.startswith(("head.","final_norm."))]
                    values = torch.autograd.grad(supervised_objective(z,labels,loss),[parameters[n] for n in names])
                    gradients.update(zip(names,values))
                    before = {n:p.clone() for n,p in model.named_parameters()}
                    with patch("torch.autograd.grad",side_effect=AssertionError("Training called autograd")), \
                         patch.object(torch.Tensor,"backward",side_effect=AssertionError("Training called backward")):
                        learner.step(x,labels)
                    self.assertEqual(set(gradients),set(before))
                    maximum = 0.
                    for name,p in model.named_parameters():
                        maximum = max(maximum,self.close(p.grad,gradients[name]))
                        self.close(p,before[name]-.001*gradients[name])
                    self.assertTrue(all(o.updates == 1 for o in learner.objectives))
                    self.assertEqual(learner.calibration_batches,0)
                    print(f"full-gradient seed={seed} loss={loss} tokens={(8//patch_size)**2+1} max_relative_error={maximum:.3g}")

    def test_matches_preserved_author_hidden_layer(self):
        path = ROOT/"tests/reference/ebd_gradients.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name == "LossFullyConnected")
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "forward")
        compatibility = SimpleNamespace(**vars(torch))
        compatibility.linalg = SimpleNamespace(**vars(torch.linalg))
        compatibility.linalg.solve = lambda a,b,left=True: torch.linalg.solve(a,b) if left else torch.linalg.solve(a.T,b.T).T
        namespace = {"torch":compatibility,"math":math}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
        for seed in range(3):
            torch.manual_seed(713+seed)
            x = torch.randn(7,4,dtype=torch.float64)
            w = torch.randn(5,4,dtype=torch.float64)
            u = x@w.T
            h = u.relu()
            e = torch.randn(7,3,dtype=torch.float64)
            cfg = EBDObjectiveConfig(correlation_beta=.9,covariance_beta=.8,power_scale=0)
            ours = EBDActivityObjective(3,1,5,cfg,generator=torch.Generator().manual_seed(seed),reference=h)
            author = SimpleNamespace(method="ebd",la_R=cfg.correlation_beta,la_R2=cfg.covariance_beta,
                Rehk=ours.correlation[:,0].clone(),Rehk0=ours.correlation[:,0].clone(),
                Rhk=ours.covariance[0].clone(),R_eps=torch.eye(5,dtype=h.dtype)*cfg.entropy_epsilon)
            values = namespace["forward"](author,h,u,e,x,0)
            ours.update(e,h[:,None])
            credit,_ = ours.credit(e,h[:,None])
            actual = ((u>0)*credit[:,0]).T@x
            expected = cfg.error_scale*values[4]+cfg.entropy_scale*values[6]+cfg.l1_scale*values[8]
            self.close(actual,expected,1e-7)  # Author entropy solve adds h+1e-8.
            self.close(ours.correlation[:,0],values[2])
            self.close(ours.covariance[0],values[3])

    def test_adam_resume_is_exact(self):
        model,x,y = make_model()
        learner = EBDLearner(model,EBDConfig())
        learner.step(x,y)
        saved = copy.deepcopy(learner.state_dict())
        restored = EBDLearner(copy.deepcopy(model),EBDConfig())
        restored.load_state_dict(saved)
        learner.step(x*.7,y.flip(0))
        restored.step(x*.7,y.flip(0))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(learner.model.parameters(),restored.model.parameters())))
        for a,b in zip(learner.objectives,restored.objectives):
            self.assertTrue(torch.equal(a.correlation,b.correlation))
            self.assertTrue(torch.equal(a.covariance,b.covariance))
            self.assertEqual(a.updates,b.updates)
        with self.assertRaises(ValueError):
            wrong = EBDLearner(copy.deepcopy(model),replace(EBDConfig(),loss="mse"))
            wrong.load_state_dict(saved)

    def test_invalid_configuration_rejected(self):
        for kwargs in ({"correlation_beta":1.},{"entropy_epsilon":0.},{"power_mode":"ambiguous"},{"power_target":float("nan")}):
            with self.assertRaises(ValueError):
                EBDObjectiveConfig(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
