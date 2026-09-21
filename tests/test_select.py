"""Selection by validation: stop on patience, cool the best checkpoint, keep the better."""
import torch

from lwd.noise.samplers import DoseMix
from lwd.stage.cf import CFDistance
from lwd.stage.select import SelectConfig, train_selected


class Noise:
    def sample(self, b, L, g, device="cpu"):
        return torch.randn(b, L, 6, generator=g)


def parts():
    torch.manual_seed(0)
    teacher = torch.nn.Linear(6, 6)
    student = torch.nn.Linear(6, 6)
    return student, teacher, DoseMix(torch.randn(12, 8, 6), Noise(), 2)


def scripted(values):
    """A validation curve fixed in advance, so the control flow is what gets tested."""
    it = iter(values)
    return lambda student: next(it)


CFG = dict(cap_steps=200, batch=4, seq_len=8, lr=1e-2, warmup=5, val_every=10, patience=2,
           cooldown_frac=0.5, min_cooldown=3, amp=False, log_every=10**9)


def test_stops_on_patience_and_selects_the_cooled_model_when_it_is_better():
    student, teacher, sampler = parts()
    #                     best ^            ^ two without a new best, then the cooled value
    val = scripted([1.0, 0.8, 0.6, 0.7, 0.9, 0.5])
    hist, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), val, log=lambda r: None)
    assert s["stop"] == "patience" and s["best_step"] == 30 and s["best_val"] == 0.6
    assert s["stable_steps"] == 50 and s["cooldown_steps"] == 15       # 0.5 x best_step
    assert s["selected"] == "cooled" and s["selected_val"] == 0.5
    assert s["validations"] == 6


def test_hands_back_the_snapshot_when_cooling_makes_it_worse():
    student, teacher, sampler = parts()
    seen = {}

    def val(st):
        seen[len(seen)] = {k: v.clone() for k, v in st.state_dict().items()}
        return [1.0, 0.6, 0.7, 0.9, 0.95][len(seen) - 1]

    _, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), val, log=lambda r: None)
    assert s["selected"] == "stable-best" and s["selected_val"] == 0.6 and s["best_step"] == 20
    for k, v in student.state_dict().items():                          # exactly the weights
        assert torch.equal(v, seen[1][k])                              # validated at the best


def test_rate_is_held_then_cooled_linearly_to_zero():
    student, teacher, sampler = parts()
    hist, s = train_selected(student, teacher, sampler, SelectConfig(**CFG),
                             scripted([1.0, 0.6, 0.7, 0.9, 0.5]), log=lambda r: None)
    stable = [r["lr"] for r in hist if r["phase"] == "stable"]
    cool = [r["lr"] for r in hist if r["phase"] == "cooldown"]
    assert stable[0] < stable[4] == 1e-2 and all(x == 1e-2 for x in stable[4:])
    assert cool == sorted(cool, reverse=True) and cool[-1] == 0.0 and cool[0] < 1e-2


def test_runs_to_the_cap_when_validation_keeps_improving():
    student, teacher, sampler = parts()
    cfg = SelectConfig(**{**CFG, "cap_steps": 40})
    _, s = train_selected(student, teacher, sampler, cfg, scripted([0.9, 0.8, 0.7, 0.6, 0.5]),
                          log=lambda r: None)
    assert s["stop"] == "cap" and s["best_step"] == 40


def test_the_record_accounts_for_every_position_and_the_dose():
    student, teacher, sampler = parts()
    hist, s = train_selected(student, teacher, sampler, SelectConfig(**CFG),
                             scripted([1.0, 0.6, 0.7, 0.9, 0.5]), log=lambda r: None)
    steps = s["stable_steps"] + s["cooldown_steps"]
    assert s["positions"] == steps * 4 * 8 == s["real_positions"] + s["noise_positions"]
    assert s["teacher_positions"] == s["positions"]
    assert abs(s["achieved_m"] - 2) < 0.2                              # not a whole cycle


def test_the_cf_term_enters_the_loss_only_when_asked():
    student, teacher, sampler = parts()
    hist, _ = train_selected(student, teacher, sampler, SelectConfig(**CFG),
                             scripted([1.0, 0.6, 0.7, 0.9, 0.5]), log=lambda r: None)
    assert all("cf" not in r for r in hist)
    student, teacher, sampler = parts()
    cfg = SelectConfig(**{**CFG, "cf_lambda": 30.0})
    hist, _ = train_selected(student, teacher, sampler, cfg, scripted([1.0, 0.6, 0.7, 0.9, 0.5]),
                             cf=CFDistance(6, M=8, seed=0), log=lambda r: None)
    assert all(r["cf"] > 0 and abs(r["loss"] - (r["mse"] + 30.0 * r["cf"])) < 1e-5 for r in hist)
