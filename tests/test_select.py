"""Selection by validation: choose the length on a smoothed curve, rewind, anneal to it."""
import torch

from lwd.noise.samplers import DoseMix
from lwd.stage.cf import CFDistance
from lwd.stage.select import SelectConfig, train_selected


class Noise:
    def sample(self, b, L, g, device="cpu"):
        return torch.randn(b, L, 6, generator=g)


def parts():
    torch.manual_seed(0)
    return torch.nn.Linear(6, 6), torch.nn.Linear(6, 6), DoseMix(torch.randn(12, 8, 6), Noise(), 2)


def scripted(values):
    """A validation curve fixed in advance, so the control flow is what gets tested."""
    it = iter(values)
    return lambda student: next(it)


CFG = dict(cap_steps=400, batch=4, seq_len=8, lr=1e-2, warmup=5, val_every=10, patience=3,
           cooldown_frac=0.5, min_cooldown=10, amp=False, log_every=10**9)
#          step:  10   20   30    40    50   60   70   80   90  100
# the run stops after the validation at step 80, so exactly eight values are consumed and
# the ninth is what the annealed model validates at
CURVE = [1.0, 0.8, 0.6, 0.62, 0.61, 1.5, 0.3, 1.6]


def test_the_length_is_the_smoothed_minimum_not_a_lucky_dip():
    """0.3 at step 70 sits between 1.5 and 1.6. The plateau is at step 40."""
    student, teacher, sampler = parts()
    _, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), scripted(CURVE + [0.55]),
                          log=lambda r: None)
    assert s["best_step"] == 40 and abs(s["smoothed_best_val"] - (0.6 + 0.62 + 0.61) / 3) < 1e-9
    assert s["raw_val_at_best_step"] == 0.62
    assert s["stop"] == "patience" and s["stable_steps"] == 80       # 3 finalized after step 40


def test_the_anneal_is_rewound_so_the_run_ends_at_the_chosen_length():
    student, teacher, sampler = parts()
    seen = []

    def val(st):
        seen.append({k: v.clone() for k, v in st.state_dict().items()})
        return (CURVE + [0.55])[len(seen) - 1]

    hist, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), val, log=lambda r: None)
    assert (s["best_step"], s["rewound_to"], s["cooldown_steps"]) == (40, 20, 20)   # 0.5 x 40
    cool = [r for r in hist if r["phase"] == "cooldown"]
    assert [r["step"] for r in cool] == list(range(20, 40))          # it ENDS at the length
    assert cool[-1]["lr"] == 0.0 and cool[0]["lr"] < 1e-2 and cool == sorted(cool, key=lambda r: -r["lr"])
    assert s["selected"] == "annealed" and s["selected_val"] == 0.55 == s["cooled_val"]
    # the weights it annealed from are the ones validated at step 20, not the ones at 40
    before = seen[1]
    student2, teacher2, sampler2 = parts()
    assert any(not torch.equal(before[k], seen[3][k]) for k in before)


def test_an_arm_that_never_stops_improving_is_annealed_at_the_cap():
    student, teacher, sampler = parts()
    cfg = SelectConfig(**{**CFG, "cap_steps": 60})
    _, s = train_selected(student, teacher, sampler, cfg, scripted([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]),
                          log=lambda r: None)
    assert s["stop"] == "cap" and s["best_step"] == 60 and s["rewound_to"] == 30 and s["cooldown_steps"] == 30


def test_an_early_peak_rewinds_to_the_start_and_keeps_its_warmup():
    student, teacher, sampler = parts()
    hist, s = train_selected(student, teacher, sampler, SelectConfig(**CFG),
                             scripted([0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 0.45]), log=lambda r: None)
    assert s["best_step"] == 10 and s["rewound_to"] == 0 and s["cooldown_steps"] == 10
    cool = [r for r in hist if r["phase"] == "cooldown"]
    assert cool[0]["lr"] < cool[3]["lr"]                              # still warming up


def test_snapshots_needed_for_a_late_rewind_are_not_pruned():
    """The best keeps moving later for a long time, then the run stalls."""
    student, teacher, sampler = parts()
    curve = [2.0 - 0.05 * i for i in range(30)] + [5.0] * 6 + [0.4]
    _, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), scripted(curve), log=lambda r: None)
    assert s["best_step"] == 290 and s["rewound_to"] == 150 and s["cooldown_steps"] == 140


def test_the_record_accounts_for_every_position_including_the_discarded_ones():
    student, teacher, sampler = parts()
    _, s = train_selected(student, teacher, sampler, SelectConfig(**CFG), scripted(CURVE + [0.55]),
                          log=lambda r: None)
    steps = s["stable_steps"] + s["cooldown_steps"]
    assert s["positions"] == steps * 4 * 8 == s["real_positions"] + s["noise_positions"]
    assert s["teacher_positions"] == s["positions"] and abs(s["achieved_m"] - 2) < 0.2


def test_the_cf_term_enters_the_loss_only_when_asked():
    student, teacher, sampler = parts()
    hist, _ = train_selected(student, teacher, sampler, SelectConfig(**CFG), scripted(CURVE + [0.55]),
                             log=lambda r: None)
    assert all("cf" not in r for r in hist)
    student, teacher, sampler = parts()
    cfg = SelectConfig(**{**CFG, "cf_lambda": 30.0})
    hist, _ = train_selected(student, teacher, sampler, cfg, scripted(CURVE + [0.55]),
                             cf=CFDistance(6, M=8, seed=0), log=lambda r: None)
    assert all(r["cf"] > 0 and abs(r["loss"] - (r["mse"] + 30.0 * r["cf"])) < 1e-5 for r in hist)
