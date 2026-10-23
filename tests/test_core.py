import os
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_processor as sp
from anomaly import baselines as B
from anomaly import evaluation as E
from anomaly import simulator as S
from anomaly.alerting import AlertEngine
from anomaly.config import SENSORS
from anomaly.model import Autoencoder, AnomalyScorer, make_sizes, save_bundle
from anomaly.storage import LocalStorage

LINE = "CNC-001,CNC_Machine,60.1,12.3,70.2,20.5,1200.5,2"


class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = S.generate_dataframe(20000, 0.15, seed=3)

    def test_fault_mix(self):
        self.assertAlmostEqual(self.df.is_fault.mean(), 0.15, delta=0.015)
        self.assertEqual(set(self.df.fault_type), {"none", "threshold", "decoupling", "mild_offset"})

    def test_rules_see_only_obvious_faults(self):
        flagged = (B.rule_severity(self.df) >= 1)
        by = self.df.assign(f=flagged).groupby("fault_type").f.mean()
        self.assertEqual(by["threshold"], 1.0)
        self.assertEqual(by["decoupling"] + by["mild_offset"] + by["none"], 0.0)

    def test_healthy_values_inside_normal_range(self):
        for mt, r in S.MACHINE_TYPES.items():
            g = self.df[(self.df.machine_type == mt) & (self.df.fault_type != "threshold")]
            for s in SENSORS:
                self.assertGreaterEqual(g[s].min(), r[s][0] - 0.01)
                self.assertLessEqual(g[s].max(), r[s][1] + 0.01)

    def test_error_count_does_not_leak_the_label(self):
        h, f = self.df[self.df.is_fault == 0].error_count.mean(), self.df[self.df.is_fault == 1].error_count.mean()
        self.assertAlmostEqual(h, f, delta=0.1)

    def test_healthy_sensors_are_related(self):
        h = self.df[self.df.fault_type == "none"]
        self.assertGreater(np.corrcoef(h.temperature, h.power_consumption)[0, 1], 0.6)

    def test_csv_roundtrip_and_stream_line_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            S.write_csv(self.df.head(200), p, labels=True)
            back = S.read_csv(p)
            self.assertEqual(len(back), 200)
            self.assertEqual(len(S.to_csv_line(self.df.iloc[0].to_dict(), labels=False).split(",")), 8)
            self.assertEqual(len(S.to_csv_line(self.df.iloc[0].to_dict(), labels=True).split(",")), 10)


class TestParsingAndRules(unittest.TestCase):
    def test_parse_both_formats(self):
        r = sp.parse(LINE)
        self.assertIsNone(r["is_fault"])
        r2 = sp.parse(LINE + ",1,decoupling")
        self.assertEqual((r2["is_fault"], r2["fault_type"]), (1, "decoupling"))

    def test_parse_rejects_bad_lines(self):
        self.assertIsNone(sp.parse("a,b,c"))
        self.assertIsNone(sp.parse(LINE.replace("60.1", "hot")))

    def test_rule_severity(self):
        r = sp.parse(LINE)
        sp.apply_rules(r)
        self.assertEqual((r["severity"], r["is_critical"]), (0, False))
        r["temperature"], r["vibration"] = 80, 25
        sp.apply_rules(r)
        self.assertEqual((r["severity"], r["is_critical"]), (2, True))


class TestMetrics(unittest.TestCase):
    def test_auc_and_ap(self):
        self.assertEqual(E.roc_auc([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(E.roc_auc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)
        self.assertEqual(E.average_precision([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]), 1.0)

    def test_prf_and_detection_by_type(self):
        m = E.prf([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual(E.detection_by_type([1, 0, 1, 1], ["a", "a", "b", "b"]), {"a": 0.5, "b": 1.0})

    def test_split_is_time_ordered_and_disjoint(self):
        df = S.generate_dataframe(1000, 0.15, 1)
        tr, va, te = E.split(df)
        self.assertEqual((len(tr), len(va), len(te)), (700, 150, 150))
        self.assertLess(tr.index.max(), va.index.min())
        self.assertLess(va.index.max(), te.index.min())

    def test_stream_metrics_by_fault_type(self):
        m = sp.blank_metrics()
        base = {"machine_type": "Lathe", "machine_id": "M1", "root_cause": "vibration", "is_critical": False}
        for sev, ae, fault, ft in ((0, False, 0, "none"), (2, True, 1, "threshold"), (0, True, 1, "decoupling"), (0, False, 1, "mild_offset")):
            sp.update_metrics(m, {**base, "severity": sev, "ae_anomaly": ae, "is_fault": fault, "fault_type": ft})
        self.assertEqual(m["ae"], {"tp": 2, "fp": 0, "fn": 1, "tn": 1})
        self.assertEqual(m["rules"], {"tp": 1, "fp": 0, "fn": 2, "tn": 1})
        self.assertEqual(m["by_fault_type"]["decoupling"], {"n": 1, "rules": 0, "ae": 1})

    def test_unlabelled_stream_updates_no_truth_metrics(self):
        m = sp.blank_metrics()
        sp.update_metrics(m, {"machine_type": "L", "machine_id": "M", "root_cause": "sound_level", "severity": 1,
                              "ae_anomaly": True, "is_fault": None, "fault_type": None})
        self.assertEqual((m["total"], m["labelled"], m["ae"]), (1, 0, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}))


class TestStorage(unittest.TestCase):
    def test_local_roundtrip_and_ordering(self):
        with tempfile.TemporaryDirectory() as d:
            s = LocalStorage(d)
            s.write_text("x", "20260101_000001_b.json", "a\nb\n")
            s.write_text("x", "20260101_000002_b.json", "c\nd\n")
            s.write_text("x", "20260101_000002_b.json", "c\nd\ne\n")
            self.assertEqual((s.count_files("x"), s.tail("x", 3), s.read_text("x", "none")), (2, ["c", "d", "e"], ""))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = S.generate_dataframe(20000, 0.15, 5)
        tr, va, _ = E.split(df)
        normal = tr[tr.is_fault == 0]
        cls.norm = E.fit_norm(normal)
        Xtr, Xvh = E.normalise(normal, cls.norm), E.normalise(va[va.is_fault == 0], cls.norm)
        cls.model = Autoencoder(make_sizes(4, 1, 32), seed=1)
        cls.hist = cls.model.fit(Xtr, Xvh, epochs=60)
        cls.Xvh = Xvh
        cls.path = os.path.join(tempfile.mkdtemp(), "b.npz")
        save_bundle(cls.path, cls.model, cls.norm, float(np.percentile(cls.model.errors(Xvh), 99)), {}, (cls.model.forward(Xvh) - Xvh).std(0))
        cls.scorer = AnomalyScorer(cls.path)
        cls.df = df

    def test_training_reduces_loss_and_records_history(self):
        self.assertLess(self.hist["train"][-1], self.hist["train"][0])
        self.assertEqual(len(self.hist["train"]), len(self.hist["val"]))

    def test_encode_has_bottleneck_width(self):
        self.assertEqual(self.model.encode(self.Xvh[:5]).shape, (5, 1))

    def test_healthy_rarely_flagged_decoupling_flagged(self):
        rows = self.df.iloc[17000:]                      # the held-out test slice
        flag = lambda r: self.scorer.score(r.machine_type, [getattr(r, s) for s in SENSORS])[1]
        healthy = [flag(r) for r in rows[rows.fault_type == "none"].head(400).itertuples()]
        deco = [flag(r) for r in rows[rows.fault_type == "decoupling"].head(100).itertuples()]
        self.assertLess(np.mean(healthy), 0.05)
        self.assertGreater(np.mean(deco), 0.8)

    def test_bundle_roundtrip_matches_model(self):
        x = self.Xvh[:20]
        err = float(self.model.errors(x)[0])
        mt = "CNC_Machine"
        raw = x[0] * np.array(self.norm[mt]["std"]) + np.array(self.norm[mt]["mean"])
        self.assertAlmostEqual(self.scorer.score_detail(mt, raw)["err"], err, places=6)


class TestAlerts(unittest.TestCase):
    def test_cooldown_and_levels(self):
        t = [0.0]
        eng = AlertEngine(cooldown_s=10, webhook="", critical_ratio=3, clock=lambda: t[0])
        r = {"ae_anomaly": True, "machine_id": "M1", "machine_type": "Lathe", "root_cause": "sound_level",
             "ae_score": 0.2, "ae_ratio": 1.5, "severity": 0, "processed_at": "x"}
        self.assertEqual(eng.consider(r)["level"], "WARNING")
        self.assertIsNone(eng.consider(r))
        self.assertEqual(eng.suppressed, 1)
        t[0] = 11
        self.assertEqual(eng.consider({**r, "ae_ratio": 4.0})["level"], "CRITICAL")
        t[0] = 30
        self.assertEqual(eng.consider({**r, "severity": 2})["level"], "CRITICAL")
        self.assertIsNone(eng.consider({**r, "ae_anomaly": False, "machine_id": "M2"}))


if __name__ == "__main__":
    unittest.main()
