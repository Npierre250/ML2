"""
alerting.py - turns per-record anomaly flags into a manageable stream of alerts.

A raw detector flags ~hundreds of records/minute. Operators need *alerts*:
  * one per machine per cooldown window (no flooding),
  * a severity level,
  * the likely root-cause sensor,
  * optional delivery to a webhook (Slack/Teams/own service) - set ALERT_WEBHOOK.

Severity:  CRITICAL  autoencoder error >= CRITICAL_RATIO x threshold, OR the autoencoder and
                     the fixed rules agree (rule severity >= 2)
           WARNING   autoencoder flag only
"""
import json
import logging
import threading
import time
import urllib.request

from anomaly import config

log = logging.getLogger("alerts")


class AlertEngine:
    def __init__(self, cooldown_s=None, webhook=None, critical_ratio=None, clock=time.time):
        self.cooldown = config.ALERT_COOLDOWN_S if cooldown_s is None else cooldown_s
        self.webhook = config.ALERT_WEBHOOK if webhook is None else webhook
        self.critical_ratio = config.CRITICAL_RATIO if critical_ratio is None else critical_ratio
        self.clock = clock
        self.last = {}          # machine_id -> time of last alert
        self.suppressed = 0

    def consider(self, r):
        """Return an alert dict for record `r`, or None."""
        if not r.get("ae_anomaly"):
            return None
        now = self.clock()
        if now - self.last.get(r["machine_id"], -1e18) < self.cooldown:
            self.suppressed += 1
            return None
        self.last[r["machine_id"]] = now
        critical = r["ae_ratio"] >= self.critical_ratio or r["severity"] >= 2
        alert = {
            "alert_time": r.get("processed_at"), "level": "CRITICAL" if critical else "WARNING",
            "machine_id": r["machine_id"], "machine_type": r["machine_type"],
            "root_cause": r["root_cause"], "ae_score": r["ae_score"], "ae_ratio": round(r["ae_ratio"], 2),
            "rule_severity": r["severity"],
            "message": f"{r['machine_id']}: abnormal {r['root_cause']} pattern "
                       f"({r['ae_ratio']:.1f}x threshold)",
        }
        if self.webhook:
            threading.Thread(target=self._post, args=(alert,), daemon=True).start()
        return alert

    def _post(self, alert):
        try:
            req = urllib.request.Request(self.webhook, json.dumps(alert).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).close()
        except Exception as e:                      # alert delivery must never break the pipeline
            log.warning("webhook failed: %s", e)
