"""Explicit, account/model-scoped live activation controlled by the operator."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class LiveAuthorization:
    login: int
    server: str
    model_hashes: tuple
    risk_hash: str
    issued_at: float
    expires_at: float
    acknowledged_login: int

    @classmethod
    def activate(cls, login, server, model_hashes, risk, acknowledged_login, hours=8):
        if type(login) is not int or login <= 0 or login != acknowledged_login or not server:
            raise ValueError("Enter the exact connected real account number to activate live trading")
        if not model_hashes or any(len(x) != 64 for x in model_hashes) or not 0 < hours <= 168:
            raise ValueError("A frozen model and a bounded activation period are required")
        now = time.time()
        return cls(login, server, tuple(model_hashes), fingerprint(risk), now, now + hours * 3600, acknowledged_login)

    def validate(self, login, server, model_hash, risk, reducing_risk=False, now=None):
        now = time.time() if now is None else now
        if (self.login != login or self.acknowledged_login != login or self.server != server
                or model_hash not in self.model_hashes or self.risk_hash != fingerprint(risk)):
            raise ValueError("Live activation does not match this account, model or risk configuration")
        if not math.isfinite(self.expires_at) or now < self.issued_at - 5:
            raise ValueError("Invalid live activation clock")
        if not reducing_risk and now >= self.expires_at:
            raise ValueError("Live activation expired; existing positions can still be closed")

    def public_status(self):
        return {"login": self.login, "server": self.server, "expires_at": self.expires_at,
                "active": time.time() < self.expires_at}
