from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

StepFn = Callable[[Dict[str, Any]], Dict[str, Any]]

@dataclass
class Step:
    name: str
    fn: StepFn
    description: str = ""

class Pipeline:
    def __init__(self):
        self.steps: Dict[str, Step] = {}

    def register(self, name: str, description: str = ""):
        def deco(fn: StepFn) -> StepFn:
            if name in self.steps:
                raise ValueError(f"Step '{name}' already registered")
            self.steps[name] = Step(name, fn, description)
            return fn
        return deco

    def run(self, order: List[str], ctx: Dict[str, Any]) -> Dict[str, Any]:
        for name in order:
            if name not in self.steps:
                raise KeyError(f"Unknown step: {name}. Known: {list(self.steps)}")
            t0 = time.time()
            ctx = self.steps[name].fn(ctx)
            ctx.setdefault("_log", []).append({"step": name, "seconds": time.time() - t0})
        return ctx
