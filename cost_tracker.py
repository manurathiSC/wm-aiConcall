# -*- coding: utf-8 -*-
"""
Tracks cost (and token usage) for every LLM call so you can see total cost per step and for the full run.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_community.callbacks import get_openai_callback


@dataclass
class CallRecord:
    """Single LLM call record."""
    step_name: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    raw_callback: Optional[Any] = None

    @property
    def cost_display(self) -> str:
        if self.cost_usd > 0:
            return f"${self.cost_usd:.6f}"
        return f"{self.total_tokens} tokens (no pricing)"

    def __repr__(self) -> str:
        return (
            f"CallRecord(step={self.step_name!r}, model={self.model!r}, "
            f"tokens={self.total_tokens}, cost={self.cost_display})"
        )


class CostTracker:
    """
    Tracks all LLM call costs. Use as context manager around OpenAI calls, or call
    add_record() after each LLM invoke for non-OpenAI providers.
    """

    # Approximate USD per 1K tokens (for display when callback not available)
    OPENAI_DEFAULT_RATES = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    }

    def __init__(self, run_name: str = "default"):
        self.run_name = run_name
        self._records: List[CallRecord] = []
        self._current_callback = None

    def _guess_cost_openai(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Rough cost estimate from token counts (per 1M tokens input/output)."""
        model_lower = model.lower()
        rate_in, rate_out = 0.15, 0.60  # gpt-4o-mini default
        for key, (ri, ro) in self.OPENAI_DEFAULT_RATES.items():
            if key in model_lower:
                rate_in, rate_out = ri, ro
                break
        return (prompt_tokens * rate_in / 1_000_000) + (completion_tokens * rate_out / 1_000_000)

    def add_record(
        self,
        step_name: str,
        provider: str = "openai",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: Optional[float] = None,
        raw_callback: Optional[Any] = None,
    ) -> CallRecord:
        total = prompt_tokens + completion_tokens
        if cost_usd is None and provider == "openai" and total > 0:
            cost_usd = self._guess_cost_openai(model, prompt_tokens, completion_tokens)
        elif cost_usd is None:
            cost_usd = 0.0
        rec = CallRecord(
            step_name=step_name,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost_usd,
            raw_callback=raw_callback,
        )
        self._records.append(rec)
        return rec

    def add_from_openai_callback(self, step_name: str, callback: Any, model: str = "openai") -> CallRecord:
        """Add a record from LangChain's get_openai_callback() result."""
        return self.add_record(
            step_name=step_name,
            provider="openai",
            model=model,
            prompt_tokens=getattr(callback, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(callback, "completion_tokens", 0) or 0,
            cost_usd=getattr(callback, "total_cost", None) or 0.0,
            raw_callback=callback,
        )

    @property
    def records(self) -> List[CallRecord]:
        return list(self._records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    def summary_by_step(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate by step_name: total tokens and cost."""
        by_step: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            if r.step_name not in by_step:
                by_step[r.step_name] = {"tokens": 0, "cost_usd": 0.0, "calls": 0}
            by_step[r.step_name]["tokens"] += r.total_tokens
            by_step[r.step_name]["cost_usd"] += r.cost_usd
            by_step[r.step_name]["calls"] += 1
        return by_step

    def print_summary(self) -> None:
        print(f"\n--- Cost summary (run: {self.run_name}) ---")
        print(f"Total tokens: {self.total_tokens}")
        print(f"Total cost:   ${self.total_cost_usd:.6f} USD")
        print("By step:")
        for step, agg in self.summary_by_step().items():
            print(f"  {step}: {agg['calls']} calls, {agg['tokens']} tokens, ${agg['cost_usd']:.6f}")
        print("---\n")

    def get_summary_for_output(self) -> Dict[str, Any]:
        """Return a dict suitable for JSON/file output."""
        return {
            "run_name": self.run_name,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "by_step": self.summary_by_step(),
            "records": [
                {
                    "step": r.step_name,
                    "model": r.model,
                    "tokens": r.total_tokens,
                    "cost_usd": round(r.cost_usd, 8),
                }
                for r in self._records
            ],
        }
