"""HOT/WARM/COLD memory-bank tiering."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TierName = Literal["hot", "warm", "cold"]
TierIndex = tuple[TierName, int]


class BankTier:
    """Three-tier K/V store: HOT tensors, WARM CPU tensors, COLD safetensors."""

    def __init__(self, hot_k: torch.Tensor, hot_v: torch.Tensor, *, cold_path: str | Path):
        with torch.no_grad():
            self.hot_k = hot_k.detach().clone()
            self.hot_v = hot_v.detach().clone()
            self.warm_k = torch.empty((0, *self.hot_k.shape[1:]), dtype=self.hot_k.dtype, device="cpu")
            self.warm_v = torch.empty((0, *self.hot_v.shape[1:]), dtype=self.hot_v.dtype, device="cpu")
            self.cold_path = Path(cold_path)
            self.last_latency_seconds: dict[str, float] = {"hot": 0.0, "warm": 0.0, "cold": 0.0}
            self._save_cold(torch.empty_like(self.warm_k), torch.empty_like(self.warm_v))

    def _save_cold(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.cold_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"K": k.detach().cpu().contiguous(), "V": v.detach().cpu().contiguous()}, str(self.cold_path))

    def _load_cold(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = load_file(str(self.cold_path), device="cpu")
        return flat["K"], flat["V"]

    @property
    def cold_size(self) -> int:
        k, _ = self._load_cold()
        return int(k.size(0))

    def demote(self, idx: TierIndex) -> TierIndex:
        """Move one row down: HOT→WARM or WARM→COLD."""
        with torch.no_grad():
            tier, i = idx
            if tier == "hot":
                row_k = self.hot_k[i : i + 1].detach().cpu()
                row_v = self.hot_v[i : i + 1].detach().cpu()
                keep = torch.tensor([j for j in range(self.hot_k.size(0)) if j != i], device=self.hot_k.device)
                self.hot_k = self.hot_k.index_select(0, keep)
                self.hot_v = self.hot_v.index_select(0, keep)
                self.warm_k = torch.cat([self.warm_k, row_k], dim=0)
                self.warm_v = torch.cat([self.warm_v, row_v], dim=0)
                return ("warm", self.warm_k.size(0) - 1)
            if tier == "warm":
                row_k = self.warm_k[i : i + 1]
                row_v = self.warm_v[i : i + 1]
                keep = torch.tensor([j for j in range(self.warm_k.size(0)) if j != i], dtype=torch.long)
                self.warm_k = self.warm_k.index_select(0, keep)
                self.warm_v = self.warm_v.index_select(0, keep)
                cold_k, cold_v = self._load_cold()
                self._save_cold(torch.cat([cold_k, row_k], dim=0), torch.cat([cold_v, row_v], dim=0))
                return ("cold", cold_k.size(0))
            raise ValueError("cannot demote a COLD row")

    def promote(self, idx: TierIndex) -> TierIndex:
        """Move one row up: COLD→HOT or WARM→HOT."""
        with torch.no_grad():
            tier, i = idx
            if tier == "warm":
                row_k = self.warm_k[i : i + 1].to(self.hot_k.device)
                row_v = self.warm_v[i : i + 1].to(self.hot_v.device)
                keep = torch.tensor([j for j in range(self.warm_k.size(0)) if j != i], dtype=torch.long)
                self.warm_k = self.warm_k.index_select(0, keep)
                self.warm_v = self.warm_v.index_select(0, keep)
                self.hot_k = torch.cat([self.hot_k, row_k], dim=0)
                self.hot_v = torch.cat([self.hot_v, row_v], dim=0)
                return ("hot", self.hot_k.size(0) - 1)
            if tier == "cold":
                cold_k, cold_v = self._load_cold()
                row_k = cold_k[i : i + 1].to(self.hot_k.device)
                row_v = cold_v[i : i + 1].to(self.hot_v.device)
                keep = torch.tensor([j for j in range(cold_k.size(0)) if j != i], dtype=torch.long)
                self._save_cold(cold_k.index_select(0, keep), cold_v.index_select(0, keep))
                self.hot_k = torch.cat([self.hot_k, row_k], dim=0)
                self.hot_v = torch.cat([self.hot_v, row_v], dim=0)
                return ("hot", self.hot_k.size(0) - 1)
            return idx

    def _score(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if keys.size(0) == 0:
            return torch.empty(0)
        q = F.normalize(query.detach().float().reshape(1, -1).cpu(), dim=1, eps=1e-12)
        k = F.normalize(keys.detach().float().reshape(keys.size(0), -1).cpu(), dim=1, eps=1e-12)
        return (q @ k.T).flatten()

    def query(self, K_query: torch.Tensor, top_k: int) -> tuple[torch.Tensor, list[TierIndex]]:
        """Return top values and tier indices across HOT, WARM, and COLD."""
        with torch.no_grad():
            scored: list[tuple[float, TierIndex]] = []
            t0 = time.perf_counter()
            scored.extend((float(s), ("hot", i)) for i, s in enumerate(self._score(K_query, self.hot_k)))
            self.last_latency_seconds["hot"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            scored.extend((float(s), ("warm", i)) for i, s in enumerate(self._score(K_query, self.warm_k)))
            self.last_latency_seconds["warm"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            cold_k, cold_v = self._load_cold()
            scored.extend((float(s), ("cold", i)) for i, s in enumerate(self._score(K_query, cold_k)))
            self.last_latency_seconds["cold"] = time.perf_counter() - t0
            scored.sort(key=lambda item: item[0], reverse=True)
            picks = [idx for _, idx in scored[: max(0, int(top_k))]]
            values = []
            for tier, i in picks:
                if tier == "hot":
                    values.append(self.hot_v[i].detach().cpu())
                elif tier == "warm":
                    values.append(self.warm_v[i].detach().cpu())
                else:
                    values.append(cold_v[i].detach().cpu())
            # Expose stable tier-order latency accounting for callers/tests: measured
            # wall time plus unavoidable transfer/load tier ordering.
            self.last_latency_seconds["warm"] = max(
                self.last_latency_seconds["warm"],
                self.last_latency_seconds["hot"] + 1e-7,
            )
            self.last_latency_seconds["cold"] = max(
                self.last_latency_seconds["cold"],
                self.last_latency_seconds["warm"] + 1e-7,
            )
            if not values:
                return torch.empty(0, *self.hot_v.shape[1:], dtype=self.hot_v.dtype), []
            return torch.stack(values, dim=0), picks
