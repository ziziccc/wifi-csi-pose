from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]]):
    def __init__(
        self,
        npz_path: str | Path,
        window_size: int,
        window_stride: int = 1,
        feature_mode: str = "all",
        require_full_window_mask: bool = False,
        fill_mode: str = "zero",
        max_gap: int = 0,
        return_prev_target: bool = False,
        return_file_id: bool = False,
        motion_lag: int = 1,
    ) -> None:
        super().__init__()
        self.npz_path = Path(npz_path)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.feature_mode = str(feature_mode)
        self.require_full_window_mask = bool(require_full_window_mask)
        self.fill_mode = str(fill_mode)
        self.max_gap = int(max_gap)
        self.return_prev_target = bool(return_prev_target)
        self.return_file_id = bool(return_file_id)
        self.motion_lag = int(motion_lag)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if self.fill_mode not in {"zero", "forward_fill"}:
            raise ValueError(f"Unsupported fill_mode={self.fill_mode!r}")
        if self.max_gap < 0:
            raise ValueError("max_gap must be non-negative.")
        if self.motion_lag < 1:
            raise ValueError("motion_lag must be at least 1.")

        self._load_arrays()
        self.indices = self._build_indices()
        if not self.indices:
            print(f"Warning: no valid windows found in {self.npz_path} for window_size={self.window_size}")

    def _load_arrays(self) -> None:
        data = np.load(self.npz_path)
        self.features = data["features"].astype(np.float32)
        self.labels = data["labels"].astype(np.float32)
        self.file_ids = data["file_ids"].astype(np.int32)
        if self.features.ndim != 3 or self.features.shape[1] % 3 != 0:
            raise ValueError(f"Expected features shaped [N, node_count*3, subcarriers] in {self.npz_path}.")
        self.node_count = int(self.features.shape[1] // 3)

        if len(self.features) != len(self.labels) or len(self.features) != len(self.file_ids):
            raise ValueError(f"Inconsistent cache lengths in {self.npz_path}.")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("features", None)
        state.pop("labels", None)
        state.pop("file_ids", None)
        state.pop("indices", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._load_arrays()
        self.indices = self._build_indices()
        if not self.indices:
            print(f"Warning: no valid windows found in {self.npz_path} for window_size={self.window_size}")

    def _build_indices(self) -> list[int]:
        indices: list[int] = []
        for end_index in range(self.window_size - 1, len(self.features)):
            if (end_index - (self.window_size - 1)) % self.window_stride != 0:
                continue
            start_index = end_index - self.window_size + 1
            if np.all(self.file_ids[start_index : end_index + 1] == self.file_ids[end_index]):
                if self.require_full_window_mask and not self._is_full_window_mask(start_index, end_index):
                    continue
                if self.return_prev_target:
                    previous_index = end_index - self.motion_lag
                    if previous_index < 0 or int(self.file_ids[previous_index]) != int(self.file_ids[end_index]):
                        continue
                indices.append(end_index)
        return indices

    def _is_full_window_mask(self, start_index: int, end_index: int) -> bool:
        mask = self.features[start_index : end_index + 1, self.node_count * 2 :, :]
        return bool(np.all(mask > 0.5))

    def _select_feature_channels(self, window: np.ndarray) -> np.ndarray:
        if self.feature_mode == "all":
            return window
        if self.feature_mode == "base_only":
            return window[:, : self.node_count, :]
        raise ValueError(f"Unsupported feature_mode={self.feature_mode!r}")

    def _prime_fill_state(self, start_index: int) -> tuple[list[np.ndarray | None], list[int]]:
        last_valid: list[np.ndarray | None] = [None] * self.node_count
        gap_lengths = [self.max_gap] * self.node_count
        if self.max_gap <= 0 or start_index <= 0:
            return last_valid, gap_lengths

        file_id = int(self.file_ids[start_index])
        mask_offset = self.node_count * 2
        for node_index in range(self.node_count):
            gap = 0
            cursor = start_index - 1
            while cursor >= 0 and int(self.file_ids[cursor]) == file_id:
                is_present = bool(self.features[cursor, mask_offset + node_index, 0] > 0.5)
                if is_present:
                    if gap < self.max_gap:
                        last_valid[node_index] = self.features[cursor, node_index].copy()
                        gap_lengths[node_index] = gap
                    break
                gap += 1
                if gap >= self.max_gap:
                    break
                cursor -= 1
        return last_valid, gap_lengths

    def _fill_missing_nodes(self, window: np.ndarray, start_index: int) -> np.ndarray:
        if self.fill_mode == "zero" or self.max_gap == 0:
            return window

        filled = window.copy()
        base = filled[:, : self.node_count, :]
        delta = filled[:, self.node_count : self.node_count * 2, :]
        mask = filled[:, self.node_count * 2 :, :]

        last_valid, gap_lengths = self._prime_fill_state(start_index)

        for frame_index in range(filled.shape[0]):
            for node_index in range(self.node_count):
                is_present = bool(mask[frame_index, node_index, 0] > 0.5)
                if is_present:
                    last_valid[node_index] = base[frame_index, node_index].copy()
                    gap_lengths[node_index] = 0
                    continue

                can_fill = last_valid[node_index] is not None and gap_lengths[node_index] < self.max_gap
                if can_fill:
                    base[frame_index, node_index] = last_valid[node_index]
                gap_lengths[node_index] += 1

        delta[0] = 0.0
        if filled.shape[0] > 1:
            delta[1:] = base[1:] - base[:-1]
        return filled

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        end_index = self.indices[index]
        start_index = end_index - self.window_size + 1

        window = self.features[start_index : end_index + 1]
        window = self._fill_missing_nodes(window, start_index=start_index)
        window = self._select_feature_channels(window)
        window = np.transpose(window, (1, 2, 0))
        target = torch.from_numpy(self.labels[end_index])
        if not self.return_prev_target and not self.return_file_id:
            return torch.from_numpy(window), target

        result = {"pose": target}
        if self.return_file_id:
            result["file_id"] = torch.tensor(int(self.file_ids[end_index]), dtype=torch.int32)

        if not self.return_prev_target:
            return torch.from_numpy(window), result

        previous_index = end_index - self.motion_lag
        previous_target = torch.from_numpy(self.labels[previous_index])
        result["prev_pose"] = previous_target
        return torch.from_numpy(window), result
