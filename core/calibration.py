# This module loads and applies optional affine score calibration.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SCORE_DIMENSIONS: Tuple[str, str, str] = (
    "visual_quality",
    "editing_alignment",
    "content_preservation",
)
REWARD_WEIGHTS: Tuple[float, float, float] = (0.3, 0.4, 0.3)


@dataclass(frozen=True)
class AffineDimensionCalibration:
    # This dataclass stores the affine coefficients for one score dimension.
    slope: float
    intercept: float


class AffineScoreCalibrator:
    # This class calibrates raw regression outputs into human-aligned score space.
    def __init__(self, package_dir: Path):
        self.package_dir = Path(package_dir).resolve()
        self.config_path = self.package_dir / "calibration.json"
        self.enabled = self.config_path.exists()
        if not self.enabled:
            self.metadata = {}
            self.coefficients = {
                name: AffineDimensionCalibration(slope=1.0, intercept=0.0) for name in SCORE_DIMENSIONS
            }
            return
        with self.config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("calibration_type") != "affine_per_dimension":
            raise ValueError(f"Unsupported calibration type in {self.config_path}: {payload.get('calibration_type')}")
        coefficients = payload.get("coefficients", {})
        missing = [name for name in SCORE_DIMENSIONS if name not in coefficients]
        if missing:
            raise ValueError(f"Missing calibration coefficients for dimensions: {missing}")
        self.metadata = payload.get("metadata", {})
        self.coefficients: Dict[str, AffineDimensionCalibration] = {
            name: AffineDimensionCalibration(
                slope=float(coefficients[name]["slope"]),
                intercept=float(coefficients[name]["intercept"]),
            )
            for name in SCORE_DIMENSIONS
        }

    # This method clips one scalar into the valid score range.
    def _clip01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    # This method returns the raw score dictionary for one model output triplet.
    def build_raw_score_dict(self, raw_scores: Sequence[float]) -> Dict[str, float]:
        if len(raw_scores) != len(SCORE_DIMENSIONS):
            raise ValueError(f"Expected {len(SCORE_DIMENSIONS)} raw scores, got {len(raw_scores)}")
        payload = {name: float(raw_scores[index]) for index, name in enumerate(SCORE_DIMENSIONS)}
        payload["overall"] = sum(payload[name] * weight for name, weight in zip(SCORE_DIMENSIONS, REWARD_WEIGHTS))
        return payload

    # This method applies affine calibration to the three regression dimensions.
    def calibrate_raw_scores(self, raw_scores: Sequence[float]) -> Dict[str, float]:
        raw_payload = self.build_raw_score_dict(raw_scores)
        calibrated = {}
        for name in SCORE_DIMENSIONS:
            coeff = self.coefficients[name]
            calibrated[name] = self._clip01(raw_payload[name] * coeff.slope + coeff.intercept)
        calibrated["overall"] = sum(
            calibrated[name] * weight for name, weight in zip(SCORE_DIMENSIONS, REWARD_WEIGHTS)
        )
        return calibrated

    # This method returns the calibrated triplet in fixed dimension order.
    def calibrate_triplet(self, raw_scores: Sequence[float]) -> List[float]:
        calibrated = self.calibrate_raw_scores(raw_scores)
        return [calibrated[name] for name in SCORE_DIMENSIONS]

    # This method exposes lightweight calibration metadata for downstream callers.
    def describe(self) -> Dict:
        return {
            "enabled": self.enabled,
            "type": "affine_per_dimension" if self.enabled else "identity",
            "config_path": self.config_path.name if self.enabled else "",
            "coefficients": {
                name: {
                    "slope": self.coefficients[name].slope,
                    "intercept": self.coefficients[name].intercept,
                }
                for name in SCORE_DIMENSIONS
            },
            "metadata": self.metadata,
        }
