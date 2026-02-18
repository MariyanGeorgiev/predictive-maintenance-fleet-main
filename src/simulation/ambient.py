"""Seasonal and daily ambient temperature model."""

import math

from src.config.constants import (
    AMBIENT_TEMP_DAILY_AMP,
    AMBIENT_TEMP_MEAN,
    AMBIENT_TEMP_SEASONAL_AMP,
)


class AmbientTemperatureModel:
    """Models ambient temperature with seasonal and daily sinusoidal cycles.

    Seasonal: period=365 days, peak at ~day 90 (summer in Northern Hemisphere).
    Daily: period=24 hours, peak at ~14:00 (2 PM).
    """

    def __init__(
        self,
        mean: float = AMBIENT_TEMP_MEAN,
        seasonal_amp: float = AMBIENT_TEMP_SEASONAL_AMP,
        daily_amp: float = AMBIENT_TEMP_DAILY_AMP,
    ):
        self.mean = mean
        self.seasonal_amp = seasonal_amp
        self.daily_amp = daily_amp

    def get_temperature(self, day_index: int, second_of_day: int) -> float:
        """Get ambient temperature at a specific time.

        Args:
            day_index: Day number (0-182 for 6 months).
            second_of_day: Second within the day (0-86399).

        Returns:
            Ambient temperature in °C.
        """
        # Seasonal: peaks around day 90 (about 3 months into simulation)
        seasonal = self.seasonal_amp * math.sin(2 * math.pi * (day_index - 90) / 365.0)

        # Daily: peaks at 14:00 (50400 seconds)
        hour_fraction = second_of_day / 86400.0
        daily = self.daily_amp * math.sin(2 * math.pi * (hour_fraction - 14.0 / 24.0))

        return self.mean + seasonal + daily
