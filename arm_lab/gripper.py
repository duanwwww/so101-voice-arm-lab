"""Simulated jaw feedback. Units and limits below apply to the course MuJoCo model only."""
from dataclasses import dataclass
import math


class FeedbackStop(Exception):
    """Motion is latched off until the operator resets the simulation."""


@dataclass(frozen=True)
class JawFeedback:
    time: float
    position: float
    fingers: int
    on_table: bool
    alarm: bool = False

    def valid_at(self, now):
        return (math.isfinite(self.position) and math.isfinite(self.time)
                and 0 <= now - self.time <= 0.035 and self.fingers in (0, 1, 2))


CONTROL_PERIOD = 0.010
CLOSE_RATE = 0.30             # rad/s
CLOSE_FOLLOW_LIMIT = 0.065   # rad, approach tracking envelope
CLOSE_TORQUE_LIMIT = 0.80    # N m, approach cap held fixed in force comparison
HOLD_TORQUE = 0.30           # N m, lowest tested successful four-layout hold
HOLD_SETTLE_SECONDS = 0.38
TEST_LIFT = 0.006            # m
PICK_RADIUS_MM = 8.0
MAX_ALIGNMENT_MM = 6.0
# The calibrated camera's centre readings on the three taught marks differ
# from their known XY by under 2 mm. Ignore displacements at that scale.
ALIGNMENT_DEADBAND_MM = 2.0
