from abc import ABC, abstractmethod


class Intervention(ABC):
    """Interface for interventions of the form T(theta, x)."""

    @abstractmethod
    def apply(self, x, theta: float):
        raise NotImplementedError


class Scalarizer(ABC):
    """Interface for scalar response maps g(y)."""

    @abstractmethod
    def __call__(self, raw_output) -> float:
        raise NotImplementedError


class ModelUnit(ABC):
    """Base model interface used by the ISQED query framework."""

    def __init__(self, name: str, scalarizer: Scalarizer = None):
        self.name = name
        self.scalarizer = scalarizer

    @abstractmethod
    def _forward(self, input_data):
        raise NotImplementedError

    def query(self, x, theta, intervention: Intervention) -> float:
        """Evaluate the model on one intervened input."""
        perturbed_x = intervention.apply(x, theta)
        raw_out = self._forward(perturbed_x)
        return self.scalarizer(raw_out) if self.scalarizer else raw_out
