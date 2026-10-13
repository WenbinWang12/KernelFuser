"""Shared interface and defaults used by the public E4 pattern definition."""
import torch

DEV, DT = "cuda", torch.bfloat16


class Pattern:
    name = "unnamed"
    backends = ("simb",)
    gate_exempt = ()

    def geometries(self):
        raise NotImplementedError

    def arms(self, geom):
        raise NotImplementedError

    def roofline_ms(self, geom):
        return None

    def label(self, geom):
        return str(geom)
