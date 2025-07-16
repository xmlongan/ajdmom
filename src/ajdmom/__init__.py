"""
Package ajdmom

This package is devoted to compute the theoretical moments and covariances of
some Affine Jump Diffusion processes, such as Heston Stochastic Volatility
models and its derived models, such as including jumps in the return process
and so on.
"""
from .poly import Poly
from .cpp_mom import mcpp, cmcpp
from .ito_mom import moment_v, moment_IEII
from .itos_mom import moment_IEI_IEII
# from .ito_cond2_mom import moment_IEII
