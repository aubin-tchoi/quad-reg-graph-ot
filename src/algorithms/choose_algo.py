from enum import Enum

from .edge_formulation import j_wasserstein_1
from .kantorovich import t_wasserstein_1
from .quadratic_cvxpy import regularized_j_wasserstein_1
from .regularized_quad_ot import regularized_quadratic_ot
from .sinkhorn import sinkhorn, stable_sinkhorn


# TODO: use this enum instead of the dict below
class Algo(Enum):
    no_reg: j_wasserstein_1
    kanto: t_wasserstein_1
    cvxpy_reg: regularized_j_wasserstein_1
    algo: regularized_quadratic_ot
    sinkhorn: sinkhorn
    stable_sinkhorn: stable_sinkhorn


choose_algo = {
    "no_reg": j_wasserstein_1,
    "kanto": t_wasserstein_1,
    "cvxpy_reg": regularized_j_wasserstein_1,
    "algo": regularized_quadratic_ot,
    "sinkhorn": sinkhorn,
    "stable_sinkhorn": stable_sinkhorn,
}
