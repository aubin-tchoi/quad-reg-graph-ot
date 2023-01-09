from .choose_algo import choose_algo
from .edge_formulation import j_wasserstein_1
from .kantorovich import t_wasserstein_1
from .quadratic_cvxpy import regularized_j_wasserstein_1
from .regularized_quad_ot import regularized_quadratic_ot
from .sinkhorn import sinkhorn, stable_sinkhorn
from .utils import timeit, return_runtime, checkpoint
