import numpy as np
import scipy as sp
from pygsp import graphs

from coarsening.utils import get_proximity_measure, \
    matching_optimal, matching_greedy, get_coarsening_matrix, coarsen_matrix, coarsen_vector, zero_diag

from coarsening.coarsening_base import Coarsen


class AffinityGs(Coarsen):

    def __init__(self, setting, data, args):
        super(AffinityGs, self).__init__(setting, data, args)
        args.method = "affinity_GS"
    def coarsen(self, G):
        K = 10
        r = self.args.reduction_rate
        max_levels = 10
        Uk = None
        lk = None
        max_level_r = 0.99,
        r = np.clip(r, 0, 0.999)
        G0 = G
        N = G.N

        # Current and target graph sizes
        n, n_target = N, np.ceil(r * N)

        C = sp.sparse.eye(N, format="csc")
        Gc = G

        Call, Gall = [], []
        Gall.append(G)
        method = "affinity_GS"
        algorithm = self.args.coarsen_strategy  # Default coarsening strategy is 'greedy'

        for level in range(1, max_levels + 1):
            G = Gc

            # How much more we need to reduce the current graph
            r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

            weights = get_proximity_measure(G, method, K=K)

            if algorithm == "optimal":
                # The edge-weight should be light at proximal edges
                weights = -weights
                if "rss" not in method:
                    weights -= min(weights)
                coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

            elif algorithm == "greedy":
                coarsening_list = matching_greedy(G, weights=weights, r=r_cur)

            iC = get_coarsening_matrix(G, coarsening_list)

            if iC.shape[1] - iC.shape[0] <= 2:
                break  # Avoid too many levels for so few nodes

            C = iC.dot(C)
            Call.append(iC)

            Wc = zero_diag(coarsen_matrix(G.W, iC))  # Coarsen and remove self-loops
            Wc = (Wc + Wc.T) / 2  # This is only needed to avoid pygsp complaining for tiny errors

            if not hasattr(G, "coords"):
                Gc = graphs.Graph(Wc)
            else:
                Gc = graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
            Gall.append(Gc)

            n = Gc.N

            if n <= n_target:
                break

        return C, Gc, Call, Gall

