# Quadratically-Regularized Optimal Transport on Graphs

This repo contains resources to solve the problem of computing the Wasserstein-1 distance on a graph with a quadratic
regularization.

More precisely, two probability distributions are defined on the nodes of a directed graph with weighted edges, and we
are looking at the optimization problem that consists in finding the transport map of minimal cost that sends the first
distribution onto the second one.

The algorithm implemented here in `regularized_quad_ot` is presented in the following
article: https://arxiv.org/pdf/1704.08200.pdf.

For comparison, other algorithms are implemented,
including [Sinkhorn algorithm](https://papers.nips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf) and
direct implementations relying on a solver.

The motivation behind the use of a quadratic regularization is detailed in the article, alongside the idea behind the
algorithm in itself.

There could be some tricks or details of implementation I did not replicate properly, the authors of the article worked
with MATLAB, which does not exactly offer the same content in terms of available libraries, particularly when it comes
to the implementation of Cholesky rank 1 updates. 
