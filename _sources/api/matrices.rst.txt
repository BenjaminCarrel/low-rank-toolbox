Matrices Module
===============

Low-rank matrix representations.

.. currentmodule:: low_rank_toolbox.matrices

Classes Overview
----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   SVD
   QR
   QuasiSVD
   LowRankMatrix

SVD Class
---------

.. autoclass:: SVD
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::
      :nosignatures:

      ~SVD.from_matrix
      ~SVD.from_quasiSVD
      ~SVD.from_low_rank
      ~SVD.reduced_svd
      ~SVD.truncated_svd
      ~SVD.generate_random
      ~SVD.truncate
      ~SVD.truncate_perpendicular
      ~SVD.norm
      ~SVD.norm_squared
      ~SVD.trace
      ~SVD.diag
      ~SVD.solve
      ~SVD.lstsq
      ~SVD.pseudoinverse
      ~SVD.sqrtm
      ~SVD.expm

QR Class
--------

.. autoclass:: QR
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::
      :nosignatures:

      ~QR.from_matrix
      ~QR.generate_random
      ~QR.truncate
      ~QR.solve
      ~QR.lstsq
      ~QR.pseudoinverse
      ~QR.cond
      ~QR.to_svd
      ~QR.from_svd

QuasiSVD Class
--------------

.. autoclass:: QuasiSVD
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::
      :nosignatures:

      ~QuasiSVD.from_matrix
      ~QuasiSVD.generate_random
      ~QuasiSVD.truncate
      ~QuasiSVD.to_svd
      ~QuasiSVD.to_qr
      ~QuasiSVD.from_qr
      ~QuasiSVD.solve
      ~QuasiSVD.lstsq
      ~QuasiSVD.pseudoinverse
      ~QuasiSVD.sqrtm

LowRankMatrix Class
-------------------

.. autoclass:: LowRankMatrix
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::
      :nosignatures:

      ~LowRankMatrix.from_matrix
      ~LowRankMatrix.from_low_rank
      ~LowRankMatrix.copy
      ~LowRankMatrix.compress
      ~LowRankMatrix.dot
      ~LowRankMatrix.dot_sparse
      ~LowRankMatrix.hadamard
      ~LowRankMatrix.norm
      ~LowRankMatrix.trace
      ~LowRankMatrix.diag
      ~LowRankMatrix.to_dense
      ~LowRankMatrix.to_sparse
      ~LowRankMatrix.save
      ~LowRankMatrix.memory_usage
      ~LowRankMatrix.compression_ratio