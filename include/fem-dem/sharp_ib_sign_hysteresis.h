// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_sharp_ib_sign_hysteresis_h
#define lethe_sharp_ib_sign_hysteresis_h

#include <deal.II/base/point.h>

#include <cstddef>
#include <map>
#include <utility>

/**
 * @brief Inside/outside classification of mesh support points with respect
 * to Sharp-IB particles, with a small hysteresis band around the surface.
 *
 * WHY THIS EXISTS
 *
 * The Sharp-IB solver decides, for every support point near a particle,
 * whether the point is inside the particle (level set phi <= 0) or outside.
 * That single bit selects the discrete equations: an inside pressure DoF of a
 * cut cell gets the interior equation p_i = p_ref, an outside one keeps the
 * fluid equation, neighbouring interior rows choose their reference pressure
 * among the outside nodes, and cells are classified as cut or inside from the
 * same bit. The particle position is itself an unknown of the coupled
 * fluid/particle Newton iteration, so the discrete system is a
 * discontinuous function of the particle position: when the surface passes
 * a node, the fluid force on the particle jumps.
 *
 * If the jump pushes the particle back across the node, the coupled problem
 * has no discrete solution: with the node classified outside, the converged
 * particle position puts it inside, and vice versa. Newton then alternates
 * between the two classifications forever. This was observed in practice: a
 * 146-particle case with h = 5 um and dt = 4e-7 s failed its first time step
 * because one node flipped at every Newton iteration from the fifth onwards,
 * the surface oscillating by about 0.03 nm (6e-6 h) around it. Such
 * a trap only needs the converged surface to land within that tiny gap of
 * any node, so with many particles it happens regularly, while small
 * cases rarely see it.
 *
 * WHAT THE BAND DOES
 *
 * A point with |phi| > band is classified by its sign, exactly as before.
 * A point with |phi| <= band keeps the classification it had at the previous
 * evaluation (the previous Newton iteration or time step). A classification
 * therefore only changes once the surface has moved a full band past the
 * point, like a thermostat that switches on and off at different
 * temperatures. As long as the band is wider than the gap described above,
 * the held classification has a self-consistent solution within the band and
 * Newton converges to it.
 *
 * The particle geometry, the level set, the stencils and their interpolation
 * weights are NOT modified. The only consequence is that a node lying up to
 * one band on the "wrong" side of the surface may be treated as if it were on
 * the other side, which is a geometric perturbation of order band / h. With
 * the default band of 1e-2 h this is far below the discretization error of
 * the stencils themselves, while being about three orders of magnitude wider
 * than the gaps measured at dt = 4e-7 s.
 *
 * Choosing the band: it must exceed the gap, which scales roughly with
 * dt^2 * (force jump) / (particle mass); it must remain much smaller than h
 * so that the geometric perturbation stays negligible. The band is chosen by
 * the caller through reinit(); FluidDynamicsSharp uses a fixed fraction of
 * the cell size (see FluidDynamicsSharp::sign_hysteresis_band_fraction, which
 * also records the open question of exposing it as a parameter).
 *
 * MEMORY AND CONSISTENCY
 *
 * Only points within two bands of a surface are remembered, and a memory
 * entry is only trusted if it was written during the current or the previous
 * evaluation pass (see start_new_pass()). A point that enters the band with
 * no recent memory is classified by its sign. Memory is keyed by the particle
 * index and the exact support point coordinates, so all components sharing a
 * support point, and all callers within one pass, see the same answer.
 * Evaluations of a periodic image far from the point (|phi| > 2 band) never
 * touch the memory. Each MPI rank keeps the memory of the points it
 * evaluates; since every rank evaluates its locally relevant points with the
 * same particle positions in every pass, ghost points are classified
 * identically on all ranks. The memory is not written to checkpoints: after a
 * restart, points start from their plain sign, which only differs from an
 * uninterrupted run if a node lies within the band at the restart time.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class SharpIBSignHysteresis
{
public:
  /**
   * @brief Set the band width and clear the memory, e.g. after a mesh change.
   *
   * @param band Half-width of the hysteresis band, in length units. A value
   * of 0 makes is_inside() equivalent to phi <= 0.
   */
  void
  reinit(const double band);

  /**
   * @brief Start a new evaluation pass (one per Newton iteration).
   *
   * Memory written during the previous pass remains usable in the new pass;
   * older memory is discarded. Call this once before the cut-cell mapping of
   * each Newton iteration.
   */
  void
  start_new_pass();

  /**
   * @brief Classify a support point as inside or outside a particle.
   *
   * @param particle_id Index of the particle.
   * @param point Support point, in the unshifted mesh frame.
   * @param levelset Signed distance of the point to the (possibly
   * periodically shifted) particle surface; negative inside.
   *
   * @return True if the point is treated as inside the particle.
   */
  bool
  is_inside(const unsigned int        particle_id,
            const dealii::Point<dim> &point,
            const double              levelset);

  /**
   * @brief Return the band half-width set by reinit().
   */
  double
  get_band() const
  {
    return band;
  }

  /**
   * @brief Return the number of remembered points (for diagnostics).
   */
  std::size_t
  n_remembered_points() const
  {
    return memory.size();
  }

private:
  /// Order support points lexicographically so they can key a std::map.
  struct PointLess
  {
    bool
    operator()(const std::pair<unsigned int, dealii::Point<dim>> &a,
               const std::pair<unsigned int, dealii::Point<dim>> &b) const;
  };

  /// Remembered classification and the pass in which it was written.
  struct Entry
  {
    bool          inside;
    unsigned long pass;
  };

  double        band = 0.;
  unsigned long pass = 0;
  std::map<std::pair<unsigned int, dealii::Point<dim>>, Entry, PointLess>
    memory;
};

#endif
