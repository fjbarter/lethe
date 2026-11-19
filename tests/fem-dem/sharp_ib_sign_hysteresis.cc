// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief Check the inside/outside classification of SharpIBSignHysteresis.
 *
 * Each scenario prints the level set (in units of the band) given at every
 * evaluation pass and the resulting classification. The key scenario is the
 * one that made the Sharp-IB Newton iteration fail: a surface oscillating by
 * a fraction of the band around a node must not change the node's side.
 */

#include <fem-dem/sharp_ib_sign_hysteresis.h>

#include <deal.II/base/point.h>

#include <../tests/tests.h>

#include <string>
#include <vector>

using namespace dealii;

namespace
{
  /**
   * @brief Evaluate one point once per pass for a sequence of level sets and
   * log the classification.
   *
   * @param hysteresis Classifier under test.
   * @param title Scenario description.
   * @param particle_id Particle index passed to the classifier.
   * @param point Support point passed to the classifier.
   * @param levelsets_in_bands Level set at each pass, in units of the band.
   */
  template <int dim>
  void
  run_sequence(SharpIBSignHysteresis<dim> &hysteresis,
               const std::string          &title,
               const unsigned int          particle_id,
               const Point<dim>           &point,
               const std::vector<double>  &levelsets_in_bands)
  {
    deallog << title << std::endl;
    const double band = hysteresis.get_band();
    for (const double levelset_in_bands : levelsets_in_bands)
      {
        hysteresis.start_new_pass();
        const bool inside =
          hysteresis.is_inside(particle_id, point, levelset_in_bands * band);
        deallog << "  phi/band = " << levelset_in_bands << " -> "
                << (inside ? "inside" : "outside") << std::endl;
      }
  }

  /**
   * @brief Run all scenarios in one dimension.
   */
  template <int dim>
  void
  test()
  {
    deallog << "dim = " << dim << std::endl;

    const double               band = 1e-3;
    SharpIBSignHysteresis<dim> hysteresis;
    Point<dim>                 node;
    for (unsigned int d = 0; d < dim; ++d)
      node[d] = 0.25 * (d + 1);

    // 1. The failure mode: the surface oscillates around the node by less
    // than the band. The node keeps its initial side (outside).
    hysteresis.reinit(band);
    run_sequence(hysteresis,
                 "Oscillation around the node, starting outside",
                 0,
                 node,
                 {0.5, -0.2, 0.3, -0.2, 0.3, -0.2});

    // 2. The surface genuinely passes the node: the side changes once the
    // node is more than one band inside, and is then held on the way back.
    hysteresis.reinit(band);
    run_sequence(hysteresis,
                 "Surface passing the node",
                 0,
                 node,
                 {0.5, -0.5, -1.5, -0.5, 0.5, 1.5});

    // 3. A point first seen inside the band, or seen again after a pass in
    // which it was not evaluated, is classified by the sign of the level set.
    hysteresis.reinit(band);
    run_sequence(
      hysteresis, "First evaluation inside the band", 0, node, {-0.5});
    hysteresis.start_new_pass(); // The point is not evaluated in this pass.
    run_sequence(hysteresis, "Evaluation after a skipped pass", 0, node, {0.5});

    // 4. Several evaluations within one pass agree, and an evaluation of a
    // distant periodic image (|phi| > 2 band) does not alter the memory.
    hysteresis.reinit(band);
    hysteresis.start_new_pass();
    hysteresis.is_inside(0, node, 0.5 * band);
    hysteresis.start_new_pass();
    const bool first  = hysteresis.is_inside(0, node, -0.5 * band);
    const bool image  = hysteresis.is_inside(0, node, 1000. * band);
    const bool second = hysteresis.is_inside(0, node, -0.5 * band);
    deallog << "Same pass: " << (first ? "inside" : "outside") << ", distant "
            << "image: " << (image ? "inside" : "outside")
            << ", again: " << (second ? "inside" : "outside") << std::endl;

    // 5. Memory is kept per particle.
    hysteresis.reinit(band);
    hysteresis.start_new_pass();
    hysteresis.is_inside(0, node, 0.5 * band);
    hysteresis.is_inside(1, node, -0.5 * band);
    hysteresis.start_new_pass();
    deallog << "Particle 0 at phi/band = -0.5: "
            << (hysteresis.is_inside(0, node, -0.5 * band) ? "inside" :
                                                             "outside")
            << ", particle 1 at phi/band = 0.5: "
            << (hysteresis.is_inside(1, node, 0.5 * band) ? "inside" :
                                                            "outside")
            << std::endl;

    // 6. A zero band is the plain sign test, with phi = 0 counted inside.
    hysteresis.reinit(0.);
    deallog << "Zero band" << std::endl;
    for (const double levelset : {1e-12, 0., 1e-12, -1e-12})
      {
        hysteresis.start_new_pass();
        deallog << "  phi = " << levelset << " -> "
                << (hysteresis.is_inside(0, node, levelset) ? "inside" :
                                                              "outside")
                << std::endl;
      }
    deallog << "Remembered points with a zero band: "
            << hysteresis.n_remembered_points() << std::endl;
  }
} // namespace

int
main()
{
  initlog();
  test<2>();
  test<3>();
}
