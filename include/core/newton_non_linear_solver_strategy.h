// SPDX-FileCopyrightText: Copyright (c) 2019-2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_newton_non_linear_solver_strategy_h
#define lethe_newton_non_linear_solver_strategy_h

#include <core/physics_solver_strategy.h>

// SHARP-TEMPORARY-DEBUG-TOP
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
// SHARP-TEMPORARY-DEBUG-BOTTOM


/**
 * @brief Non-linear solver for non-linear systems of equations which uses a Newton
 * method with \f$\alpha\f$ relaxation to ensure that the residual is
 * monotonically decreasing.
 */
template <typename VectorType>
class NewtonNonLinearSolverStrategy : public PhysicsSolverStrategy<VectorType>
{
public:
  /**
   * @brief Constructor.
   *
   * @param[in] physics_solver A pointer to the physics solver to which the
   * non-linear solver is attached.
   *
   * @param[in] param Non-linear solver parameters as specified in the
   * simulation parameter file.
   *
   */
  NewtonNonLinearSolverStrategy(PhysicsSolver<VectorType> *physics_solver,
                                const Parameters::NonLinearSolver &param);


  /**
   * @brief Solve the non-linear system of equations.
   *
   */
  void
  solve() override;
};

template <typename VectorType>
NewtonNonLinearSolverStrategy<VectorType>::NewtonNonLinearSolverStrategy(
  PhysicsSolver<VectorType>         *physics_solver,
  const Parameters::NonLinearSolver &params)
  : PhysicsSolverStrategy<VectorType>(physics_solver, params)
{}

template <typename VectorType>
void
NewtonNonLinearSolverStrategy<VectorType>::solve()
{
  double global_res;
  double current_res;
  double last_res;
  this->outer_iteration = 0;
  last_res              = 1e6;
  current_res           = 1e6;
  global_res            = 1e6;

  // current_res and global_res are different as one is defined based on the l2
  // norm of the residual vector (current_res) and the other (global_res) is
  // defined by the physical solver and may differ from the l2_norm of the
  // residual vector. Only the global_res is compared to the tolerance in order
  // to evaluate if the nonlinear system is solved. Only current_res is used for
  // the alpha scheme as this scheme only monitors the convergence of the
  // non-linear system of equation (the matrix problem).

  PhysicsSolver<VectorType> *solver = this->physics_solver;

  const double rescale_metric = solver->get_residual_rescale_metric();

  auto &evaluation_point = solver->get_evaluation_point();
  auto &present_solution = solver->get_present_solution();

  while ((global_res > this->params.tolerance) &&
         this->outer_iteration < this->params.max_iterations)
    {
      evaluation_point = present_solution;

      solver->assemble_system_matrix();

      if (!this->params.reuse_preconditioner || this->outer_iteration == 0)
        solver->setup_preconditioner();

      const bool assemble_rhs_this_iteration =
        this->params.force_rhs_calculation || this->outer_iteration == 0;
      if (assemble_rhs_this_iteration)
        solver->assemble_system_rhs();

      // SHARP-TEMPORARY-DEBUG-TOP
      const double debug_fresh_res =
        solver->get_system_rhs().l2_norm() / rescale_metric;
      const auto debug_emit =
        [&](const char *phase, double a, double b, double c) {
          std::ostringstream out;
          out << std::scientific << std::setprecision(12)
              << "[TEMP-NEWTON] iteration=" << this->outer_iteration
              << " phase=" << phase << " a=" << a << " b=" << b << " c=" << c;
          solver->pcout << out.str() << std::endl;
        };
      debug_emit("fresh_rhs_vs_carried_vs_linesearch_reference",
                 debug_fresh_res,
                 current_res,
                 last_res);
      // SHARP-TEMPORARY-DEBUG-BOTTOM
      if (this->outer_iteration == 0)
        {
          auto &system_rhs = solver->get_system_rhs();
          current_res      = system_rhs.l2_norm() / rescale_metric;
          last_res         = current_res;
        }

      if (this->params.verbosity != Parameters::Verbosity::quiet)
        {
          solver->pcout << "Newton iteration: " << this->outer_iteration
                        << "  - Residual:  " << current_res << std::endl;
        }

      if (this->skip_linear_solve_if_fresh_rhs_is_already_converged(
            assemble_rhs_this_iteration,
            rescale_metric,
            current_res,
            last_res,
            global_res))
        continue;

      // SHARP-TEMPORARY-DEBUG-TOP
      const auto debug_linear_start = std::chrono::steady_clock::now();
      debug_emit("linear_solve_begin", debug_fresh_res, 0., 0.);
      // SHARP-TEMPORARY-DEBUG-BOTTOM
      solver->solve_linear_system();
      // SHARP-TEMPORARY-DEBUG-TOP
      debug_emit("linear_solve_seconds",
                 std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - debug_linear_start)
                   .count(),
                 0.,
                 0.);
      // SHARP-TEMPORARY-DEBUG-BOTTOM

      double last_alpha_res = current_res;

      unsigned int alpha_iter = 0;
      for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
        {
          auto &local_evaluation_point = solver->get_local_evaluation_point();
          auto &newton_update          = solver->get_newton_update();
          local_evaluation_point       = present_solution;
          local_evaluation_point.add(alpha, newton_update);
          solver->apply_constraints();
          evaluation_point = local_evaluation_point;
          solver->assemble_system_rhs();

          auto &system_rhs = solver->get_system_rhs();
          current_res      = system_rhs.l2_norm() / rescale_metric;

          if (this->params.verbosity != Parameters::Verbosity::quiet)
            {
              solver->pcout << "\talpha = " << std::defaultfloat << std::setw(6)
                            << alpha << std::setw(6) << std::scientific
                            << "\tres = " << std::setw(6) << current_res;

              solver->output_newton_update_norms();
            }

          // SHARP-TEMPORARY-DEBUG-TOP
          debug_emit("trial_alpha_rhs_ratio_to_fresh",
                     alpha,
                     current_res,
                     current_res / std::max(debug_fresh_res, 1e-300));
          // SHARP-TEMPORARY-DEBUG-BOTTOM
          // If it's not the first iteration of alpha check if the residual is
          // smaller than the last alpha iteration. If it's not smaller, we fall
          // back to the last alpha iteration.
          if (current_res > last_alpha_res and alpha_iter != 0)
            {
              alpha                  = 2 * alpha;
              local_evaluation_point = present_solution;
              local_evaluation_point.add(alpha, newton_update);
              solver->apply_constraints();
              evaluation_point = local_evaluation_point;
              solver->assemble_system_rhs();

              if (this->params.verbosity != Parameters::Verbosity::quiet)
                {
                  solver->pcout << "\t\talpha value was kept at alpha = "
                                << std::defaultfloat << alpha
                                << " since alpha = " << alpha / 2
                                << std::scientific << " increased the residual"
                                << std::endl;
                }
              // SHARP-TEMPORARY-DEBUG-TOP
              debug_emit("restored_rhs_vs_cached",
                         solver->get_system_rhs().l2_norm() / rescale_metric,
                         last_alpha_res,
                         alpha);
              // SHARP-TEMPORARY-DEBUG-BOTTOM
              current_res = last_alpha_res;
              break;
            }
          if (current_res < this->params.step_tolerance * last_res ||
              last_res < this->params.tolerance)
            {
              break;
            }
          last_alpha_res = current_res;
          alpha_iter++;
        }

      // SHARP-TEMPORARY-DEBUG-TOP
      debug_emit("accepted_rhs_vs_fresh_vs_reference",
                 solver->get_system_rhs().l2_norm() / rescale_metric,
                 debug_fresh_res,
                 last_res);
      // SHARP-TEMPORARY-DEBUG-BOTTOM
      global_res       = solver->get_current_residual() / rescale_metric;
      present_solution = evaluation_point;
      last_res         = current_res;
      ++this->outer_iteration;
    }

  // If the non-linear solver has not converged abort simulation if
  // abort_at_convergence_failure=true
  if ((global_res > this->params.tolerance) &&
      this->outer_iteration >= this->params.max_iterations &&
      this->params.abort_at_convergence_failure)
    {
      throw(std::runtime_error(
        "Stopping simulation because the non-linear solver has failed to converge"));
    }
}

#endif
