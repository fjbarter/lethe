// SPDX-FileCopyrightText: Copyright (c) 2020-2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#include <core/bdf.h>
#include <core/grids.h>
#include <core/lethe_grid_tools.h>
#include <core/solutions_output.h>
#include <core/tensors_and_points_dimension_manipulation.h>
#include <core/time_integration_utilities.h>
#include <core/utilities.h>

#include <solvers/isothermal_compressible_navier_stokes_cls_assembler.h>
#include <solvers/navier_stokes_cls_assemblers.h>
#include <solvers/postprocessing_cfd.h>

#include <dem/force_chains_visualization.h>
#include <dem/set_particle_particle_contact_force_model.h>
#include <dem/update_local_particle_containers.h>
#include <dem/visualization.h>
#include <fem-dem/fluid_dynamics_sharp.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <boost/signals2/connection.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>

// SHARP-TEMPORARY-DEBUG-TOP
#include <core/shape.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <sstream>
// SHARP-TEMPORARY-DEBUG-BOTTOM

// SHARP-TEMPORARY-DEBUG-TOP
// Temporary diagnostics: all vector operations use separate owned work vectors.
// No extra residual assembly, particle integration, or solver update is
// performed.
template <int dim>
void
FluidDynamicsSharp<dim>::debug_phase(
  const std::string                    &phase,
  std::chrono::steady_clock::time_point start)
{
  const double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
  const double lo   = Utilities::MPI::min(seconds, this->mpi_communicator);
  const double hi   = Utilities::MPI::max(seconds, this->mpi_communicator);
  const double mean = Utilities::MPI::sum(seconds, this->mpi_communicator) /
                      Utilities::MPI::n_mpi_processes(this->mpi_communicator);
  std::ostringstream out;
  out << std::scientific << std::setprecision(8) << "[SHARP-TEMP] step="
      << this->simulation_control->get_iteration_number()
      << " newton=" << this->get_current_newton_iteration()
      << " phase=" << phase << " seconds_min_mean_max=" << lo << ',' << mean
      << ',' << hi;
  this->pcout << out.str() << std::endl;
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_capture_matrix()
{
  const auto start = std::chrono::steady_clock::now();
  if (!debug_components_ready)
    {
      std::vector<types::global_dof_index> indices(this->fe->n_dofs_per_cell());
      for (const auto &cell : this->dof_handler->active_cell_iterators())
        if (!cell->is_artificial())
          {
            cell->get_dof_indices(indices);
            for (unsigned int j = 0; j < indices.size(); ++j)
              if (this->locally_owned_dofs.is_element(indices[j]))
                debug_components[indices[j]] =
                  this->fe->system_to_component_index(j).first;
          }
      debug_components_ready = true;
    }
  debug_base_solution.reinit(this->locally_owned_dofs, this->mpi_communicator);
  debug_base_rhs.reinit(this->locally_owned_dofs, this->mpi_communicator);
  debug_prediction.reinit(this->locally_owned_dofs, this->mpi_communicator);
  debug_base_solution = this->evaluation_point;
  debug_base_rhs      = this->system_rhs;
  debug_base_valid    = true;

  // Matrix health: zero diagonals alone are not proof of singularity in a
  // saddle-point problem; zero rows and nonfinite entries deserve attention.
  std::vector<double> counts(4, 0.);
  double              largest_entry = 0., largest_row = 0., smallest_ratio = 1.;
  for (const auto row : this->locally_owned_dofs)
    {
      double row_sum = 0.;
      for (auto entry = this->system_matrix.begin(row);
           entry != this->system_matrix.end(row);
           ++entry)
        {
          const double value = entry->value();
          if (!std::isfinite(value))
            ++counts[0];
          else
            {
              row_sum += std::abs(value);
              largest_entry = std::max(largest_entry, std::abs(value));
            }
        }
      const double diagonal = std::abs(this->system_matrix.el(row, row));
      if (row_sum == 0.)
        ++counts[1];
      if (diagonal == 0.)
        ++counts[2];
      if (this->zero_constraints.is_constrained(row))
        ++counts[3];
      if (row_sum > 0.)
        smallest_ratio = std::min(smallest_ratio, diagonal / row_sum);
      largest_row = std::max(largest_row, row_sum);
    }
  Utilities::MPI::sum(counts, this->mpi_communicator, counts);
  largest_entry  = Utilities::MPI::max(largest_entry, this->mpi_communicator);
  largest_row    = Utilities::MPI::max(largest_row, this->mpi_communicator);
  smallest_ratio = Utilities::MPI::min(smallest_ratio, this->mpi_communicator);
  this->pcout << "[SHARP-TEMP] matrix nonfinite_entries=" << counts[0]
              << " zero_rows=" << counts[1] << " zero_diagonals=" << counts[2]
              << " constrained_rows=" << counts[3]
              << " max_abs_entry=" << largest_entry
              << " max_row_l1=" << largest_row
              << " min_abs_diag_over_row_l1=" << smallest_ratio << std::endl;
  // Inspect the two sign conventions without changing any matrix entry.
  std::vector<double> constraint_check(4, 0.);
  for (const auto &[row, kind] : debug_row_kind)
    if (kind == 5 && this->locally_owned_dofs.is_element(row))
      {
        const auto *entries =
          this->nonzero_constraints.get_constraint_entries(row);
        if (entries == nullptr)
          entries = this->zero_constraints.get_constraint_entries(row);
        if (entries == nullptr)
          continue;
        const double diagonal = this->system_matrix.el(row, row);
        ++constraint_check[0];
        for (const auto &[master, weight] : *entries)
          {
            const double coefficient = this->system_matrix.el(row, master);
            constraint_check[1] += std::pow(coefficient + weight * diagonal, 2);
            constraint_check[2] += std::pow(coefficient - weight * diagonal, 2);
          }
      }
  // Matrix columns may extend beyond the locally relevant constraint set on
  // an adaptively refined mesh. Import ownership-resolved flags instead of
  // querying AffineConstraints for an index it does not store.
  IndexSet debug_columns(this->dof_handler->n_dofs());
  debug_columns.add_indices(this->locally_owned_dofs);
  if (std::getenv("SHARP_DEBUG_DENSE_RELEVANT_GRAPH") != nullptr)
    debug_columns.add_range(0, this->dof_handler->n_dofs());
  else
    for (const auto row : this->locally_owned_dofs)
      for (auto entry = this->system_matrix.begin(row);
           entry != this->system_matrix.end(row);
           ++entry)
        debug_columns.add_index(entry->column());
  debug_columns.compress();
  GlobalVectorType debug_owned_flags(this->locally_owned_dofs,
                                     this->mpi_communicator);
  for (const auto row : this->locally_owned_dofs)
    {
      const auto *entries = this->zero_constraints.get_constraint_entries(row);
      debug_owned_flags(row) =
        this->zero_constraints.is_constrained(row) ?
          (entries != nullptr && !entries->empty() ? 2. : 1.) :
          0.;
    }
  debug_owned_flags.compress(VectorOperation::insert);
  GlobalVectorType debug_column_flags;
  debug_column_flags.reinit(this->locally_owned_dofs,
                            debug_columns,
                            this->mpi_communicator);
  debug_column_flags = debug_owned_flags;
  unsigned int debug_printed_constrained_columns = 0;
  for (const auto row : this->locally_owned_dofs)
    if (!this->zero_constraints.is_constrained(row))
      for (auto entry = this->system_matrix.begin(row);
           entry != this->system_matrix.end(row);
           ++entry)
        if (debug_column_flags(entry->column()) > 0.5)
          {
            constraint_check[3] += entry->value() * entry->value();
            if (std::getenv("SHARP_DEBUG_CONSTRAINED_COLUMN_DETAILS") != nullptr &&
                std::abs(entry->value()) > 1e-8 &&
                debug_printed_constrained_columns++ < 12)
              {
                const auto column = entry->column();
                const bool relevant = this->locally_relevant_dofs.is_element(column);
                std::cerr << "[SHARP-TEMP] constrained_column rank="
                          << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                          << " row=" << row << " row_kind="
                          << (debug_row_kind.count(row) ? debug_row_kind.at(row) : 0)
                          << " column=" << column << " value=" << entry->value()
                          << " owner_constraint_kind=" << debug_column_flags(column)
                          << " locally_relevant=" << relevant
                          << " local_zero_constraint="
                          << (relevant && this->zero_constraints.is_constrained(column))
                          << " local_nonzero_constraint="
                          << (relevant && this->nonzero_constraints.is_constrained(column))
                          << std::endl;
              }
          }
  Utilities::MPI::sum(constraint_check,
                      this->mpi_communicator,
                      constraint_check);
  this->pcout << "[SHARP-TEMP] sharp_affine_rows=" << constraint_check[0]
              << " coefficient_error_for_negative_master_sign="
              << std::sqrt(constraint_check[1])
              << " coefficient_error_for_positive_master_sign="
              << std::sqrt(constraint_check[2])
              << " independent_row_coupling_to_constrained_columns_l2="
              << std::sqrt(constraint_check[3]) << std::endl;
  debug_residual("after_matrix_bulk_rhs_not_refreshed");
  debug_phase("debug_matrix_audit_overhead", start);
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_force_feedback()
{
  const auto start = std::chrono::steady_clock::now();
  // Geometry remains fixed throughout this probe. Save only the force outputs
  // that force_on_ib writes; production kinematics are never changed here.
  const auto capture = [&]() {
    std::vector<std::array<Tensor<1, 3>, 4>> values;
    for (const auto &particle : particles)
      values.push_back({{particle.fluid_forces,
                         particle.fluid_pressure_forces,
                         particle.fluid_viscous_forces,
                         particle.fluid_torque}});
    return values;
  };
  const auto       original = capture();
  GlobalVectorType saved_solution;
  saved_solution.reinit(this->locally_owned_dofs, this->mpi_communicator);
  saved_solution     = this->evaluation_point;
  const auto restore = [&]() {
    this->evaluation_point = saved_solution;
    for (unsigned int p = 0; p < particles.size(); ++p)
      {
        particles[p].fluid_forces          = original[p][0];
        particles[p].fluid_pressure_forces = original[p][1];
        particles[p].fluid_viscous_forces  = original[p][2];
        particles[p].fluid_torque          = original[p][3];
      }
  };
  try
    {
      // A repeated evaluation detects dependence on warmed geometry caches.
      force_on_ib();
      const auto repeated = capture();
      if (debug_force_baseline_valid &&
          debug_previous_force_solution.size() == saved_solution.size() &&
          debug_previous_force_values.size() == particles.size())
        {
          this->evaluation_point = debug_previous_force_solution;
          force_on_ib();
          const auto frozen = capture();
          for (unsigned int p = 0; p < particles.size(); ++p)
            {
              std::ostringstream out;
              out << std::scientific << std::setprecision(12)
                  << "[SHARP-TEMP] force_feedback newton="
                  << this->get_current_newton_iteration() << " particle=" << p
                  << " pressure_force=" << original[p][1]
                  << " viscous_force=" << original[p][2]
                  << " fluid_field_delta=" << original[p][0] - frozen[p][0]
                  << " geometry_stencil_delta="
                  << frozen[p][0] - debug_previous_force_values[p][0]
                  << " pressure_field_delta=" << original[p][1] - frozen[p][1]
                  << " pressure_geometry_delta="
                  << frozen[p][1] - debug_previous_force_values[p][1]
                  << " torque_field_delta=" << original[p][3] - frozen[p][3]
                  << " torque_geometry_delta="
                  << frozen[p][3] - debug_previous_force_values[p][3]
                  << " force_repeat_error="
                  << (repeated[p][0] - original[p][0]).norm()
                  << " torque_repeat_error="
                  << (repeated[p][3] - original[p][3]).norm();
              this->pcout << out.str() << std::endl;
            }
        }
      else
        this->pcout << "[SHARP-TEMP] force_feedback baseline newton="
                    << this->get_current_newton_iteration() << std::endl;
    }
  catch (...)
    {
      restore();
      throw;
    }
  restore();
  debug_previous_force_solution.reinit(this->locally_owned_dofs,
                                       this->mpi_communicator);
  debug_previous_force_solution = saved_solution;
  debug_previous_force_values   = original;
  debug_force_baseline_valid    = true;
  debug_phase("force_feedback_probe", start);
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_prepare_rhs()
{
  ++debug_rhs_calls;
  if (!debug_base_valid)
    return;
  GlobalVectorType delta;
  delta.reinit(this->locally_owned_dofs, this->mpi_communicator);
  delta = this->evaluation_point;
  delta.add(-1., debug_base_solution);
  debug_trial_norm = delta.l2_norm();
  // Compare outlet activation at the matrix state and the current trial. This
  // evaluates only boundary faces and never modifies either solution vector.
  GlobalVectorType debug_ghost_base;
  debug_ghost_base.reinit(this->locally_owned_dofs,
                          this->locally_relevant_dofs,
                          this->mpi_communicator);
  debug_ghost_base = debug_base_solution;
  FEFaceValues<dim>                debug_face(*this->mapping,
                               *this->fe,
                               *this->face_quadrature,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
  const FEValuesExtractors::Vector debug_velocity(0);
  std::vector<Tensor<1, dim>>      base_u(this->face_quadrature->size());
  std::vector<Tensor<1, dim>>      trial_u(this->face_quadrature->size());
  double outlet_area = 0., activated_area = 0., deactivated_area = 0.;
  double stable_remainder = 0., switched_remainder = 0.;
  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          {
            const auto  id = cell->face(face)->boundary_id();
            const auto &types =
              this->simulation_parameters.boundary_conditions.type;
            const auto type = types.find(id);
            if (type == types.end() ||
                type->second != BoundaryConditions::BoundaryType::outlet)
              continue;
            debug_face.reinit(cell, face);
            debug_face[debug_velocity].get_function_values(debug_ghost_base,
                                                           base_u);
            debug_face[debug_velocity].get_function_values(
              this->evaluation_point, trial_u);
            for (unsigned int q = 0; q < base_u.size(); ++q)
              {
                const auto   n = debug_face.normal_vector(q);
                const double a = base_u[q] * n, b = trial_u[q] * n;
                const auto   du        = trial_u[q] - base_u[q];
                auto         predicted = std::min(a, 0.) * base_u[q];
                if (a < 0.)
                  predicted += a * du + (du * n) * base_u[q];
                const double remainder =
                  (std::min(b, 0.) * trial_u[q] - predicted).norm();
                const double area = debug_face.JxW(q);
                outlet_area += area;
                if (a >= 0. && b < 0.)
                  activated_area += area;
                if (a < 0. && b >= 0.)
                  deactivated_area += area;
                if ((a < 0.) != (b < 0.))
                  switched_remainder += area * remainder;
                else
                  stable_remainder += area * remainder;
              }
          }
  this->pcout << "[SHARP-TEMP] outlet_trial rhs_call=" << debug_rhs_calls
              << " area="
              << Utilities::MPI::sum(outlet_area, this->mpi_communicator)
              << " activated_area="
              << Utilities::MPI::sum(activated_area, this->mpi_communicator)
              << " deactivated_area="
              << Utilities::MPI::sum(deactivated_area, this->mpi_communicator)
              << " stable_flux_remainder_integral="
              << Utilities::MPI::sum(stable_remainder, this->mpi_communicator)
              << " switched_flux_remainder_integral="
              << Utilities::MPI::sum(switched_remainder, this->mpi_communicator)
              << std::endl;

  // The matrix acts on a pressure-scaled Newton update. Convert the physical
  // trial displacement back to that convention before testing its prediction.
  const double pressure_scale =
    this->simulation_parameters.stabilization.pressure_scaling_factor;
  for (const auto &[index, component] : debug_components)
    if (component == dim)
      delta(index) /= pressure_scale;
  delta.compress(VectorOperation::insert);
  this->system_matrix.vmult(debug_prediction, delta);
  debug_prediction *= -1.;
  debug_prediction.add(1., debug_base_rhs);
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_finish_rhs()
{
  if (!debug_base_valid)
    return;
  // Matrix assembly does not refresh the bulk RHS. Establish the reference
  // from the first full RHS assembly at the unchanged base solution, otherwise
  // a stale bulk residual would masquerade as a Jacobian inconsistency.
  if (debug_rhs_calls == 1 && debug_trial_norm == 0.)
    {
      debug_base_rhs = this->system_rhs;
      this->pcout << "[SHARP-TEMP] captured_fresh_rhs_baseline" << std::endl;
      for (unsigned int component = 0;
           component < static_cast<unsigned int>(dim + 1);
           ++component)
        {
          double                  peak = 0.;
          types::global_dof_index row  = numbers::invalid_dof_index;
          for (const auto index : this->locally_owned_dofs)
            if (debug_components.at(index) == component &&
                std::abs(this->system_rhs(index)) > peak)
              {
                peak = std::abs(this->system_rhs(index));
                row  = index;
              }
          const double global_peak =
            Utilities::MPI::max(peak, this->mpi_communicator);
          const auto global_row =
            Utilities::MPI::min(peak == global_peak ?
                                  row :
                                  numbers::invalid_dof_index,
                                this->mpi_communicator);
          std::string message;
          if (global_row != numbers::invalid_dof_index &&
              this->locally_owned_dofs.is_element(global_row))
            {
              std::ostringstream out;
              out << std::scientific << std::setprecision(12)
                  << "[SHARP-TEMP] fresh_peak_rhs component=" << component
                  << " row=" << global_row
                  << " raw_value=" << this->system_rhs(global_row) << " kind="
                  << (debug_row_kind.count(global_row) ?
                        debug_row_kind.at(global_row) :
                        0)
                  << " physical_value=" << this->evaluation_point(global_row);
              if (debug_pressure_reference.count(global_row))
                out << " pressure_reference="
                    << debug_pressure_reference.at(global_row)
                    << " reference_value="
                    << this->evaluation_point(
                         debug_pressure_reference.at(global_row));
              message = out.str();
            }
          const auto messages =
            Utilities::MPI::gather(this->mpi_communicator, message);
          for (const auto &entry : messages)
            if (!entry.empty())
              this->pcout << entry << std::endl;
        }
      return;
    }
  std::vector<double>     defect(9, 0.), change(9, 0.), bad(9, 0.);
  double                  peak    = 0.;
  types::global_dof_index peak_id = numbers::invalid_dof_index;
  for (const auto index : this->locally_owned_dofs)
    {
      const auto         it   = debug_row_kind.find(index);
      const unsigned int kind = it == debug_row_kind.end() ? 0 : it->second;
      const double       d = this->system_rhs(index) - debug_prediction(index);
      const double       c = this->system_rhs(index) - debug_base_rhs(index);
      if (!std::isfinite(d) || !std::isfinite(c))
        {
          ++bad[kind];
          continue;
        }
      defect[kind] += d * d;
      change[kind] += c * c;
      if (!this->zero_constraints.is_constrained(index) &&
          !this->nonzero_constraints.is_constrained(index) &&
          std::abs(d) > peak)
        {
          peak    = std::abs(d);
          peak_id = index;
        }
    }
  Utilities::MPI::sum(defect, this->mpi_communicator, defect);
  Utilities::MPI::sum(change, this->mpi_communicator, change);
  Utilities::MPI::sum(bad, this->mpi_communicator, bad);
  const double global_peak = Utilities::MPI::max(peak, this->mpi_communicator);
  const auto   global_id =
    Utilities::MPI::min(peak == global_peak ? peak_id :
                                              numbers::invalid_dof_index,
                        this->mpi_communicator);
  std::ostringstream out;
  out
    << std::scientific << std::setprecision(10)
    << "[SHARP-TEMP] rhs_trial=" << debug_rhs_calls
    << " physical_delta_l2=" << debug_trial_norm
    << " max_independent_linearization_defect=" << global_peak
    << " dof=" << global_id
    << " row_classes=bulk,touched,interpolation,fallback,dummy_pressure,sharp_affine,sharp_dirichlet,condensed_affine,condensed_dirichlet";
  for (unsigned int k = 0; k < 9; ++k)
    out << " class" << k << "_defect_change_nonfinite=" << std::sqrt(defect[k])
        << ',' << std::sqrt(change[k]) << ',' << bad[k];
  this->pcout << out.str() << std::endl;
  std::string detail;
  if (global_id != numbers::invalid_dof_index &&
      this->locally_owned_dofs.is_element(global_id))
    {
      std::ostringstream row;
      row << std::scientific << std::setprecision(12)
          << "[SHARP-TEMP] peak_independent_defect_row=" << global_id
          << " component=" << debug_components.at(global_id) << " kind="
          << (debug_row_kind.count(global_id) ? debug_row_kind.at(global_id) :
                                                0)
          << " actual_rhs=" << this->system_rhs(global_id)
          << " base_rhs=" << debug_base_rhs(global_id)
          << " predicted_rhs=" << debug_prediction(global_id)
          << " diagonal=" << this->system_matrix.el(global_id, global_id);
      const auto *entries =
        this->zero_constraints.get_constraint_entries(global_id);
      if (entries != nullptr)
        for (const auto &[master, weight] : *entries)
          row << " master=" << master << " weight=" << weight
              << " matrix_coefficient="
              << this->system_matrix.el(global_id, master);
      std::vector<types::global_dof_index> indices(this->fe->n_dofs_per_cell());
      bool                                 found = false;
      for (const auto &cell : this->dof_handler->active_cell_iterators())
        {
          if (cell->is_artificial())
            continue;
          cell->get_dof_indices(indices);
          for (unsigned int j = 0; j < indices.size(); ++j)
            if (indices[j] == global_id)
              {
                row << " position="
                    << this->get_mapping()->transform_unit_to_real_cell(
                         cell, this->fe->get_unit_support_points()[j])
                    << " cell=" << cell->id().to_string();
                found = true;
                break;
              }
          if (found)
            break;
        }
      detail = row.str();
    }
  const auto details = Utilities::MPI::gather(this->mpi_communicator, detail);
  for (const auto &entry : details)
    if (!entry.empty())
      this->pcout << entry << std::endl;
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_residual(const std::string &phase)
{
  if (!debug_components_ready)
    return;
  std::vector<double> square(dim + 1, 0.), peak(dim + 1, 0.), bad(dim + 1, 0.);
  std::vector<double> row_square(9, 0.);
  for (const auto &[index, component] : debug_components)
    {
      const double value = this->system_rhs(index);
      if (!std::isfinite(value))
        {
          ++bad[component];
          continue;
        }
      square[component] += value * value;
      peak[component] = std::max(peak[component], std::abs(value));
      const auto it   = debug_row_kind.find(index);
      row_square[it == debug_row_kind.end() ? 0 : it->second] += value * value;
    }
  Utilities::MPI::sum(square, this->mpi_communicator, square);
  Utilities::MPI::max(peak, this->mpi_communicator, peak);
  Utilities::MPI::sum(bad, this->mpi_communicator, bad);
  Utilities::MPI::sum(row_square, this->mpi_communicator, row_square);
  std::ostringstream out;
  out << std::scientific << std::setprecision(10)
      << "[SHARP-TEMP] residual_phase=" << phase
      << " rhs_call=" << debug_rhs_calls
      << " scale=" << this->get_residual_rescale_metric();
  for (unsigned int c = 0; c < square.size(); ++c)
    out << " component" << c << "_raw_l2_max_nonfinite=" << std::sqrt(square[c])
        << ',' << peak[c] << ',' << bad[c];
  out
    << " raw_row_l2_bulk_touched_interp_fallback_dummy_sharp_affine_sharp_dirichlet_condensed_affine_condensed_dirichlet=";
  for (const double value : row_square)
    out << std::sqrt(value) << ',';
  this->pcout << out.str() << std::endl;
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_stencils()
{
  std::vector<double> values(debug_stencil_counts.begin(),
                             debug_stencil_counts.end());
  auto                total = values;
  Utilities::MPI::sum(values, this->mpi_communicator, total);
  auto maximum = values;
  Utilities::MPI::max(values, this->mpi_communicator, maximum);
  this->pcout
    << "[SHARP-TEMP] stencils attempted_wrapped_wall_fallback_pressure_overconstrained=";
  for (unsigned int i = 0; i < 6; ++i)
    this->pcout << total[i] << ',';
  this->pcout << " max_reference_cell_extrapolation=" << maximum[6]
              << " max_abs_stencil_weight=" << maximum[7] << std::endl;
}

template <int dim>
void
FluidDynamicsSharp<dim>::debug_particles(const std::string &phase, bool save)
{
  const auto count     = static_cast<unsigned long long>(particles.size());
  const auto min_count = Utilities::MPI::min(count, this->mpi_communicator);
  const auto max_count = Utilities::MPI::max(count, this->mpi_communicator);
  if (min_count != max_count)
    {
      this->pcout << "[SHARP-TEMP] ERROR particle counts differ across ranks: "
                  << min_count << '/' << max_count << std::endl;
      return;
    }
  std::vector<double> state(particles.size() * 9, 0.);
  unsigned int        nonfinite = 0;
  double              max_move = 0., max_dv = 0., max_domega = 0.;
  unsigned int        worst = 0;
  const auto period = sharp_ib_periodic_boundaries.get_periodic_offset();
  for (unsigned int p = 0; p < particles.size(); ++p)
    {
      double move2 = 0., dv2 = 0., dw2 = 0.;
      for (int d = 0; d < 3; ++d)
        {
          state[9 * p + d]     = d < dim ? particles[p].position[d] : 0.;
          state[9 * p + 3 + d] = particles[p].velocity[d];
          state[9 * p + 6 + d] = particles[p].omega[d];
          if (!save && debug_particle_baseline.size() == state.size())
            {
              double dx = state[9 * p + d] - debug_particle_baseline[9 * p + d];
              if (d < dim && period[d] > 0.)
                dx -= std::round(dx / period[d]) * period[d];
              move2 += dx * dx;
              dv2 += std::pow(state[9 * p + 3 + d] -
                                debug_particle_baseline[9 * p + 3 + d],
                              2);
              dw2 += std::pow(state[9 * p + 6 + d] -
                                debug_particle_baseline[9 * p + 6 + d],
                              2);
            }
        }
      if (std::sqrt(dv2) > max_dv)
        {
          max_dv = std::sqrt(dv2);
          worst  = p;
        }
      max_move   = std::max(max_move, std::sqrt(move2));
      max_domega = std::max(max_domega, std::sqrt(dw2));
    }
  // Replace nonfinite values only in the diagnostic reduction buffer, never in
  // the particles. MPI extrema with NaNs are not a reliable consistency check.
  for (double &value : state)
    if (!std::isfinite(value))
      {
        ++nonfinite;
        value = 0.;
      }
  auto lo = state;
  Utilities::MPI::min(state, this->mpi_communicator, lo);
  auto hi = state;
  Utilities::MPI::max(state, this->mpi_communicator, hi);
  double disagreement = 0.;
  for (unsigned int i = 0; i < state.size(); ++i)
    disagreement = std::max(disagreement, hi[i] - lo[i]);
  nonfinite = Utilities::MPI::sum(nonfinite, this->mpi_communicator);
  std::ostringstream out;
  out << std::scientific << std::setprecision(10)
      << "[SHARP-TEMP] particles phase=" << phase
      << " count=" << particles.size() << " rank_disagreement=" << disagreement
      << " nonfinite_rank_sum=" << nonfinite
      << " root_max_move_dv_domega=" << max_move << ',' << max_dv << ','
      << max_domega;
  if (!particles.empty())
    {
      const auto  &p  = particles[worst];
      const double dt = this->simulation_control->get_time_step();
      out << " root_worst_dv_particle=" << worst << " mass=" << p.mass
          << " volume=" << p.volume << " density=" << p.mass / p.volume
          << " inertia_diag=" << p.inertia[0][0] << ',' << p.inertia[1][1]
          << ',' << p.inertia[2][2] << " E=" << p.youngs_modulus
          << " fluid_force_norm=" << p.fluid_forces.norm()
          << " contact_plus_lubrication_average_force_norm="
          << p.contact_impulsion.norm() / dt
          << " fluid_torque_norm=" << p.fluid_torque.norm()
          << " contact_plus_lubrication_average_torque_norm="
          << p.omega_contact_impulsion.norm() / dt;
      if (!save && debug_velocity_inverse.size() == particles.size())
        out << " inverse_velocity_relaxation=" << debug_velocity_inverse[worst];
      if (!save && debug_omega_inverse.size() == particles.size())
        out << " inverse_omega_relaxation=" << debug_omega_inverse[worst];
      out << " DEM_substeps="
          << this->simulation_parameters.particlesParameters->coupling_frequency
          << " dt=" << dt << " particle_alpha="
          << this->simulation_parameters.particlesParameters->alpha;
    }
  const bool debug_only_spheres =
    std::all_of(particles.begin(), particles.end(), [](const auto &particle) {
      return dynamic_cast<const Sphere<dim> *>(particle.shape.get()) != nullptr;
    });
  if (debug_only_spheres && particles.size() > 1)
    {
      double       gap      = std::numeric_limits<double>::max();
      unsigned int overlaps = 0, first = 0, second = 0;
      for (unsigned int i = 0; i < particles.size(); ++i)
        for (unsigned int j = 0; j < i; ++j)
          {
            auto delta = particles[i].position - particles[j].position;
            for (int d = 0; d < dim; ++d)
              if (period[d] > 0.)
                delta[d] -= std::round(delta[d] / period[d]) * period[d];
            const double distance =
              delta.norm() - particles[i].radius - particles[j].radius;
            if (distance < 0.)
              ++overlaps;
            if (distance < gap)
              {
                gap    = distance;
                first  = j;
                second = i;
              }
          }
      out << " root_sphere_pair_min_gap=" << gap << " pair=" << first << ','
          << second << " overlaps=" << overlaps;
    }
  this->pcout << out.str() << std::endl;
  if (save)
    debug_particle_baseline = state;
}
// SHARP-TEMPORARY-DEBUG-BOTTOM

// Constructor for class FluidDynamicsSharp
template <int dim>
FluidDynamicsSharp<dim>::FluidDynamicsSharp(
  CFDDEMSimulationParameters<dim> &p_nsparam)
  : FluidDynamicsMatrixBased<dim>(p_nsparam.cfd_parameters)
  , cfd_dem_parameters(p_nsparam)
  , all_spheres(true)
  , combined_shapes()
{}

template <int dim>
FluidDynamicsSharp<dim>::~FluidDynamicsSharp() = default;


template <int dim>
void
FluidDynamicsSharp<dim>::vertices_cell_mapping()
{
  // Find all the cells around each vertex
  TimerOutput::Scope t(this->computing_timer, "Map vertices to cell");

  LetheGridTools::vertices_cell_mapping(*this->dof_handler, vertices_to_cell);
  // Bound the exact-query cache to one timestep and refresh mesh iterators.
  if (sharp_ib_periodic_boundaries.is_periodic_enabled())
    periodic_stencil_search.reinit(*this->dof_handler);
}

template <int dim>
void
FluidDynamicsSharp<dim>::check_whether_all_particles_are_sphere()
{
  all_spheres = false;

  // TODO The optimized cut-cell mapping seems to lead to instability
  /*
  for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
    {
      if (particles[p_i].shape->get_shape_name().second !=
          Shape<dim>::ShapeType::sphere)
        {
          all_spheres = false;
          std::cout
            << "A non-spherical particle was found: using regular
  cut_cells_mapping."
            << std::endl;
          break;
        }
    }
    */
}

template <int dim>
void
FluidDynamicsSharp<dim>::generate_cut_cells_map()
{
  // Potential amelioration of this function. This function could try to use the
  // information of the previous cut cell mapping as an initial guess of the new
  // cell mapping by looking at the parent level of the cells to look for new
  // cells cut.



  // check all the cells if they are cut or not. Put the information in a map
  // with the key being the cell.
  TimerOutput::Scope t(this->computing_timer, "Map cut cells");


  // A vector of the unordered map. Each map stores if a point is inside or
  // outside a given particle.
  std::vector<std::unordered_map<types::global_dof_index, bool>>
    inside_outside_support_point_vector(particles.size());
  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(*this->mapping, *this->dof_handler);

  // When the finite element degree > 1, overconstrained cells are impossible
  // since there is always at least a DOF inside the element that is not
  // overconstrained. We therefore only have to check when the velocity degree
  // == 1.
  const bool mapping_overconstrained_cells =
    this->simulation_parameters.fem_parameters.velocity_degree == 1;

  cut_cells_map.clear();
  cells_inside_map.clear();
  overconstrained_fluid_cell_map.clear();
  dof_with_more_then_one_particle.clear();
  if (mapping_overconstrained_cells)
    {
      local_dof_overconstrained.reinit(this->locally_owned_dofs,
                                       this->mpi_communicator);
      dof_overconstrained.reinit(this->locally_owned_dofs,
                                 this->locally_relevant_dofs,
                                 this->mpi_communicator);
    }

  const auto        &cell_iterator = this->dof_handler->active_cell_iterators();
  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int dofs_per_face = this->fe->dofs_per_face;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_face_dof_indices(dofs_per_face);

  auto              &v_x_fe                  = this->fe->get_sub_fe(0, 1);
  const unsigned int dofs_per_cell_local_v_x = v_x_fe.dofs_per_cell;


  // // Loop on all the cells and check if they are cut.
  for (const auto &cell : cell_iterator)
    {
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          bool         cell_is_cut                                = false;
          bool         cell_is_inside                             = false;
          unsigned int particle_id_which_cuts_this_cell           = 0;
          unsigned int particle_id_in_which_this_cell_is_embedded = 0;
          std::vector<ParticlePeriodicFrame> cutting_particle_frames;
          // is the particle index that cuts the cell if it's cut. If the cell
          // is not cut the default value is stored (0). If the cell is not cut
          // this value will never be used.
          // cutting_particle_frames records the particles
          // that cut a cell if multiple particles cut this cell. This is used
          // to treat cells that are cut by multiple particles differently.
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int p = 0; p < particles.size(); ++p)
            {
              const Point<dim> saved_particle_position = particles[p].position;
              const std::vector<Tensor<1, dim>> candidate_periodic_shifts =
                get_periodic_particle_shifts(saved_particle_position,
                                             particle_support_radius(
                                               particles[p]));

              // Particles defined from a stl or an iges don't have a signed
              // distance function. To identify cells that are cut a special
              // algorithm is required. For this type of particle, no cell are
              // identified as being inside the particle. This means that the
              // fluid is always solved inside the particle.
              if (particles[p].shape->additional_info_on_shape == "iges")
                {
                  bool           particle_cuts_cell = false;
                  Tensor<1, dim> best_periodic_shift;

                  // Unsigned CAD distances are evaluated in each image frame.
                  for (const auto &periodic_shift : candidate_periodic_shifts)
                    {
                      set_particle_position_from_periodic_shift(
                        p, saved_particle_position, periodic_shift);
                      if (cell_cut_by_p_absolute_distance(cell,
                                                          support_points,
                                                          p))
                        {
                          AssertThrow(
                            !particle_cuts_cell,
                            ExcMessage(
                              "Refine the mesh: a cell intersects two periodic images of one Sharp-IB particle."));
                          particle_cuts_cell  = true;
                          best_periodic_shift = periodic_shift;
                        }
                    }

                  if (particles[p].position != saved_particle_position)
                    particles[p].set_position(saved_particle_position);

                  if (particle_cuts_cell)
                    {
                      if (cutting_particle_frames.empty())
                        {
                          cell_is_cut                      = true;
                          particle_id_which_cuts_this_cell = p;
                        }
                      if (std::ranges::none_of(
                            cutting_particle_frames,
                            [p](const auto &particle_frame) {
                              return particle_frame.particle_id == p;
                            }))
                        {
                          cutting_particle_frames.push_back(
                            {p, best_periodic_shift});
                        }
                    }
                }
              else
                {
                  unsigned int   best_nb_dof_inside = 0;
                  Tensor<1, dim> best_periodic_shift;

                  for (const auto &periodic_shift : candidate_periodic_shifts)
                    {
                      unsigned int nb_dof_inside = 0;
                      set_particle_position_from_periodic_shift(
                        p, saved_particle_position, periodic_shift);

                      for (unsigned int j = 0; j < local_dof_indices.size();
                           ++j)
                        {
                          if (0 == this->fe->system_to_component_index(j).first)
                            {
                              if (particles[p].get_levelset(
                                    support_points[local_dof_indices[j]],
                                    cell) <= 0)
                                {
                                  ++nb_dof_inside;
                                  if (periodic_shift.norm_square() == 0.0)
                                    inside_outside_support_point_vector
                                      [p][local_dof_indices[j]] = true;
                                }
                            }
                        }

                      AssertThrow(
                        nb_dof_inside == 0 || best_nb_dof_inside == 0,
                        ExcMessage(
                          "Refine the mesh: a cell intersects two periodic images of one Sharp-IB particle."));
                      if (nb_dof_inside > best_nb_dof_inside)
                        {
                          best_nb_dof_inside  = nb_dof_inside;
                          best_periodic_shift = periodic_shift;
                        }
                    }

                  if (particles[p].position != saved_particle_position)
                    particles[p].set_position(saved_particle_position);

                  // If some DOFs are inside the boundary, the cell is inside
                  // the particle or cut by the particle.
                  if (best_nb_dof_inside != 0)
                    {
                      // If all the DOFs are inside the boundary this cell is
                      // inside the particle. Otherwise, the particle is cut.
                      if (best_nb_dof_inside == dofs_per_cell_local_v_x)
                        {
                          //  We register only the particle with the lowest id
                          //  as the particle in which this cell is embedded if
                          //  the cell is in multiple particles.
                          if (cutting_particle_frames.empty())
                            {
                              cell_is_cut                      = false;
                              particle_id_which_cuts_this_cell = 0;
                              cell_is_inside                   = true;
                              particle_id_in_which_this_cell_is_embedded = p;
                            }
                          break;
                        }
                      else
                        {
                          // We only register the particle with the lowest id as
                          // the particle by this cell is cut if the cell is cut
                          // in multiple particles.
                          if (cutting_particle_frames.empty())
                            {
                              cell_is_cut                      = true;
                              particle_id_which_cuts_this_cell = p;
                              cell_is_inside                   = false;
                              particle_id_in_which_this_cell_is_embedded = 0;
                            }
                          if (std::ranges::none_of(
                                cutting_particle_frames,
                                [p](const auto &particle_frame) {
                                  return particle_frame.particle_id == p;
                                }))
                            {
                              cutting_particle_frames.push_back(
                                {p, best_periodic_shift});
                            }
                          if (cutting_particle_frames.size() > 1)
                            {
                              for (const auto &dof_index : local_dof_indices)
                                {
                                  dof_with_more_then_one_particle[dof_index] =
                                    true;
                                }
                            }
                        }
                    }
                  else
                    {
                      if (cutting_particle_frames.empty())
                        {
                          cell_is_cut                                = false;
                          particle_id_which_cuts_this_cell           = 0;
                          cell_is_inside                             = false;
                          particle_id_in_which_this_cell_is_embedded = 0;
                        }
                    }
                }
            }

          if (mapping_overconstrained_cells)
            {
              // If a cell is cut, we register its DOFs as "cut" also. This
              // will allow us to detect fluid cells that have all their DOFs
              // cut, and consider them as cut cells.
              if (cell_is_cut)
                {
                  for (const auto &id : local_dof_indices)
                    {
                      local_dof_overconstrained(id) = 1;
                    }
                }

              // Count boundary DoFs as constrained, except on a weak outlet
              // in periodic Sharp simulations. The outlet does not prescribe
              // the velocity and must not make a fluid cell overconstrained.
              size_t nb_faces = cell->n_faces();
              for (unsigned int f = 0; f < nb_faces; ++f)
                {
                  const auto face = cell->face(f);
                  if (face->at_boundary())
                    {
                      const auto &types =
                        this->simulation_parameters.boundary_conditions.type;
                      const auto boundary_type = types.find(face->boundary_id());
                      if (sharp_ib_periodic_boundaries.is_periodic_enabled() &&
                          boundary_type != types.end() &&
                          boundary_type->second ==
                            BoundaryConditions::BoundaryType::outlet)
                        continue;
                      face->get_dof_indices(local_face_dof_indices);
                      for (const auto &id : local_face_dof_indices)
                        {
                          // SHARP-TEMPORARY-DEBUG-TOP
                          // Compare geometric boundary marking with marking
                          // only DoFs that carry actual affine constraints.
                          if (std::getenv("SHARP_DEBUG_CONSTRAINED_BOUNDARY_ONLY") !=
                                nullptr &&
                              !this->zero_constraints.is_constrained(id) &&
                              !this->nonzero_constraints.is_constrained(id))
                            continue;
                          // SHARP-TEMPORARY-DEBUG-BOTTOM
                          local_dof_overconstrained(id) = 1;
                        }
                    }
                }
            }

          cut_cells_map[cell]    = {cell_is_cut,
                                    particle_id_which_cuts_this_cell,
                                    cutting_particle_frames};
          cells_inside_map[cell] = {cell_is_inside,
                                    particle_id_in_which_this_cell_is_embedded};
        }
    }

  if (mapping_overconstrained_cells)
    {
      local_dof_overconstrained.compress(VectorOperation::insert);
      dof_overconstrained = local_dof_overconstrained;


      for (const auto &cell : cell_iterator)
        {
          if (cell->is_locally_owned() || cell->is_ghost())
            {
              bool cell_is_cut;
              bool cell_is_inside;
              cell->get_dof_indices(local_dof_indices);
              cell_is_cut    = cut_cells_map[cell].is_cut;
              cell_is_inside = cells_inside_map[cell].is_inside;
              if (!cell_is_cut && !cell_is_inside)
                {
                  unsigned int number_of_vertices_in_cell =
                    local_dof_indices.size();
                  unsigned int number_of_vertices_cut = 0;
                  unsigned int particle_current_id =
                    std::numeric_limits<int>::max();

                  double particle_current_distance(
                    std::numeric_limits<double>::max());
                  double particle_candidate_distance(
                    std::numeric_limits<double>::max());
                  Tensor<1, dim> periodic_shift_for_overconstrained_cell;

                  unsigned int global_dof_id;
                  // We count the number of vertices that are constrained.
                  for (unsigned int j = 0; j < number_of_vertices_in_cell; ++j)
                    {
                      global_dof_id = local_dof_indices[j];
                      if (dof_overconstrained(global_dof_id) > 0)
                        {
                          number_of_vertices_cut += 1;
                        }
                    }
                  // We check which particle is the closest to the cell, so
                  // that we know which one to use for applying the immersed
                  // boundary constraints.
                  if (number_of_vertices_in_cell == number_of_vertices_cut)
                    {
                      for (unsigned int p = 0; p < particles.size(); p++)
                        {
                          const Point<dim> saved_particle_position =
                            particles[p].position;
                          const auto candidate_periodic_shifts =
                            get_periodic_particle_shifts(
                              saved_particle_position,
                              particle_support_radius(particles[p]));

                          Tensor<1, dim> best_periodic_shift;
                          for (const auto &periodic_shift :
                               candidate_periodic_shifts)
                            {
                              set_particle_position_from_periodic_shift(
                                p, saved_particle_position, periodic_shift);
                              const double candidate_distance =
                                particles[p].get_levelset(cell->barycenter(),
                                                          cell);
                              if (std::abs(candidate_distance) <
                                  std::abs(particle_candidate_distance))
                                {
                                  particle_candidate_distance =
                                    candidate_distance;
                                  best_periodic_shift = periodic_shift;
                                }
                            }

                          if (particles[p].position != saved_particle_position)
                            particles[p].set_position(saved_particle_position);

                          if (abs(particle_candidate_distance) <
                              abs(particle_current_distance))
                            {
                              particle_current_id = p;
                              particle_current_distance =
                                particle_candidate_distance;
                              periodic_shift_for_overconstrained_cell =
                                best_periodic_shift;
                            }
                        }

                      for (const auto &local_dof_index : local_dof_indices)
                        {
                          dof_with_more_then_one_particle[local_dof_index] =
                            true;
                        }

                      overconstrained_fluid_cell_map[cell] = {
                        true,
                        particle_current_id,
                        particle_current_distance,
                        periodic_shift_for_overconstrained_cell};
                    }
                }
            }
        }
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::refinement_control(const bool initial_refinement)
{
  // We add a post-refinement check to update precalculations only if needed.
  // Two booleans are used: the first one is used to update the precalculations,
  // and the other one is used to remove superfluous nodes. We make this
  // separation between the steps to avoid removing useful information until all
  // triangulation refinement steps are done.
  bool update_precalculations_flag  = false;
  bool remove_superfluous_data_flag = false;
  // The slot captures a local variable, so the connection is scoped to
  // disconnect it when leaving this function.
  const boost::signals2::scoped_connection post_refinement_connection =
    this->triangulation->signals.post_refinement.connect(
      [&update_precalculations_flag]() { update_precalculations_flag = true; });

  //  This function applies the various refinement steps depending on the
  //  parameters and the state.
  if (initial_refinement)
    {
      // Apply the initial box refinement
      this->box_refine_mesh(false);
      update_precalculations_for_ib();
    }
  if (this->simulation_parameters.particlesParameters
        ->time_extrapolation_of_refinement_zone ||
      initial_refinement)
    {
      // Stores variable for refinement around the particle.
      double temp_refine =
        this->simulation_parameters.mesh_adaptation.variables.begin()
          ->second.refinement_fraction;
      double temp_coarse =
        this->simulation_parameters.mesh_adaptation.variables.begin()
          ->second.coarsening_fraction;
      this->simulation_parameters.mesh_adaptation.variables.begin()
        ->second.refinement_fraction = 0;
      this->simulation_parameters.mesh_adaptation.variables.begin()
        ->second.coarsening_fraction = 0;

      for (unsigned int i = 0;
           i <
           this->simulation_parameters.particlesParameters->initial_refinement;
           ++i)
        {
          update_precalculations_flag = false;
          this->pcout << "Initial refinement around IB particles - Step : "
                      << i + 1 << " of "
                      << this->simulation_parameters.particlesParameters
                           ->initial_refinement
                      << std::endl;
          mesh_adapt_ib(initial_refinement);
          NavierStokesBase<dim, GlobalVectorType, IndexSet>::refine_mesh();
          if (update_precalculations_flag)
            {
              remove_superfluous_data_flag = true;
              update_precalculations_for_ib();
            }
        }
      this->simulation_parameters.mesh_adaptation.variables.begin()
        ->second.refinement_fraction = temp_refine;
      this->simulation_parameters.mesh_adaptation.variables.begin()
        ->second.coarsening_fraction = temp_coarse;
    }
  if (initial_refinement == false)
    {
      update_precalculations_flag = false;
      mesh_adapt_ib(initial_refinement);
      NavierStokesBase<dim, GlobalVectorType, IndexSet>::refine_mesh();
      if (update_precalculations_flag)
        {
          remove_superfluous_data_flag = true;
          update_precalculations_for_ib();
        }
    }
  if (remove_superfluous_data_flag || initial_refinement)
    {
      for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
        {
          TimerOutput::Scope t(this->computing_timer, "Remove RBF nodes");
          particles[p_i].remove_superfluous_data(
            *this->dof_handler, particles[p_i].mesh_based_precalculations);
        }
    }
}

template <int dim>
bool
FluidDynamicsSharp<dim>::cell_cut_by_p_absolute_distance(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  std::map<types::global_dof_index, Point<dim>>        &support_points,
  unsigned int                                          p)
{
  // This function aims at defining if a cell is cut when the level set used to
  // define the particle is not signed. This function works by analyzing the
  // location of the projection of the support point and the cell centroid on
  // the particle's surface. If the projected point at the surface of the
  // particle is inside the cell then the cell is cut.

  // First step: define useful variables.
  const unsigned int                   dofs_per_cell = this->fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);
  bool       cell_is_cut      = false;
  Point<dim> centroid_of_cell = cell->barycenter();
  Point<dim> projected_point;
  particles[p].closest_surface_point(centroid_of_cell, projected_point, cell);

  // Check the centroid of the cell first.
  if (cell->point_inside(projected_point))
    {
      cell_is_cut = true;
      return cell_is_cut;
    }
  else
    {
      // If the projection of the centroid is not inside the cell we also check
      // for each dof support point of the cell.
      for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
        {
          // Only check for the support point of the velocity in x.
          if (0 == this->fe->system_to_component_index(j).first)
            {
              particles[p].closest_surface_point(
                support_points[local_dof_indices[j]], projected_point, cell);

              if (cell->point_inside(projected_point))
                {
                  cell_is_cut = true;
                  break;
                }
              if ((projected_point - support_points[local_dof_indices[j]])
                    .norm() < cell->diameter() * 0.1)
                {
                  cell_is_cut = true;
                  break;
                }
            }
        }
    }
  return cell_is_cut;
}

template <int dim>
void
FluidDynamicsSharp<dim>::optimized_generate_cut_cells_map()
{
  TimerOutput::Scope t(this->computing_timer, "Optmize cut cells mapping");
  MappingQ1<dim>     mapping;
  unsigned int       max_children = GeometryInfo<dim>::max_children_per_cell;

  std::set<typename DoFHandler<dim>::cell_iterator> all_candidate_cells;
  std::set<typename DoFHandler<dim>::cell_iterator>
    previous_all_candidate_cells;

  cut_cells_map.clear();
  cells_inside_map.clear();
  for (const auto &cell : this->dof_handler->active_cell_iterators())
    {
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cut_cells_map[cell]    = {false,
                                    0,
                                    std::vector<ParticlePeriodicFrame>()};
          cells_inside_map[cell] = {false, 0};
        }
    }

  // Loop over particles in reverse.
  // This is done in reverse to guarantee that the lowest particle ID is
  // associated to the cut cell or the inside particle cell. For instance, if
  // you have two particles cutting the same cell, it guarantees that the lowest
  // ID particle is associated with the cell in the map.
  for (int p = particles.size() - 1; p >= 0; --p)
    {
      bool         empty    = true;
      unsigned int lvl_iter = 0;

      // Fix max level search
      unsigned int max_lvl_search =
        this->dof_handler->get_triangulation().n_levels() - 2;
      if (max_lvl_search < 4)
        max_lvl_search = this->dof_handler->get_triangulation().n_levels();

      // Search for candidates until at least one is found or until the
      // max_lvl_search is reached
      while (empty || lvl_iter < max_lvl_search)
        {
          const auto &cell_iterator =
            this->dof_handler->cell_iterators_on_level(lvl_iter);

          // Loop over the cells on level lvl_iter of the mesh
          for (const auto &cell : cell_iterator)
            {
              auto is_candidate = generate_cut_cell_candidates(cell, p);

              // check whether the cell is candidate for inside of cut
              if (is_candidate.first)
                {
                  all_candidate_cells.insert(cell);
                }
              if (is_candidate.second)
                {
                  all_candidate_cells.insert(cell);
                }
            }
          if (all_candidate_cells.size() > 0)
            empty = false;

          lvl_iter += 1;
        }

      // Store list with all candidate cells
      previous_all_candidate_cells = all_candidate_cells;

      // If there are candidate cells
      if (all_candidate_cells.size() != 0)
        {
          bool all_cells_are_active = false;

          // Loop until candidate cells are active
          while (!all_cells_are_active)
            {
              all_cells_are_active = true;
              all_candidate_cells.clear();

              // loop over the last set of candidates to check if it is still a
              // candidate
              for (auto cell : previous_all_candidate_cells)
                {
                  auto is_candidate = generate_cut_cell_candidates(cell, p);

                  // Check if cell is inside particles
                  if (is_candidate.first)
                    {
                      // If it is an active cell, add it to cells_inside_map
                      if (cell->is_active())
                        {
                          cells_inside_map[cell] = {
                            true, static_cast<unsigned int>(p)};
                        }
                      else
                        {
                          // If we are here, the cell has children.
                          all_cells_are_active = false;
                          for (unsigned int j = 0; j < max_children; ++j)
                            {
                              // Store the children of the cell for
                              // further checks.
                              all_candidate_cells.insert(cell->child(j));
                            }
                        }
                    }

                  if (is_candidate.second)
                    {
                      // If it is an active cell, add it to cells_cut_map
                      if (cell->is_active())
                        {
                          cut_cells_map[cell] = {
                            true,
                            static_cast<unsigned int>(p),
                            std::vector<ParticlePeriodicFrame>()};
                        }
                      else
                        {
                          // If we are here, the cell has children.
                          all_cells_are_active = false;
                          for (unsigned int j = 0; j < max_children; ++j)
                            {
                              // Store the children of the cell for
                              // further checks.
                              all_candidate_cells.insert(cell->child(j));
                            }
                        }
                    }
                }
              previous_all_candidate_cells.clear();
              previous_all_candidate_cells = all_candidate_cells;
            }
        }
    }
}


template <int dim>
std::pair<bool, bool>
FluidDynamicsSharp<dim>::generate_cut_cell_candidates(
  const typename DoFHandler<dim>::cell_iterator &cell,
  const unsigned int                             p_id)
{
  bool cell_is_inside = false;
  bool cell_is_cut    = false;

  double search_radius = particles[p_id].radius * (1.0 - 1e-07);
  auto   position      = particles[p_id].position;

  bool point_inside_cell = cell->point_inside(position);

  // Check how many vertices are inside the particle
  unsigned int nb_vertices_inside = 0;
  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
    {
      if ((cell->vertex(i) - position).norm() < search_radius)
        ++nb_vertices_inside;
    }

  // If vertices are found inside the particle
  if (nb_vertices_inside > 0)
    {
      // If the number of vertices inside the cell is equal to the number of
      // vertices per cell, the cell is inside the particle
      if (nb_vertices_inside == GeometryInfo<dim>::vertices_per_cell)
        {
          cell_is_inside = true;
          cell_is_cut    = false;
        }
      // Otherwise, the cell is cut
      else
        {
          cell_is_inside = false;
          cell_is_cut    = true;
        }
      return {cell_is_inside, cell_is_cut};
    }

  // If the particle intersect the cell and is not known whether all
  // vertices are inside the particles, set all true by default
  if (point_inside_cell)
    {
      cell_is_inside = true;
      cell_is_cut    = true;
      return {cell_is_inside, cell_is_cut};
    }

  // The last check consists of projecting the particle's position on the cells'
  // face and checking whether the projected points are within the particle. If
  // one of the projected points is inside the particle, either the cell is
  // inside or is cut by the particles. If this is true, the next
  // level will be tested again by this function.

  // Initialize superpoint of manifold
  std::vector<Point<dim>> manifold_points(
    GeometryInfo<dim - 1>::vertices_per_cell);

  // Loop through faces
  for (const auto face : cell->face_indices())
    {
      auto  face_iter           = cell->face(face);
      auto &local_face_manifold = face_iter->get_manifold();

      // Loop through face vertices
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
        {
          // Assign vertex to manifold superpoint
          manifold_points[i] = face_iter->vertex(i);
        }

      // Create array of points on face
      auto surrounding_face_points =
        make_array_view(manifold_points.begin(), manifold_points.end());

      // Loop through face vertices
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
        {
          // Project points to face
          Point<dim> projected_point =
            local_face_manifold.project_to_manifold(surrounding_face_points,
                                                    position);

          bool           projected_point_over_face = true;
          Tensor<1, dim> projected_point_tensor(projected_point);

          // Check whether the projected point is within the cell
          try
            {
              projected_point_over_face = cell->point_inside(projected_point);

              if ((position - projected_point).norm() < search_radius &&
                  projected_point_over_face)
                {
                  cell_is_inside = true;
                  cell_is_cut    = true;
                  return {cell_is_inside, cell_is_cut};
                }
            }
          // If the method crashes, change default for false
          catch (...)
            {
              projected_point_over_face = false;
            }

          for (int d = 0; d != dim; d++)
            {
              if (projected_point_tensor[d] < 0 ||
                  projected_point_tensor[d] > 1)
                {
                  projected_point_over_face = false;
                }
            }
        }
    }
  return {cell_is_inside, cell_is_cut};
}

template <int dim>
void
FluidDynamicsSharp<dim>::define_particles()
{
  some_particles_are_coupled = false;
  // initialized the particles
  if (this->simulation_parameters.particlesParameters
        ->load_particles_from_file == false)
    {
      particles.resize(
        this->simulation_parameters.particlesParameters->nb_particles);
      for (unsigned int i = 0;
           i < this->simulation_parameters.particlesParameters->nb_particles;
           ++i)
        {
          particles[i] =
            this->simulation_parameters.particlesParameters->particles[i];
          if (particles[i].integrate_motion == true)
            {
              some_particles_are_coupled = true;
            }
        }
    }
  else
    {
      // The number of particles is defined by the file, not by the "number of
      // particles" parameter.
      load_particles_from_file();
      for (unsigned int i = 0; i < particles.size(); ++i)
        {
          if (particles[i].integrate_motion == true)
            {
              some_particles_are_coupled = true;
            }
        }
    }

  TimerOutput::Scope timer(this->computing_timer,
                           "Initialize particle coupling");
  table_p.resize(particles.size());
  ib_dem.initialize(
    this->simulation_parameters.particlesParameters,
    std::make_shared<Parameters::Lagrangian::FloatingWalls<dim>>(
      cfd_dem_parameters.dem_parameters.floating_walls),
    std::make_shared<Parameters::Lagrangian::BCDEM>(
      cfd_dem_parameters.dem_parameters.boundary_conditions),
    this->mpi_communicator,
    particles);

  check_whether_all_particles_are_sphere();

  std::vector<std::shared_ptr<Shape<dim>>> all_shapes(particles.size());
  for (const IBParticle<dim> &particle : particles)
    {
      all_shapes[particle.particle_id] = particle.shape;
    }
  combined_shapes =
    std::make_shared<CompositeShape<dim>>(all_shapes, Point<dim>(), Point<3>());
  this->multiphysics->set_immersed_solid_shape(combined_shapes);
}


template <int dim>
void
FluidDynamicsSharp<dim>::mesh_adapt_ib(const bool initial_refinement)
{
  // If this is not the initial refinement cycle nor a usual refinement step,
  // we refine the cells around immersed solids only if they have moved since
  // the last time step.
  if (!initial_refinement && !this->simulation_control->is_refinement_step(
                               this->simulation_parameters.mesh_adaptation))
    {
      bool stationary_particles = true;
      for (unsigned int p = 0; p < particles.size(); ++p)
        {
          if ((particles[p].position - particles[p].previous_positions[0])
                  .norm() > 1e-16 ||
              (particles[p].orientation - particles[p].previous_orientation[0])
                  .norm() > 1e-16)
            {
              stationary_particles = false;
              break;
            }
        }
      if (stationary_particles)
        return;
    }

  TimerOutput::Scope                            t(this->computing_timer,
                       "Refine around immersed boundary");
  Point<dim>                                    center_immersed;
  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(*this->mapping, *this->dof_handler);

  double dt = this->simulation_control->get_time_steps_vector()[0];

  const unsigned int                   dofs_per_cell = this->fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  bool extrapolate_particle_position =
    this->simulation_parameters.particlesParameters
      ->time_extrapolation_of_refinement_zone;

  double smallest_cut_cell = std::numeric_limits<double>::max();
  bool   minimal_crown_refinement_enabled =
    abs(this->simulation_parameters.particlesParameters
          ->refinement_outside_distance_factor -
        1) < 1e-16 &&
    abs(this->simulation_parameters.particlesParameters
          ->refinement_inside_distance_factor -
        1) < 1e-16;
  if (minimal_crown_refinement_enabled)
    {
      const auto &cell_iterator_smallest_cell =
        this->dof_handler->active_cell_iterators();
      for (const auto &cell : cell_iterator_smallest_cell)
        {
          if (cell->is_locally_owned())
            smallest_cut_cell = std::min(smallest_cut_cell, cell->diameter());
        }
      smallest_cut_cell =
        Utilities::MPI::min(smallest_cut_cell, this->mpi_communicator);
    }

  const auto &cell_iterator = this->dof_handler->active_cell_iterators();
  for (const auto &cell : cell_iterator)
    {
      if (cell->is_locally_owned())
        {
          bool cell_can_be_coarsened =
            this->simulation_parameters.particlesParameters->enable_coarsening;

          cell->get_dof_indices(local_dof_indices);
          for (unsigned int p = 0; p < particles.size(); ++p)
            {
              unsigned int count_small = 0;
              center_immersed          = particles[p].position;
              Tensor<1, dim> r;
              r[0] = particles[p].radius;

              Point<dim>   particle_position;
              Tensor<1, 3> particle_orientation;
              if (extrapolate_particle_position)
                {
                  particle_position    = particles[p].position;
                  particle_orientation = particles[p].orientation;
                  particles[p].orientation =
                    particles[p].orientation + particles[p].omega * dt;
                  particles[p].position[0] =
                    particles[p].position[0] + particles[p].velocity[0] * dt;
                  particles[p].position[1] =
                    particles[p].position[1] + particles[p].velocity[1] * dt;

                  if constexpr (dim == 3)
                    {
                      particles[p].position[2] = particles[p].position[2] +
                                                 particles[p].velocity[2] * dt;
                    }
                  particles[p].set_position(particles[p].position);
                  particles[p].set_orientation(particles[p].orientation);
                }

              if (extrapolate_particle_position &&
                  sharp_ib_periodic_boundaries.is_periodic_enabled())
                {
                  const Point<dim> normalized =
                    sharp_ib_periodic_boundaries.wrap_point(
                      particles[p].position,
                      periodic_domain_lower,
                      periodic_domain_upper);
                  if (normalized != particles[p].position)
                    particles[p].set_position(normalized);
                }

              // Build list of particle positions to check for refinement:
              // actual position + periodic images if near periodic boundary
              std::vector<Point<dim>> positions_to_check;
              positions_to_check.push_back(particles[p].position);

              if (sharp_ib_periodic_boundaries.is_periodic_enabled())
                {
                  // The crown thickness is measured from the shape surface,
                  // not from its reference point. Include coarsening support.
                  const auto &parameters =
                    *this->simulation_parameters.particlesParameters;
                  const double crown_thickness =
                    minimal_crown_refinement_enabled ?
                      smallest_cut_cell / std::sqrt(dim) :
                      particles[p].radius *
                        std::max(0.,
                                 parameters.refinement_outside_distance_factor -
                                   1.);
                  const double coarsening_thickness =
                    cell_can_be_coarsened ?
                      particles[p].radius *
                        std::max(0.,
                                 parameters.coarsening_distance_factor - 1.) :
                      0.;
                  const double refinement_radius =
                    particle_support_radius(particles[p]) +
                    std::max(crown_thickness, coarsening_thickness);

                  // Get periodic images for this particle
                  std::vector<Point<dim>> images =
                    get_periodic_particle_images(particles[p].position,
                                                 refinement_radius);

                  // Add images to list of positions to check
                  positions_to_check.insert(positions_to_check.end(),
                                            images.begin(),
                                            images.end());
                }
              // Check if a point on the random point on the IB is contained in
              // that cell. If the particle is much smaller than the cell, all
              // its vertices may be outside of the particle. In that case the
              // cell won't be refined. To prevent that, we check if a random
              // point on the boundary is contained in the cell.
              bool cell_as_ib_inside = false;
              for (const auto &pos : positions_to_check)
                {
                  if (cell->point_inside(pos + r))
                    {
                      cell_as_ib_inside = true;
                      break;
                    }
                }

              bool cell_is_near_particle_for_coarsening = false;

              for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
                {
                  // Only check the dof of velocity in x.
                  if (this->fe->system_to_component_index(j).first == 0)
                    {
                      // Check if this DoF is in crown around any position
                      // (actual particle or periodic image)
                      for (const auto &pos : positions_to_check)
                        {
                          // Temporarily set particle position to check position
                          Point<dim> saved_particle_position =
                            particles[p].position;
                          if (particles[p].position != pos)
                            particles[p].set_position(pos);

                          bool is_inside_crown;
                          if (minimal_crown_refinement_enabled)
                            {
                              // The factor should be equal to the length of the
                              // current smallest cell. Since cell->diameter()
                              // returns the longest diagonal, we divide the
                              // diameter by the diagonal of a unit hypercube to
                              // obtain the correct length.
                              double factor   = 1 / sqrt(dim);
                              is_inside_crown = particles[p].is_inside_crown(
                                support_points[local_dof_indices[j]],
                                factor * smallest_cut_cell,
                                -factor * smallest_cut_cell,
                                true, // indicates that we use the absolute
                                      // distance definition
                                cell);
                            }
                          else
                            {
                              is_inside_crown = particles[p].is_inside_crown(
                                support_points[local_dof_indices[j]],
                                this->simulation_parameters.particlesParameters
                                  ->refinement_outside_distance_factor,
                                this->simulation_parameters.particlesParameters
                                  ->refinement_inside_distance_factor,
                                false, // indicates that we use the
                                       // radius relative distance definition
                                cell);
                            }

                          if (cell_can_be_coarsened)
                            {
                              const double coarsening_distance =
                                particles[p].shape->value_with_cell_guess(
                                  support_points[local_dof_indices[j]], cell);
                              const double coarsening_threshold =
                                particles[p].shape->effective_radius *
                                (this->simulation_parameters
                                   .particlesParameters
                                   ->coarsening_distance_factor -
                                 1);

                              if (coarsening_distance <= coarsening_threshold)
                                {
                                  cell_is_near_particle_for_coarsening = true;
                                }
                            }

                          // Restore actual particle position
                          if (particles[p].position != saved_particle_position)
                            particles[p].set_position(saved_particle_position);

                          if (is_inside_crown)
                            {
                              ++count_small;
                              break; // Don't double-count for multiple images
                            }

                          // Another image may still require refinement even
                          // after this image has ruled out coarsening.
                        }
                    }
                }

              if (extrapolate_particle_position)
                {
                  particles[p].position    = particle_position;
                  particles[p].orientation = particle_orientation;
                  particles[p].set_position(particles[p].position);
                  particles[p].set_orientation(particles[p].orientation);
                }

              if (cell_as_ib_inside || cell_is_near_particle_for_coarsening)
                cell_can_be_coarsened = false;

              if (count_small > 0 || cell_as_ib_inside)
                {
                  cell->set_refine_flag();
                  break;
                }
            }

          if (cell_can_be_coarsened && !cell->refine_flag_set())
            cell->set_coarsen_flag();
        }
    }
}



template <int dim>
void
FluidDynamicsSharp<dim>::force_on_ib()
{
  // This function defines the force and torque applied on an Immersed Boundary
  // based on the sharp edge method on a hyper_sphere of dim=2 or dim=3

  TimerOutput::Scope t(this->computing_timer,
                       "Evaluate force on immersed boundary");

  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Vector velocities(0);

  std::vector<double> time_steps_vector =
    this->simulation_control->get_time_steps_vector();
  // Define a map to all DOFs and their support points
  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(*this->mapping, *this->dof_handler);

  // Initalize fe value objects in order to do calculation with it later
  QGauss<dim>            q_formula(this->number_quadrature_points);
  MappingQ<dim - 1, dim> local_face_map(
    (2 > this->simulation_parameters.fem_parameters.velocity_degree) ?
      2 :
      this->simulation_parameters.fem_parameters.velocity_degree);

  FESystem<dim - 1, dim> local_face_fe(
    FE_Q<dim - 1, dim>(
      this->simulation_parameters.fem_parameters.velocity_degree),
    dim,
    FE_Q<dim - 1, dim>(
      this->simulation_parameters.fem_parameters.pressure_degree),
    1);
  FEValues<dim - 1, dim> fe_face_projection_values(local_face_map,
                                                   local_face_fe,
                                                   *this->face_quadrature,
                                                   update_values |
                                                     update_quadrature_points |
                                                     update_gradients |
                                                     update_JxW_values |
                                                     update_normal_vectors);



  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int dofs_per_face = this->fe->dofs_per_face;

  AssertThrow(this->simulation_parameters.physical_properties_manager
                .density_is_constant(),
              RequiresConstantDensity("FluidDynamicsSharp<dim>::force_on_ib"));

  unsigned int degree =
    this->simulation_parameters.particlesParameters->stencil_degree;
  auto density_model =
    this->simulation_parameters.physical_properties_manager.get_density(0);
  std::map<field, double> field_values;
  field_values[field::temperature] = 0;
  double fluid_density             = density_model->value(field_values);
  double length_ratio =
    this->simulation_parameters.particlesParameters->length_ratio;
  IBStencil<dim>      stencil;
  std::vector<double> ib_coef = stencil.coefficients(degree, length_ratio);

  // Rheological model for viscosity properties
  double     kinematic_viscosity;
  const auto rheological_model =
    this->simulation_parameters.physical_properties_manager.get_rheology();

  const unsigned int vertices_per_face = GeometryInfo<dim>::vertices_per_face;
  const unsigned int n_q_points_face   = this->face_quadrature->size();

  std::vector<double>                      pressure_values(ib_coef.size());
  Tensor<1, dim>                           normal_vector;
  std::vector<Tensor<2, dim>>              velocity_gradients(ib_coef.size());
  std::vector<std::vector<Tensor<1, dim>>> velocity_gradients_component(dim +
                                                                        1);
  for (int i = 0; i < dim + 1; ++i)
    velocity_gradients_component[i].resize(ib_coef.size());
  Tensor<2, dim>              fluid_viscous_stress;
  Tensor<2, dim>              fluid_pressure;
  Tensor<2, dim>              fluid_viscous_stress_at_ib;
  Tensor<2, dim>              fluid_pressure_stress_at_ib;
  Tensor<2, dim>              shear_rate;
  DoFHandler<dim - 1, dim>    local_face_dof_handler;
  Triangulation<dim - 1, dim> local_face_projection_triangulation;

  std::vector<Point<dim>>        vertices_of_face_projection(vertices_per_face);
  std::vector<CellData<dim - 1>> local_face_cell_data(1);

  typename Mapping<dim>::InternalDataBase mapping_data;
  // Define multiple local_dof_indices one for the cell iterator one for the
  // cell with the second point for the sharp edge stencil and one for
  // manipulation on the neighbour’s cell.

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_2(dofs_per_cell);

  std::vector<types::global_dof_index> local_face_dof_indices(dofs_per_face);

  std::vector<Point<dim>> unite_cell_interpolation_points(ib_coef.size());
  std::vector<Point<dim>> cell_interpolation_points(ib_coef.size());
  std::vector<double>     local_interp_sol(ib_coef.size());

  // Cache extrapolated force tensors per (DoF, particle key, periodic frame).
  // Primary frames share key zero; multi-particle DoFs are recomputed below.
  // Near a periodic interface the same geometric DoF can be visited from
  // different wrapped frames of the same particle, and reusing the stress
  // tensor across those frames is incorrect because the supporting stencil
  // cell can change. The cache key therefore mirrors the frame-aware stencil
  // cache used in sharp_edge().
  std::map<PeriodicFrameDofCacheKey, std::pair<Tensor<2, dim>, Tensor<2, dim>>>
    force_eval;

  // Define cell iterator
  const auto &cell_iterator = this->dof_handler->active_cell_iterators();

  // Clear particle force and torque
  for (unsigned int i = 0; i < particles.size(); ++i)
    {
      particles[i].fluid_forces          = 0;
      particles[i].fluid_viscous_forces  = 0;
      particles[i].fluid_pressure_forces = 0;
      particles[i].fluid_torque          = 0;
    }

  // Loop over all the cell
  for (const auto &cell : cell_iterator)
    {
      if (cell->is_locally_owned())
        {
          // Particle id that cut the cell.
          const auto &cut_cell_info = cut_cells_map[cell];
          bool        cell_is_cut   = cut_cell_info.is_cut;
          // If the cell is cut
          if (cell_is_cut)
            {
              // loop over all particles that cut this precise cell
              for (const auto &particle_frame :
                   cut_cell_info.cutting_particle_frames)
                {
                  const unsigned int p = particle_frame.particle_id;
                  const int          particle_frame_index =
                    get_periodic_frame_index(particle_frame.periodic_shift);
                  // Preserve the primary-frame per-DoF cache. Distinguish
                  // translated particle images without changing existing IB
                  // assembly or force reuse away from the periodic interface.
                  const unsigned int cache_particle_id =
                    particle_frame_index != 0 ? p : 0;
                  const Point<dim> saved_force_particle_pos =
                    particles[p].position;
                  set_particle_position_from_periodic_shift(
                    p, saved_force_particle_pos, particle_frame.periodic_shift);

                  // Loop over all cut cell faces'.
                  for (const auto face : cell->face_indices())
                    {
                      auto local_face = cell->face(face);
                      cell->face(face)->get_dof_indices(local_face_dof_indices);

                      // Check if the face is cut
                      unsigned int nb_dof_inside = 0;
                      for (const auto &dof_index : local_face_dof_indices)
                        {
                          // Count the number of DOFs that are inside
                          // of the particles. If all the DOfs are on one side
                          // the cell is not cut by the boundary.
                          if (particles[p].get_levelset(
                                support_points[dof_index], cell) <= 0)
                            ++nb_dof_inside;
                        }

                      // If the face is not cut and the face is outside the
                      // IB, the face of the cell is on the boundary of the
                      // computational domain.
                      if (nb_dof_inside == 0)
                        {
                          // Create a vector to approximate the normal of the
                          // cell to orient the projection of the face.
                          Tensor<1, dim> approximate_surface_cell_normal;
                          // Project the face on the surface of the IB. This
                          // creates a surface cell where we can evaluate the
                          // solution. Define the triangulation of the surface
                          // cell.
                          for (unsigned int i = 0; i < vertices_per_face; ++i)
                            {
                              Point<dim> vertex_projection;
                              particles[p].closest_surface_point(
                                local_face->vertex(i), vertex_projection, cell);
                              approximate_surface_cell_normal +=
                                (local_face->vertex(i) - vertex_projection) /
                                ((local_face->vertex(i) - vertex_projection)
                                   .norm() +
                                 DBL_MIN);

                              // Define the vertices of the surface cell.
                              // Create the list of vertices
                              for (int j = 0; j < dim; ++j)
                                {
                                  vertices_of_face_projection[i][j] =
                                    vertex_projection[j];
                                }
                              // Create the connectivity of the vertices of the
                              // cell
                              local_face_cell_data[0].vertices[i] = i;
                            }
                          approximate_surface_cell_normal =
                            approximate_surface_cell_normal / vertices_per_face;

                          local_face_cell_data[0].material_id = 0;

                          local_face_projection_triangulation =
                            Triangulation<dim - 1, dim>();
                          // Create a dof_handler that contains the
                          // triangulation of the projection of the face on the
                          // IB. This creates the surface cell on the IB
                          local_face_projection_triangulation
                            .create_triangulation(vertices_of_face_projection,
                                                  local_face_cell_data,
                                                  SubCellData());
                          local_face_projection_triangulation
                            .set_all_manifold_ids(0);
                          local_face_projection_triangulation.set_manifold(
                            0, *particles[p].shape->get_shape_manifold());

                          local_face_dof_handler.reinit(
                            local_face_projection_triangulation);
                          local_face_dof_handler.distribute_dofs(local_face_fe);

                          // Define the solution on IB surface cell by using
                          // the IB stencil to extrapolate the fluid stress
                          // tensor.
                          std::vector<Tensor<2, dim>>
                            local_face_viscous_stress_tensor(dofs_per_face);
                          std::vector<Tensor<2, dim>>
                            local_face_pressure_tensor(dofs_per_face);
                          for (unsigned int i = 0;
                               i < local_face_dof_indices.size();
                               ++i)
                            {
                              const unsigned int component_i =
                                this->fe->face_system_to_component_index(i)
                                  .first;
                              const PeriodicFrameDofCacheKey force_cache_key = {
                                local_face_dof_indices[i],
                                cache_particle_id,
                                particle_frame_index};
                              // Reuse primary-frame stresses as before, while
                              // keeping translated images separate by particle
                              // and periodic frame. Shared multi-particle DoFs
                              // still require a fresh evaluation below.
                              if (force_eval.find(force_cache_key) ==
                                    force_eval.end() or
                                  dof_with_more_then_one_particle
                                      [local_face_dof_indices[i]] == true)
                                {
                                  if (component_i == 0)
                                    {
                                      // Only need one extrapolation by dof
                                      // location;

                                      // Count the number of evaluation

                                      auto [point, interpolation_points] =
                                        stencil
                                          .support_points_for_interpolation(
                                            degree,
                                            length_ratio,
                                            particles[p],
                                            support_points
                                              [local_face_dof_indices[i]],
                                            cell);

                                      auto           cell_2 = cell;
                                      Tensor<1, dim> force_periodic_shift;
                                      const PeriodicFrameDofCacheKey
                                        stencil_cache_key = {
                                          local_face_dof_indices[i],
                                          cache_particle_id,
                                          particle_frame_index};
                                      // Check if we already have the cell used
                                      // to define the IB constraint of that
                                      // dof. We always have that information
                                      // except if the dof is not owned.
                                      const auto cached_stencil_cell =
                                        ib_done.find(stencil_cache_key);
                                      if (cached_stencil_cell ==
                                            ib_done.end() or
                                          dof_with_more_then_one_particle
                                              [local_face_dof_indices[i]] ==
                                            true)
                                        {
                                          // Get the cell use for the
                                          // extrapolation
                                          auto point_to_find_cell =
                                            stencil.point_for_cell_detection(
                                              particles[p],
                                              support_points
                                                [local_face_dof_indices[i]],
                                              cell);
                                          if (sharp_ib_periodic_boundaries
                                                .is_periodic_enabled())
                                            {
                                              cell_2 =
                                                find_periodic_stencil_cell(
                                                  cell,
                                                  point_to_find_cell,
                                                  force_periodic_shift);
                                              if (cell_2 ==
                                                  this->dof_handler->end())
                                                cell_2 = cell;
                                            }
                                          else
                                            {
                                              try
                                                {
                                                  cell_2 = LetheGridTools::
                                                    find_cell_around_point_with_neighbors<
                                                      dim>(*this->dof_handler,
                                                           vertices_to_cell,
                                                           cell,
                                                           point_to_find_cell);
                                                }
                                              catch (...)
                                                {
                                                  cell_2 = cell;
                                                }
                                            }
                                          if (sharp_ib_periodic_boundaries
                                                .is_periodic_enabled())
                                            ib_done[stencil_cache_key] = {
                                              cell_2, force_periodic_shift};
                                        }
                                      else
                                        {
                                          cell_2 =
                                            cached_stencil_cell->second.first;
                                          force_periodic_shift =
                                            cached_stencil_cell->second.second;
                                        }

                                      cell_2->get_dof_indices(
                                        local_dof_indices_2);

                                      // Shift interpolation points if stencil
                                      // was found across periodic boundary
                                      {
                                        Point<dim> shifted_point;
                                        for (int d = 0; d < dim; ++d)
                                          shifted_point[d] =
                                            point[d] + force_periodic_shift[d];
                                        unite_cell_interpolation_points[0] =
                                          this->mapping
                                            ->transform_real_to_unit_cell(
                                              cell_2, shifted_point);
                                      }
                                      for (unsigned int j = 1;
                                           j < ib_coef.size();
                                           ++j)
                                        {
                                          Point<dim> shifted_point;
                                          for (int d = 0; d < dim; ++d)
                                            shifted_point[d] =
                                              interpolation_points[j - 1][d] +
                                              force_periodic_shift[d];
                                          unite_cell_interpolation_points[j] =
                                            this->mapping
                                              ->transform_real_to_unit_cell(
                                                cell_2, shifted_point);
                                        }

                                      fluid_viscous_stress_at_ib  = 0;
                                      fluid_pressure_stress_at_ib = 0;

                                      // Create a quadrature that is based on
                                      // the IB stencil
                                      Quadrature<dim> q_local(
                                        unite_cell_interpolation_points,
                                        ib_coef);
                                      FEValues<dim> fe_values_cell2(
                                        *this->fe,
                                        q_local,
                                        update_quadrature_points |
                                          update_gradients | update_values);

                                      // Evaluate the relevant information at
                                      // the quadrature points to do the
                                      // extrapolation.
                                      fe_values_cell2.reinit(cell_2);
                                      fe_values_cell2[velocities]
                                        .get_function_gradients(
                                          this->evaluation_point,
                                          velocity_gradients);
                                      fe_values_cell2[pressure]
                                        .get_function_values(
                                          this->evaluation_point,
                                          pressure_values);

                                      // Extrapolate the fluid stress tensor on
                                      // the surface of the IB.

                                      for (unsigned int k = 0;
                                           k < ib_coef.size();
                                           ++k)
                                        {
                                          fluid_pressure = 0;
                                          for (int d = 0; d < dim; ++d)
                                            {
                                              fluid_pressure[d][d] =
                                                pressure_values[k];
                                            }
                                          shear_rate =
                                            velocity_gradients[k] +
                                            transpose(velocity_gradients[k]);


                                          const double shear_rate_magnitude =
                                            calculate_shear_rate_magnitude(
                                              shear_rate);

                                          std::map<field, double> field_values;
                                          field_values[field::shear_rate] =
                                            shear_rate_magnitude;

                                          kinematic_viscosity =
                                            rheological_model->value(
                                              field_values);

                                          fluid_viscous_stress =
                                            -kinematic_viscosity * shear_rate;

                                          fluid_viscous_stress_at_ib -=
                                            fluid_viscous_stress * ib_coef[k];

                                          fluid_pressure_stress_at_ib -=
                                            fluid_pressure * ib_coef[k];
                                        }
                                      // Store the stress tensor that results
                                      // from the extrapolation in the local
                                      // evaluation vector of the IB surface
                                      // cell and in a map that is used if the
                                      // same extrapolation is needed in another
                                      // face.
                                      local_face_viscous_stress_tensor[i] =
                                        fluid_viscous_stress_at_ib;
                                      local_face_pressure_tensor[i] =
                                        fluid_pressure_stress_at_ib;


                                      force_eval[force_cache_key] =
                                        std::make_pair(
                                          fluid_viscous_stress_at_ib,
                                          fluid_pressure_stress_at_ib);
                                    }
                                }
                              else
                                {
                                  // Use the results from a previously evaluated
                                  // extrapolation. This step comes with an
                                  // error due to the curvature of the surface
                                  // in Q2 and higher degree elements.
                                  local_face_viscous_stress_tensor[i] =
                                    force_eval[force_cache_key].first;
                                  local_face_pressure_tensor[i] =
                                    force_eval[force_cache_key].second;
                                }
                            }
                          // Use the extrapolation of fluid stress tensor at the
                          // dof location of the IB surface cell to integrate
                          // the stress tensor on the surface of the IB
                          auto local_face_viscous_stress_tensor_old =
                            local_face_viscous_stress_tensor;
                          auto local_face_pressure_tensor_old =
                            local_face_pressure_tensor;

                          local_face_viscous_stress_tensor.clear();
                          local_face_pressure_tensor.clear();

                          local_face_viscous_stress_tensor.resize(
                            local_face_dof_indices.size());
                          local_face_pressure_tensor.resize(
                            local_face_dof_indices.size());

                          for (const auto &projection_cell_face :
                               local_face_dof_handler.active_cell_iterators())
                            {
                              fe_face_projection_values.reinit(
                                projection_cell_face);
                              std::vector<Point<dim>> q_points =
                                fe_face_projection_values
                                  .get_quadrature_points();
                              try
                                {
                                  if (this->simulation_parameters.fem_parameters
                                        .velocity_degree > 1)
                                    {
                                      FullMatrix<double> interpolation_matrix(
                                        local_face_dof_indices.size(),
                                        local_face_dof_indices.size());
                                      FullMatrix<double>
                                        inv_interpolation_matrix(
                                          local_face_dof_indices.size(),
                                          local_face_dof_indices.size());

                                      // Define the interpolation matrix of the
                                      // surface cell
                                      for (unsigned int i = 0;
                                           i < local_face_dof_indices.size();
                                           ++i)
                                        {
                                          Point<dim> point_projection;
                                          particles[p].closest_surface_point(
                                            support_points
                                              [local_face_dof_indices[i]],
                                            point_projection,
                                            cell);

                                          auto projected_point_unit =
                                            local_face_map
                                              .transform_real_to_unit_cell(
                                                projection_cell_face,
                                                point_projection);

                                          for (unsigned int j = 0;
                                               j <
                                               local_face_dof_indices.size();
                                               ++j)
                                            {
                                              interpolation_matrix[i][j] = 0;
                                              if (
                                                this->fe
                                                  ->face_system_to_component_index(
                                                    j)
                                                  .first ==
                                                this->fe
                                                  ->face_system_to_component_index(
                                                    i)
                                                  .first)
                                                interpolation_matrix[i][j] +=
                                                  local_face_fe.shape_value(
                                                    j, projected_point_unit);
                                            }
                                        }
                                      inv_interpolation_matrix.invert(
                                        interpolation_matrix);
                                      // Define the value of the fluid stress
                                      // tensor on the surface cell at the DOF
                                      // support points location.
                                      for (unsigned int i = 0;
                                           i < local_face_dof_indices.size();
                                           ++i)
                                        {
                                          for (unsigned int j = 0;
                                               j <
                                               local_face_dof_indices.size();
                                               ++j)
                                            {
                                              local_face_viscous_stress_tensor
                                                [i] +=
                                                inv_interpolation_matrix[i][j] *
                                                local_face_viscous_stress_tensor_old
                                                  [j];
                                              local_face_pressure_tensor[i] +=
                                                inv_interpolation_matrix[i][j] *
                                                local_face_pressure_tensor_old
                                                  [j];
                                            }
                                        }
                                    }
                                  else
                                    {
                                      local_face_viscous_stress_tensor =
                                        local_face_viscous_stress_tensor_old;
                                      local_face_pressure_tensor =
                                        local_face_pressure_tensor_old;
                                    }
                                }
                              catch (...)
                                {
                                  local_face_viscous_stress_tensor =
                                    local_face_viscous_stress_tensor_old;
                                  local_face_pressure_tensor =
                                    local_face_pressure_tensor_old;
                                }
                              for (unsigned int q = 0; q < n_q_points_face; q++)
                                {
                                  // Evaluate the total surface
                                  // Redefined the normal at the quadrature
                                  // point since we don't control the
                                  // orientation of the cell.
                                  normal_vector =
                                    fe_face_projection_values.normal_vector(q);
                                  if (scalar_product(
                                        normal_vector,
                                        approximate_surface_cell_normal) < 0)
                                    normal_vector = -normal_vector;

                                  fluid_viscous_stress = 0;
                                  fluid_pressure       = 0;

                                  // Integrate
                                  for (unsigned int i = 0;
                                       i < local_face_dof_indices.size();
                                       ++i)
                                    {
                                      const unsigned int component_i =
                                        local_face_fe
                                          .system_to_component_index(i)
                                          .first;
                                      if (component_i == 0)
                                        {
                                          fluid_viscous_stress +=
                                            fe_face_projection_values
                                              .shape_value(i, q) *
                                            local_face_viscous_stress_tensor[i];

                                          fluid_pressure +=
                                            fe_face_projection_values
                                              .shape_value(i, q) *
                                            local_face_pressure_tensor[i];
                                        }
                                    }

                                  auto viscous_force =
                                    fluid_viscous_stress * normal_vector *
                                    fe_face_projection_values.JxW(q);

                                  auto pressure_force =
                                    fluid_pressure * normal_vector *
                                    fe_face_projection_values.JxW(q);

                                  auto force = viscous_force + pressure_force;

                                  if (force != force)
                                    {
                                      // The force is nan; this happens when the
                                      // face projection is a line. This
                                      // generally happens when the surface on
                                      // which the force is evaluated is
                                      // perfectly flat and aligned with the
                                      // mesh. Since the area associated with
                                      // this face projection is zero, we set
                                      // the local force contribution to zero.
                                      force          = 0;
                                      viscous_force  = 0;
                                      pressure_force = 0;
                                    }

                                  // Add the local contribution of this surface
                                  // cell.

                                  particles[p].fluid_viscous_forces +=
                                    tensor_nd_to_3d(viscous_force);

                                  particles[p].fluid_pressure_forces +=
                                    tensor_nd_to_3d(pressure_force);

                                  auto distance =
                                    q_points[q] - particles[p].position;
                                  if constexpr (dim == 2)
                                    {
                                      particles[p].fluid_torque[0] += 0.;
                                      particles[p].fluid_torque[1] += 0.;
                                      particles[p].fluid_torque[2] +=
                                        distance[0] * force[1] -
                                        distance[1] * force[0];
                                    }
                                  else if constexpr (dim == 3)
                                    {
                                      particles[p].fluid_torque[0] +=
                                        distance[1] * force[2] -
                                        distance[2] * force[1];
                                      particles[p].fluid_torque[1] +=
                                        distance[2] * force[0] -
                                        distance[0] * force[2];
                                      particles[p].fluid_torque[2] +=
                                        distance[0] * force[1] -
                                        distance[1] * force[0];
                                    }
                                }
                            }
                        }
                    }

                  if (particles[p].position != saved_force_particle_pos)
                    particles[p].set_position(saved_force_particle_pos);
                }
            }
        }
    }

  // Sums the force evaluation on each of the processor.
  for (unsigned int i = 0; i < particles.size(); ++i)
    {
      particles[i].fluid_viscous_forces =
        Utilities::MPI::sum(particles[i].fluid_viscous_forces,
                            this->mpi_communicator) *
        fluid_density;
      particles[i].fluid_pressure_forces =
        Utilities::MPI::sum(particles[i].fluid_pressure_forces,
                            this->mpi_communicator) *
        fluid_density;
      particles[i].fluid_forces =
        particles[i].fluid_viscous_forces + particles[i].fluid_pressure_forces;
      particles[i].fluid_torque =
        Utilities::MPI::sum(particles[i].fluid_torque, this->mpi_communicator) *
        fluid_density;
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::write_force_ib()
{
  TimerOutput::Scope t(this->computing_timer,
                       "Output forces on immersed boundary");
  for (unsigned int p = 0; p < particles.size(); ++p)
    {
      {
        if (this->this_mpi_process == 0)
          {
            std::string filename =
              this->simulation_parameters.simulation_control.output_folder +
              this->simulation_parameters.particlesParameters
                ->ib_force_output_file +
              "." + Utilities::int_to_string(p, 2) + ".dat";
            std::ofstream output(filename.c_str());

            table_p[p].write_text(output);
          }
        MPI_Barrier(this->mpi_communicator);
      }
    }
}

template <int dim>
std::vector<OutputStruct<dim, GlobalVectorType>>
FluidDynamicsSharp<dim>::gather_output_hook()
{
  std::vector<OutputStruct<dim, GlobalVectorType>> solution_output_structs;
  // If the particles have moved, the combined shapes will have been cleared by
  // the particle integration routines.
  levelset_postprocessor =
    std::make_shared<LevelsetPostprocessor<dim>>(combined_shapes);
  solution_output_structs.emplace_back(
    std::in_place_type<OutputStructPostprocessor<dim, GlobalVectorType>>,
    *this->dof_handler,
    *this->present_solution,
    levelset_postprocessor);

  Vector<float> cell_cuts(this->triangulation->n_active_cells());
  Vector<float> cell_overconstrained(this->triangulation->n_active_cells());
  const unsigned int                   dofs_per_cell = this->fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  // If the enable extra verbose output is activated we add an output field to
  // the results where we identify which particle cuts each cell of the domain.
  if (this->simulation_parameters.particlesParameters
        ->enable_extra_sharp_interface_vtu_output_field)
    {
      // Define cell iterator
      const auto  &cell_iterator = this->dof_handler->active_cell_iterators();
      unsigned int i             = 0;

      for (const auto &cell : cell_iterator)
        {
          if (cell->is_locally_owned())
            {
              // Here we identify all cut cells, and if a cell is cut we output
              // the ID of the cell that cuts it. If the cell is not cut, a
              // default value of -1 is output.
              const auto &cut_cell_info = cut_cells_map[cell];
              bool        cell_is_cut   = cut_cell_info.is_cut;
              int         particle_id   = cut_cell_info.primary_particle_id;
              if (cell_is_cut)
                cell_cuts(i) = particle_id;
              else
                cell_cuts(i) = -1;

              // Here we identify all overconstrained cells. If a cell is
              // overconstrained, we output the ID of the closest particle (the
              // one used for extrapolation of the velocity immersed boundary
              // condition). If the cell is not overconstrained, a default value
              // of -1 is output.
              const auto &overconstrained_cell_info =
                overconstrained_fluid_cell_map[cell];
              bool cell_is_overconstrained =
                overconstrained_cell_info.is_overconstrained;
              particle_id = overconstrained_cell_info.particle_id;
              if (cell_is_overconstrained)
                cell_overconstrained(i) = particle_id;
              else
                cell_overconstrained(i) = -1;
            }
          i += 1;
        }
      solution_output_structs.emplace_back(
        std::in_place_type<OutputStructCellVector>, cell_cuts, "cell_cut");
      solution_output_structs.emplace_back(
        std::in_place_type<OutputStructCellVector>,
        cell_overconstrained,
        "overconstrained");

      levelset_gradient_postprocessor =
        std::make_shared<LevelsetGradientPostprocessor<dim>>(combined_shapes);
      solution_output_structs.emplace_back(
        std::in_place_type<OutputStructPostprocessor<dim, GlobalVectorType>>,
        *this->dof_handler,
        *this->present_solution,
        levelset_gradient_postprocessor);
    }
  return solution_output_structs;
}

template <int dim>
void
FluidDynamicsSharp<dim>::postprocess_fd(bool firstIter)
{
  bool enable =
    this->simulation_parameters.analytical_solution->calculate_error();
  this->simulation_parameters.analytical_solution->set_enable(false);
  NavierStokesBase<dim, GlobalVectorType, IndexSet>::postprocess_fd(firstIter);
  this->simulation_parameters.analytical_solution->set_enable(enable);
  // Calculate the error with respect to the analytical solution
  if (!firstIter &&
      this->simulation_parameters.analytical_solution->calculate_error())
    {
      // Update the time of the exact solution to the actual time
      this->exact_solution->set_time(
        this->simulation_control->get_current_time());
      const std::pair<double, double> error =
        this->calculate_L2_error_particles();

      if (this->simulation_parameters.simulation_control.method ==
          Parameters::SimulationControl::TimeSteppingMethod::steady)
        {
          this->error_table.add_value(
            "cells", this->triangulation->n_global_active_cells());
          this->error_table.add_value("error_velocity", error.first);
          this->error_table.add_value("error_pressure", error.second);

          auto summary = this->computing_timer.get_summary_data(
            this->computing_timer.total_wall_time);
          double total_time = 0;
          for (auto it = summary.begin(); it != summary.end(); ++it)
            {
              total_time += summary[it->first];
            }
          this->error_table.add_value("total_time", total_time);
        }
      else
        {
          this->error_table.add_value(
            "time", this->simulation_control->get_current_time());
          this->error_table.add_value("error_velocity", error.first);
          this->error_table.add_value("error_pressure", error.second);

          if (this->simulation_parameters.timer.write_time_in_error_table)
            {
              auto summary = this->computing_timer.get_summary_data(
                this->computing_timer.total_wall_time);
              double total_time = 0;
              for (auto it = summary.begin(); it != summary.end(); ++it)
                {
                  total_time += summary[it->first];
                }
              this->error_table.add_value("total_time", total_time);
            }
        }
      if (this->simulation_parameters.analytical_solution->verbosity ==
          Parameters::Verbosity::verbose)
        {
          this->pcout << "L2 error velocity : " << error.first
                      << " L2 error pressure: " << error.second << std::endl;
        }
    }
}


template <int dim>
std::pair<double, double>
FluidDynamicsSharp<dim>::calculate_L2_error_particles()
{
  TimerOutput::Scope t(this->computing_timer, "error");
  QGauss<dim>        quadrature_formula(this->number_quadrature_points + 1);
  FEValues<dim>      fe_values(*this->mapping,
                          *this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim>  fe_face_values(*this->mapping,
                                   *this->fe,
                                   *this->face_quadrature,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  const unsigned int dofs_per_cell =
    this->fe->dofs_per_cell; // This gives you DOFs per cell
  std::vector<types::global_dof_index> local_dof_indices(
    dofs_per_cell); //  Local connectivity

  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_q_points_face = this->face_quadrature->size();

  std::vector<Vector<double>> q_exactSol(n_q_points, Vector<double>(dim + 1));

  std::vector<Tensor<1, dim>> local_velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> local_face_velocity_values(n_q_points_face);
  std::vector<double>         local_pressure_values(n_q_points);
  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);

  Function<dim> *l_exact_solution = this->exact_solution;

  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(*this->mapping, *this->dof_handler);

  double l2errorU                  = 0.;
  double l2errorU_boundary         = 0.;
  double l2errorP                  = 0.;
  double total_velocity_divergence = 0.;
  double pressure_integral         = 0;
  double exact_pressure_integral   = 0;
  double volume                    = 0;

  // loop over elements
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          ->begin_active(),
                                                 endc =
                                                   this->dof_handler->end();

  // loop over elements to calculate average pressure
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          const bool cell_is_cut    = cut_cells_map[cell].is_cut;
          const bool cell_is_inside = cells_inside_map[cell].is_inside;
          const bool cell_is_overconstrained =
            overconstrained_fluid_cell_map[cell].is_overconstrained;

          if ((!cell_is_cut && !cell_is_inside) && !cell_is_overconstrained)
            {
              auto &evaluation_point = this->evaluation_point;
              fe_values.reinit(cell);

              fe_values[pressure].get_function_values(evaluation_point,
                                                      local_pressure_values);
              // Get the exact solution at all gauss points
              l_exact_solution->vector_value_list(
                fe_values.get_quadrature_points(), q_exactSol);


              // Retrieve the effective "connectivity matrix" for this element
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int q = 0; q < n_q_points; q++)
                {
                  pressure_integral +=
                    local_pressure_values[q] * fe_values.JxW(q);
                  exact_pressure_integral +=
                    q_exactSol[q][dim] * fe_values.JxW(q);
                  volume += fe_values.JxW(q);
                }
            }
        }
    }

  pressure_integral =
    Utilities::MPI::sum(pressure_integral, this->mpi_communicator);
  exact_pressure_integral =
    Utilities::MPI::sum(exact_pressure_integral, this->mpi_communicator);
  volume = Utilities::MPI::sum(volume, this->mpi_communicator);


  double average_pressure       = pressure_integral / volume;
  double average_exact_pressure = exact_pressure_integral / volume;
  cell = this->dof_handler->begin_active(), endc = this->dof_handler->end();

  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          const bool cell_is_cut = cut_cells_map[cell].is_cut;

          const bool cell_is_overconstrained =
            overconstrained_fluid_cell_map[cell].is_overconstrained;

          const bool cell_is_inside = cells_inside_map[cell].is_inside;
          if (cell->at_boundary() &&
              this->check_existance_of_bc(
                BoundaryConditions::BoundaryType::function_weak))
            {
              for (const auto face : cell->face_indices())
                {
                  if (cell->face(face)->at_boundary())
                    {
                      unsigned int boundary_id =
                        cell->face(face)->boundary_id();
                      if (this->simulation_parameters.boundary_conditions.type
                            .at(boundary_id) ==
                          BoundaryConditions::BoundaryType::function_weak)
                        {
                          NavierStokesFunctionDefined<dim> function_v(
                            &this->simulation_parameters.boundary_conditions
                               .navier_stokes_functions.at(boundary_id)
                               ->u,
                            &this->simulation_parameters.boundary_conditions
                               .navier_stokes_functions.at(boundary_id)
                               ->v,
                            &this->simulation_parameters.boundary_conditions
                               .navier_stokes_functions.at(boundary_id)
                               ->w);

                          fe_face_values.reinit(cell, face);
                          fe_face_values[velocities].get_function_values(
                            *this->present_solution,
                            local_face_velocity_values);
                          for (unsigned int q = 0; q < n_q_points_face; q++)
                            {
                              double u_x = local_face_velocity_values[q][0];
                              double u_y = local_face_velocity_values[q][1];

                              double u_x_a = function_v.value(
                                fe_face_values.quadrature_point(q), 0);
                              double u_y_a = function_v.value(
                                fe_face_values.quadrature_point(q), 1);

                              l2errorU_boundary +=
                                ((u_x - u_x_a) * (u_x - u_x_a) +
                                 (u_y - u_y_a) * (u_y - u_y_a)) *
                                fe_face_values.JxW(q);
                              if constexpr (dim == 3)
                                {
                                  double u_z = local_face_velocity_values[q][2];
                                  double u_z_a = function_v.value(
                                    fe_face_values.quadrature_point(q), 2);
                                  l2errorU_boundary += (u_z - u_z_a) *
                                                       (u_z - u_z_a) *
                                                       fe_face_values.JxW(q);
                                }
                            }
                        }
                    }
                }
            }

          if (!cell_is_cut && !cell_is_overconstrained)
            {
              auto &evaluation_point = this->evaluation_point;
              auto &present_solution = *this->present_solution;
              fe_values.reinit(cell);
              fe_values[velocities].get_function_values(present_solution,
                                                        local_velocity_values);
              fe_values[pressure].get_function_values(present_solution,
                                                      local_pressure_values);
              fe_values[velocities].get_function_gradients(
                evaluation_point, present_velocity_gradients);


              // Retrieve the effective "connectivity matrix" for this element
              cell->get_dof_indices(local_dof_indices);

              // Get the exact solution at all gauss points
              l_exact_solution->vector_value_list(
                fe_values.get_quadrature_points(), q_exactSol);
              for (unsigned int q = 0; q < n_q_points; q++)
                {
                  double present_velocity_divergence =
                    trace(present_velocity_gradients[q]);
                  double mass_source =
                    this->simulation_parameters.source_term
                      .navier_stokes_source->value(
                        fe_values.get_quadrature_points()[q], dim);


                  // Find the values of x and u_h (the finite element solution)
                  // at the quadrature points
                  double ux_sim   = local_velocity_values[q][0];
                  double ux_exact = q_exactSol[q][0];

                  double uy_sim   = local_velocity_values[q][1];
                  double uy_exact = q_exactSol[q][1];
                  l2errorU += (ux_sim - ux_exact) * (ux_sim - ux_exact) *
                              fe_values.JxW(q);
                  l2errorU += (uy_sim - uy_exact) * (uy_sim - uy_exact) *
                              fe_values.JxW(q);

                  if constexpr (dim == 3)
                    {
                      double uz_sim   = local_velocity_values[q][2];
                      double uz_exact = q_exactSol[q][2];
                      l2errorU += (uz_sim - uz_exact) * (uz_sim - uz_exact) *
                                  fe_values.JxW(q);
                    }
                  if (!cell_is_inside)
                    {
                      total_velocity_divergence +=
                        (present_velocity_divergence - mass_source) *
                        fe_values.JxW(q);
                      double p_sim =
                        local_pressure_values[q] - average_pressure;
                      double p_exact =
                        q_exactSol[q][dim] - average_exact_pressure;
                      l2errorP += (p_sim - p_exact) * (p_sim - p_exact) *
                                  fe_values.JxW(q);
                    }
                }
            }
        }
    }
  l2errorU = Utilities::MPI::sum(l2errorU, this->mpi_communicator);
  l2errorU_boundary =
    Utilities::MPI::sum(l2errorU_boundary, this->mpi_communicator);
  l2errorP = Utilities::MPI::sum(l2errorP, this->mpi_communicator);
  total_velocity_divergence =
    Utilities::MPI::sum(total_velocity_divergence, this->mpi_communicator);

  this->pcout << "div u : " << total_velocity_divergence << std::endl;
  if (this->check_existance_of_bc(
        BoundaryConditions::BoundaryType::function_weak))
    {
      this->pcout << "error at the weak BC : " << std::sqrt(l2errorU_boundary)
                  << std::endl;
    }

  return std::make_pair(std::sqrt(l2errorU), std::sqrt(l2errorP));
}


template <int dim>
void
FluidDynamicsSharp<dim>::apply_prescribed_particle_motion(
  IBParticle<dim> &particle,
  const double     time,
  const double     dt) const
{
  if (this->simulation_parameters.particlesParameters
        ->load_particles_from_file == false)
    {
      particle.f_position->set_time(time);
      particle.f_velocity->set_time(time);
      particle.f_omega->set_time(time);
      particle.f_orientation->set_time(time);

      particle.position[0] = particle.f_position->value(particle.position, 0);
      particle.position[1] = particle.f_position->value(particle.position, 1);
      particle.velocity[0] = particle.f_velocity->value(particle.position, 0);
      particle.velocity[1] = particle.f_velocity->value(particle.position, 1);
      particle.omega[0]    = particle.f_omega->value(particle.position, 0);
      particle.omega[1]    = particle.f_omega->value(particle.position, 1);
      particle.omega[2]    = particle.f_omega->value(particle.position, 2);
      particle.orientation[0] =
        particle.f_orientation->value(particle.position, 0);
      particle.orientation[1] =
        particle.f_orientation->value(particle.position, 1);
      particle.orientation[2] =
        particle.f_orientation->value(particle.position, 2);
      if constexpr (dim == 3)
        {
          particle.position[2] =
            particle.f_position->value(particle.position, 2);
          particle.velocity[2] =
            particle.f_velocity->value(particle.position, 2);
        }
    }
  else
    {
      // The velocity and angular velocity of a particle loaded from a file are
      // constant. The new state is computed from the state at the start of the
      // time step, so that it does not change when this function is called at
      // each Newton iteration.
      for (int d = 0; d < dim; ++d)
        particle.position[d] =
          particle.previous_positions[0][d] + particle.velocity[d] * dt;
      particle.orientation =
        particle.previous_orientation[0] + particle.omega * dt;
    }
}


template <int dim>
void
FluidDynamicsSharp<dim>::integrate_particles()
{
  // Undo the previous trial's wrap before recomputing motion. Accepted BDF
  // history stays in its original frame until finish_time_step_particles().
  if (sharp_ib_periodic_boundaries.is_periodic_enabled())
    {
      particle_position_wrap.resize(particles.size());
      for (unsigned int p = 0; p < particles.size(); ++p)
        if (particle_position_wrap[p].norm_square() != 0.)
          particles[p].set_position(particles[p].position -
                                    particle_position_wrap[p]);
      std::fill(particle_position_wrap.begin(),
                particle_position_wrap.end(),
                Tensor<1, dim>());
    }

  particle_residual = 0;

  TimerOutput::Scope t(this->computing_timer, "Integrate particles");
  // Integrate the velocity of the particle. If integrate motion is set to
  // true in the parameter this function will also integrate the force to update
  // the velocity. Otherwise, the velocity is kept constant

  // To integrate the forces and update the velocity, this function uses the
  // implicit Euler algorithm. To find the force at t+dt the function use the
  // fix point algorithm in parallel to the newton iteration used for the fluid
  // resolution.
  using dealii::numbers::PI;
  double dt    = this->simulation_control->get_time_steps_vector()[0];
  double time  = this->simulation_control->get_current_time();
  double alpha = this->simulation_parameters.particlesParameters->alpha;
  this->simulation_parameters.particlesParameters->f_gravity->set_time(time);
  double dr = GridTools::minimal_cell_diameter(*this->triangulation) /
              std::numbers::sqrt2;

  const auto rheological_model =
    this->simulation_parameters.physical_properties_manager.get_rheology();

  // Explicit coupling reuses the first DEM trial on subsequent fluid
  // iterations. Only roll back spring history when DEM will run again.
  const bool recompute_contacts =
    some_particles_are_coupled && time > 0 &&
    (this->get_current_newton_iteration() == 0 ||
     (!this->simulation_parameters.particlesParameters
         ->explicit_contact_impulsion_calculation &&
      !this->simulation_parameters.particlesParameters
         ->explicit_position_integration_calculation));
  ib_dem.update_particles(particles, time - dt, recompute_contacts);

  std::map<field, double> field_values;
  field_values[field::shear_rate]  = 1;
  field_values[field::temperature] = 1;

  // Check if the parameters' combination is compatible. This is temporary and
  // will be moved to a new class that tests the parameter combination in the
  // parameter initialization.

  AssertThrow(!(this->simulation_parameters.physical_properties_manager
                  .is_non_newtonian() and
                this->simulation_parameters.particlesParameters
                  ->enable_lubrication_force and
                some_particles_are_coupled),
              RequiresConstantViscosity(
                "FluidDynamicsSharp<dim>::integrate_particles"));

  double h_min =
    dr * this->simulation_parameters.particlesParameters->lubrication_range_min;
  double h_max =
    dr * this->simulation_parameters.particlesParameters->lubrication_range_max;
  // Deactivated the lubrication force if the fluid is Non-newtonian.
  if (this->simulation_parameters.particlesParameters
        ->enable_lubrication_force == false)
    h_max = 0;

  particle_residual = 0;
  if (some_particles_are_coupled && time > 0)
    {
      AssertThrow(this->simulation_parameters.physical_properties_manager
                    .density_is_constant(),
                  RequiresConstantDensity(
                    "FluidDynamicsSharp<dim>::integrate_particles"));
      auto density_model =
        this->simulation_parameters.physical_properties_manager.get_density(0);
      double fluid_density = density_model->value(field_values);
      double kinematic_viscosity =
        rheological_model->value(field_values) * fluid_density;

      Vector<double> particles_residual_vect;
      particles_residual_vect.reinit(particles.size());
      Tensor<1, 3> g;
      this->simulation_parameters.particlesParameters->f_gravity->set_time(
        time);
      if (recompute_contacts)
        {
          ib_dem.integrate_particles_motion(
            dt, h_max, h_min, fluid_density, kinematic_viscosity);
        }

      unsigned int worst_residual_particle_id = UINT_MAX;

      bool clear_combined_shape_cache = false;

      for (unsigned int p = 0; p < particles.size(); ++p)
        {
          if (particles[p].integrate_motion)
            {
              // If the motion of a single particle is to be integrated, the
              // combined cache should be cleared after the integration.
              // Otherwise, the visualization of the level set will be wrong.
              clear_combined_shape_cache = true;

              // calculate the volume of fluid displaced by the particle.
              double volume = particles[p].volume;

              g[0] = this->simulation_parameters.particlesParameters->f_gravity
                       ->value(particles[p].position, 0);
              g[1] = this->simulation_parameters.particlesParameters->f_gravity
                       ->value(particles[p].position, 1);
              if constexpr (dim == 3)
                g[2] = this->simulation_parameters.particlesParameters
                         ->f_gravity->value(particles[p].position, 2);

              Tensor<1, 3> gravity_buoyancy_force =
                g * (particles[p].mass / volume - fluid_density) * volume;

              // Transfers the impulsion evaluated in the sub-time-stepping
              // scheme to the particle at the CFD timescale.
              particles[p].impulsion = ib_dem.dem_particles[p].impulsion;
              particles[p].omega_impulsion =
                ib_dem.dem_particles[p].omega_impulsion;
              particles[p].contact_impulsion =
                ib_dem.dem_particles[p].contact_impulsion;
              particles[p].omega_contact_impulsion =
                ib_dem.dem_particles[p].omega_contact_impulsion;
              // Time stepping information
              const auto method =
                this->simulation_control->get_assembly_method();
              std::vector<double> time_steps_vector =
                this->simulation_control->get_time_steps_vector();

              // Vector for the BDF coefficients
              const Vector<double> &bdf_coefs =
                this->simulation_control->get_bdf_coefficients();

              // Define the residual of the particle dynamics.
              Tensor<1, 3> residual_velocity =
                -(bdf_coefs[0] * particles[p].velocity);

              for (unsigned int i = 1;
                   i < number_of_previous_solutions(method) + 1;
                   ++i)
                {
                  residual_velocity +=
                    -(bdf_coefs[i] * particles[p].previous_velocity[i - 1]);
                }
              residual_velocity +=
                (particles[p].contact_impulsion / dt +
                 particles[p].fluid_forces + gravity_buoyancy_force) /
                particles[p].mass;

              // SHARP-TEMPORARY-DEBUG-TOP
              const auto debug_fluid_acceleration =
                particles[p].fluid_forces / particles[p].mass;
              const auto debug_contact_acceleration =
                particles[p].contact_impulsion / (dt * particles[p].mass);
              const auto debug_gravity_acceleration =
                gravity_buoyancy_force / particles[p].mass;
              debug_previous_fluid_acceleration.resize(particles.size());
              debug_previous_contact_acceleration.resize(particles.size());
              debug_previous_gravity_acceleration.resize(particles.size());
              if (this->get_current_newton_iteration() > 0)
                {
                  const auto dv =
                    particles[p].velocity - particles[p].velocity_iter;
                  const auto dr =
                    residual_velocity - particles[p].previous_velocity_residual;
                  const auto df = debug_fluid_acceleration -
                                  debug_previous_fluid_acceleration[p];
                  const auto dc = debug_contact_acceleration -
                                  debug_previous_contact_acceleration[p];
                  const auto dg = debug_gravity_acceleration -
                                  debug_previous_gravity_acceleration[p];
                  const double       dv2 = dv.norm_square();
                  std::ostringstream out;
                  out << std::scientific << std::setprecision(12)
                      << "[SHARP-TEMP] secant_budget newton="
                      << this->get_current_newton_iteration()
                      << " particle=" << p << " bdf0=" << bdf_coefs[0]
                      << " delta_fluid_acceleration=" << df
                      << " delta_contact_acceleration=" << dc
                      << " delta_gravity_acceleration=" << dg
                      << " closure_error="
                      << (dr + bdf_coefs[0] * dv - df - dc - dg).norm()
                      << " dv_squared=" << dv2 << " fluid_projected_slope="
                      << (dv2 > 0. ? scalar_product(dv, df) / dv2 : 0.)
                      << " contact_projected_slope="
                      << (dv2 > 0. ? scalar_product(dv, dc) / dv2 : 0.);
                  this->pcout << out.str() << std::endl;
                }
              debug_previous_fluid_acceleration[p] = debug_fluid_acceleration;
              debug_previous_contact_acceleration[p] =
                debug_contact_acceleration;
              debug_previous_gravity_acceleration[p] =
                debug_gravity_acceleration;
              // SHARP-TEMPORARY-DEBUG-BOTTOM

              double inverse_of_relaxation_coefficient_velocity =
                -bdf_coefs[0] -
                0.5 * volume * fluid_density / particles[p].mass / dt + DBL_MIN;

              // Evaluate the relaxation parameter using a generalization of the
              // secant method.
              if (particles[p].previous_velocity_residual.norm() != 0)
                {
                  auto vector_of_velocity_variation =
                    (particles[p].velocity - particles[p].velocity_iter);
                  auto vector_of_residual_variation =
                    residual_velocity - particles[p].previous_velocity_residual;
                  double dot_product_of_the_variation_vectors =
                    scalar_product(vector_of_velocity_variation,
                                   vector_of_residual_variation);
                  inverse_of_relaxation_coefficient_velocity =
                    1 / (dot_product_of_the_variation_vectors /
                           (vector_of_residual_variation.norm_square()) +
                         DBL_MIN);
                }
              // SHARP-TEMPORARY-DEBUG-TOP
              if (debug_velocity_inverse.size() != particles.size())
                debug_velocity_inverse.resize(particles.size());
              debug_velocity_inverse[p] =
                inverse_of_relaxation_coefficient_velocity;
              // SHARP-TEMPORARY-DEBUG-BOTTOM
              // Relaxation parameter for the particle dynamics.
              double local_alpha = 1.;
              // We keep in memory the state of the particle before the update
              // in case something went wrong and the particle is now outside of
              // the domain. If this is the case, the particle won't be updated.
              IBParticle<dim> save_particle_state         = particles[p];
              bool            save_particle_state_is_used = false;
              // Define the correction vector.
              Tensor<1, 3> velocity_correction_vector;
              // Define the correction vector. And check if it is not NAN. NAN
              // happen when the residual tends to zero in cases where the fluid
              // density is 0.
              if (inverse_of_relaxation_coefficient_velocity ==
                  inverse_of_relaxation_coefficient_velocity)
                {
                  velocity_correction_vector =
                    residual_velocity * 1. /
                    inverse_of_relaxation_coefficient_velocity;
                }
              else
                velocity_correction_vector = residual_velocity;

              // Update the particle state and keep in memory the last iteration
              // information.
              particles[p].velocity_iter = particles[p].velocity;
              particles[p].velocity =
                particles[p].velocity_iter -
                velocity_correction_vector * alpha * local_alpha;
              particles[p].impulsion_iter             = particles[p].impulsion;
              particles[p].previous_velocity_residual = residual_velocity;

              // If the particles have impacted a wall or another particle, we
              // want to use the sub-time step position. Otherwise, we solve the
              // new position directly with the new velocity found.
              if (particles[p].contact_impulsion.norm() < 1e-12 and
                  (this->simulation_parameters.particlesParameters
                     ->explicit_position_integration_calculation == false))
                {
                  particles[p].position.clear();
                  for (int d = 0; d < dim; ++d)
                    {
                      for (unsigned int i = 1;
                           i < number_of_previous_solutions(method) + 1;
                           ++i)
                        {
                          double position_update =
                            -bdf_coefs[i] *
                            particles[p].previous_positions[i - 1][d] /
                            bdf_coefs[0];

                          particles[p].set_position(particles[p].position[d] +
                                                      position_update,
                                                    d);
                        }

                      double position_update =
                        particles[p].velocity[d] / bdf_coefs[0];
                      particles[p].set_position(particles[p].position[d] +
                                                  position_update,
                                                d);
                    }
                }
              else
                {
                  particles[p].position = ib_dem.dem_particles[p].position;
                  particles[p].set_position(particles[p].position);
                }

              // SHARP-TEMPORARY-DEBUG-TOP
              Tensor<1, dim> debug_position_equation;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  debug_position_equation[d] =
                    bdf_coefs[0] * particles[p].position[d] -
                    particles[p].velocity[d];
                  for (unsigned int i = 1;
                       i <= number_of_previous_solutions(method);
                       ++i)
                    debug_position_equation[d] +=
                      bdf_coefs[i] * particles[p].previous_positions[i - 1][d];
                }
              std::ostringstream debug_consistency;
              debug_consistency
                << std::scientific << std::setprecision(12)
                << "[SHARP-TEMP] particle_consistency newton="
                << this->get_current_newton_iteration() << " particle=" << p
                << " position_bdf_velocity_defect=" << debug_position_equation
                << " dem_vs_corrected_velocity="
                << (ib_dem.dem_particles[p].velocity - particles[p].velocity)
                << " frozen_force_residual_after="
                << (residual_velocity -
                    bdf_coefs[0] *
                      (particles[p].velocity - save_particle_state.velocity))
                     .norm()
                << " actual_residual_before=" << residual_velocity.norm();
              this->pcout << debug_consistency.str() << std::endl;
              // SHARP-TEMPORARY-DEBUG-BOTTOM

              // Test the normalized geometry without changing the integration
              // frame. All motion modes are normalized together below.
              const Point<dim> domain_position =
                sharp_ib_periodic_boundaries.wrap_point(particles[p].position,
                                                        periodic_domain_lower,
                                                        periodic_domain_upper);

              // Check if the particle is in the domain. Throw an error if it's
              // the case.
              try
                {
                  const auto &cell =
                    LetheGridTools::find_cell_around_point_with_tree(
                      *this->dof_handler,
                      domain_position,
                      sharp_ib_periodic_boundaries.is_periodic_enabled());
                  (void)cell;
                }
              catch (...)
                {
                  this->pcout
                    << "particle " << p
                    << " is now outside the domain we do not update this particle. Position: "
                    << particles[p].position[0] << ", "
                    << particles[p].position[1];
                  if constexpr (dim == 3)
                    this->pcout << ", " << particles[p].position[2];
                  this->pcout << std::endl;
                  save_particle_state_is_used = true;
                }

              // For the rotation velocity : same logic as the velocity.
              auto inv_inertia = invert(particles[p].inertia);

              // Calculate the torque in the particle frame of reference.
              Tensor<1, 3> total_torque =
                particles[p].rotation_matrix *
                (particles[p].omega_contact_impulsion / dt +
                 particles[p].fluid_torque);

              // Calculate angular acceleration in particle frame
              Tensor<1, 3> angular_velocity_in_particle_frame =
                particles[p].rotation_matrix * particles[p].omega;
              Tensor<1, 3> angular_acceleration_in_particle_frame;
              angular_acceleration_in_particle_frame[0] =
                (total_torque[0] -
                 (particles[p].inertia[2][2] - particles[p].inertia[1][1]) *
                   angular_velocity_in_particle_frame[2] *
                   angular_velocity_in_particle_frame[1]) /
                particles[p].inertia[0][0];
              angular_acceleration_in_particle_frame[1] =
                (total_torque[1] -
                 (particles[p].inertia[0][0] - particles[p].inertia[2][2]) *
                   angular_velocity_in_particle_frame[2] *
                   angular_velocity_in_particle_frame[0]) /
                particles[p].inertia[1][1];
              angular_acceleration_in_particle_frame[2] =
                (total_torque[2] -
                 (particles[p].inertia[1][1] - particles[p].inertia[0][0]) *
                   angular_velocity_in_particle_frame[0] *
                   angular_velocity_in_particle_frame[1]) /
                particles[p].inertia[2][2];

              // Evaluate the residual
              Tensor<1, 3> residual_omega =
                -(bdf_coefs[0] * particles[p].omega);
              for (unsigned int i = 1;
                   i < number_of_previous_solutions(method) + 1;
                   ++i)
                {
                  residual_omega +=
                    -(bdf_coefs[i] * particles[p].previous_omega[i - 1]);
                }
              // Rotate angular acceleration in world frame
              residual_omega += // omega_accel
                invert(particles[p].rotation_matrix) *
                angular_acceleration_in_particle_frame;


              double inverse_of_relaxation_coefficient_omega =
                -bdf_coefs[0] - 0.5 * 2. / 5 * volume * fluid_density *
                                  particles[p].radius * particles[p].radius *
                                  inv_inertia.norm() / dt;
              // Evaluate the relaxation parameter using a generalization of the
              // secant method.
              if (particles[p].previous_omega_residual.norm() != 0)
                {
                  auto vector_of_omega_variation =
                    (particles[p].omega - particles[p].omega_iter);
                  auto vector_of_residual_variation =
                    residual_omega - particles[p].previous_omega_residual;

                  double dot_product_of_the_variation_vectors =
                    scalar_product(vector_of_omega_variation,
                                   vector_of_residual_variation);
                  inverse_of_relaxation_coefficient_omega =
                    1 / (dot_product_of_the_variation_vectors /
                           (vector_of_residual_variation.norm_square()) +
                         DBL_MIN);
                }
              // SHARP-TEMPORARY-DEBUG-TOP
              if (debug_omega_inverse.size() != particles.size())
                debug_omega_inverse.resize(particles.size());
              debug_omega_inverse[p] = inverse_of_relaxation_coefficient_omega;
              // SHARP-TEMPORARY-DEBUG-BOTTOM
              // Define the correction vector. And check if it is not NAN. NAN
              // happen when the residual tends to zero in cases where the fluid
              // density is 0.
              Tensor<1, 3> omega_correction_vector;
              if (inverse_of_relaxation_coefficient_omega ==
                  inverse_of_relaxation_coefficient_omega)
                {
                  omega_correction_vector =
                    residual_omega * 1 /
                    inverse_of_relaxation_coefficient_omega;
                }
              else
                {
                  omega_correction_vector = residual_omega;
                }

              // SHARP-TEMPORARY-DEBUG-TOP
              const auto debug_dv =
                particles[p].velocity_iter - save_particle_state.velocity_iter;
              const auto debug_dr =
                residual_velocity -
                save_particle_state.previous_velocity_residual;
              const auto debug_dw =
                particles[p].omega - particles[p].omega_iter;
              const auto debug_drw =
                residual_omega - particles[p].previous_omega_residual;
              std::ostringstream debug_particle;
              debug_particle
                << std::scientific << std::setprecision(10)
                << "[SHARP-TEMP] particle_secant step="
                << this->simulation_control->get_iteration_number()
                << " newton=" << this->get_current_newton_iteration()
                << " particle=" << p
                << " velocity_before=" << save_particle_state.velocity
                << " velocity_after=" << particles[p].velocity
                << " velocity_residual=" << residual_velocity
                << " delta_velocity=" << debug_dv
                << " delta_residual=" << debug_dr
                << " secant_dot=" << scalar_product(debug_dv, debug_dr)
                << " delta_residual_squared=" << debug_dr.norm_square()
                << " inverse_velocity="
                << inverse_of_relaxation_coefficient_velocity
                << " omega_before=" << particles[p].omega
                << " omega_residual=" << residual_omega
                << " delta_omega=" << debug_dw
                << " delta_omega_residual=" << debug_drw
                << " omega_secant_dot=" << scalar_product(debug_dw, debug_drw)
                << " inverse_omega=" << inverse_of_relaxation_coefficient_omega
                << " omega_correction=" << omega_correction_vector * alpha
                << " fluid_force=" << particles[p].fluid_forces
                << " contact_lubrication_average="
                << particles[p].contact_impulsion / dt
                << " gravity_buoyancy=" << gravity_buoyancy_force
                << " mass=" << particles[p].mass << " position_from_DEM="
                << !(particles[p].contact_impulsion.norm() < 1e-12 &&
                     !this->simulation_parameters.particlesParameters
                        ->explicit_position_integration_calculation);
              this->pcout << debug_particle.str() << std::endl;
              // SHARP-TEMPORARY-DEBUG-BOTTOM
              double local_alpha_omega = 1;

              particles[p].omega_iter = particles[p].omega;
              particles[p].omega =
                particles[p].omega_iter -
                omega_correction_vector * alpha * local_alpha_omega;
              particles[p].omega_impulsion_iter = particles[p].omega_impulsion;
              particles[p].previous_omega_residual = residual_omega;

              // If the particles have impacted a wall or another particle, we
              // want to use the sub-time step position. Otherwise, we solve the
              // new position directly with the new velocity found.
              if (particles[p].omega_contact_impulsion.norm() < 1e-12 and
                  (this->simulation_parameters.particlesParameters
                     ->explicit_position_integration_calculation == false))
                {
                  if (particles[p].omega.norm() > 0)
                    {
                      // Impose a new orientation of the particle based on the
                      // integration and previous orientation.
                      particles[p].rotation_matrix       = 0;
                      particles[p].rotation_matrix[0][0] = 1.0;
                      particles[p].rotation_matrix[1][1] = 1.0;
                      particles[p].rotation_matrix[2][2] = 1.0;
                      if constexpr (dim == 2)
                        {
                          Tensor<1, 3> axis;
                          axis[2]                      = 1.0;
                          particles[p].rotation_matrix = Physics::
                            Transformations::Rotations::rotation_matrix_3d(
                              axis, particles[p].previous_orientation[0][2]);
                        }
                      if constexpr (dim == 3)
                        {
                          for (unsigned int i = 0; i < 3; ++i)
                            {
                              Tensor<1, 3> axis;
                              axis[2 - i] = 1.0;
                              particles[p].rotation_matrix =
                                Physics::Transformations::Rotations::
                                  rotation_matrix_3d(
                                    axis,
                                    particles[p]
                                      .previous_orientation[0][2 - i]) *
                                particles[p].rotation_matrix;
                            }
                        }
                      particles[p].rotation_matrix =
                        Physics::Transformations::Rotations::rotation_matrix_3d(
                          particles[p].omega / particles[p].omega.norm(),
                          particles[p].omega.norm() * dt) *
                        particles[p].rotation_matrix;
                      particles[p].orientation =
                        particles[p].shape->rotation_matrix_to_xyz_angles(
                          particles[p].rotation_matrix);
                    }
                }
              else
                {
                  // We have contact so we impose the DEM orientation as the new
                  // orientation.
                  particles[p].orientation =
                    ib_dem.dem_particles[p].orientation;
                  particles[p].set_orientation(particles[p].orientation);
                }
              // If something went wrong during the update, the particle's state
              // would be reversed to its original state here.
              if (save_particle_state_is_used)
                particles[p] = save_particle_state;

              // Evaluate global residual of the particle dynamics.
              double this_particle_residual =
                sqrt(std::pow(residual_velocity.norm(), 2) +
                     std::pow(residual_omega.norm() * particles[p].radius, 2)) *
                dt;

              // Keep in memory the residual.
              particles_residual_vect[p] = this_particle_residual;
              // L_inf of all the particles' residual.
              if (this_particle_residual > particle_residual)
                {
                  particle_residual          = this_particle_residual;
                  worst_residual_particle_id = p;
                }
              particles[p].set_position(particles[p].position);
              particles[p].set_orientation(particles[p].orientation);
            }
          else
            {
              apply_prescribed_particle_motion(particles[p], time, dt);
              if (particles[p].position != particles[p].previous_positions[0] ||
                  particles[p].orientation !=
                    particles[p].previous_orientation[0])
                {
                  particles[p].clear_shape_cache();
                  // One particle has moved significantly: the combined shape
                  // cache should be cleared for visualization purposes.
                  clear_combined_shape_cache = true;
                }

              particles[p].set_position(particles[p].position);
              particles[p].set_orientation(particles[p].orientation);
            }
        }

      if (clear_combined_shape_cache)
        combined_shapes->clear_cache();

      if (this->simulation_parameters.physics_solving_strategy
            .at(PhysicsID::fluid_dynamics)
            .verbosity != Parameters::Verbosity::quiet)
        {
          this->pcout << "L_inf particle residual : " << particle_residual
                      << " particle id : " << worst_residual_particle_id
                      << std::endl;
          this->pcout << "L2 particle residual L2 "
                      << particles_residual_vect.l2_norm() << std::endl;
        }
    }
  else
    {
      // Direct application of the function for the velocity and position if the
      // particle dynamics is not integrated.
      for (unsigned int p = 0; p < particles.size(); ++p)
        {
          apply_prescribed_particle_motion(particles[p], time, dt);
          if (particles[p].position != particles[p].previous_positions[0] ||
              particles[p].orientation != particles[p].previous_orientation[0])
            particles[p].clear_shape_cache();

          particles[p].set_position(particles[p].position);
          particles[p].set_orientation(particles[p].orientation);
        }
      particle_residual = 0;
    }
  if (sharp_ib_periodic_boundaries.is_periodic_enabled())
    for (unsigned int p = 0; p < particles.size(); ++p)
      {
        const Point<dim> wrapped =
          sharp_ib_periodic_boundaries.wrap_point(particles[p].position,
                                                  periodic_domain_lower,
                                                  periodic_domain_upper);
        particle_position_wrap[p] = wrapped - particles[p].position;
        if (particle_position_wrap[p].norm_square() != 0.)
          particles[p].set_position(wrapped);
      }
}



template <int dim>
void
FluidDynamicsSharp<dim>::Visualization_IB::build_patches(
  std::vector<IBParticle<dim>> particles)
{
  properties_to_write = particles[0].get_properties_name();
  // A list of field names for all data components stored in patches.
  vector_datasets.clear();
  dataset_names.clear();
  // Defining property field position
  int field_position = 0;
  // Iterating over properties
  for (const auto &properties_iterator : properties_to_write)
    {
      // Get the property field name
      const std::string field_name = properties_iterator.first;

      // Number of components of the corresponding property
      const unsigned components_number = properties_iterator.second;

      // Check to see if the property is a vector
      if (components_number == 3)
        {
          vector_datasets.emplace_back(
            field_position,
            field_position + components_number - 1,
            field_name,
            DataComponentInterpretation::component_is_part_of_vector);
        }
      dataset_names.push_back(field_name);
      ++field_position;
    }

  // Building the patch data
  patches.resize(particles.size());

  // Looping over particle to get the properties from the particle_handler
  for (unsigned int p = 0; p < particles.size(); ++p)
    {
      // Particle location
      patches[p].vertices[0] = particles[p].position;
      patches[p].patch_index = p;
      patches[p].data.reinit(particles[p].get_number_properties(), 1);

      // ID and other properties

      // Calculating force for visualization
      auto particle_properties = particles[p].get_properties();

      for (unsigned int property_index = 0;
           property_index < particles[p].get_number_properties();
           ++property_index)
        patches[p].data(property_index, 0) =
          particle_properties[property_index];
    }
}

template <int dim>
const std::vector<DataOutBase::Patch<0, dim>> &
FluidDynamicsSharp<dim>::Visualization_IB::get_patches() const
{
  return patches;
}

template <int dim>
std::vector<std::string>
FluidDynamicsSharp<dim>::Visualization_IB::get_dataset_names() const
{
  return dataset_names;
}

template <int dim>
std::vector<
  std::tuple<unsigned int,
             unsigned int,
             std::string,
             DataComponentInterpretation::DataComponentInterpretation>>
FluidDynamicsSharp<dim>::Visualization_IB::get_nonscalar_data_ranges() const
{
  return vector_datasets;
}

template <int dim>
FluidDynamicsSharp<dim>::Visualization_IB::~Visualization_IB() = default;

template <int dim>
void
FluidDynamicsSharp<dim>::finish_time_step_particles()
{
  // Store information about the particle used for the integration and print the
  // results if requested.

  table_all_p.clear();
  for (unsigned int p = 0; p < particles.size(); ++p)
    {
      // Apply the accepted translation uniformly before inserting the new
      // position. Older history entries may legitimately span many periods.
      if (sharp_ib_periodic_boundaries.is_periodic_enabled() &&
          p < particle_position_wrap.size())
        {
          sharp_ib_periodic_boundaries.translate_history(
            particles[p].previous_positions, particle_position_wrap[p]);
          particle_position_wrap[p].clear();
        }
      for (unsigned int i = particles[p].previous_velocity.size() - 1; i > 0;
           --i)
        {
          particles[p].previous_positions[i] =
            particles[p].previous_positions[i - 1];
          particles[p].previous_velocity[i] =
            particles[p].previous_velocity[i - 1];
          particles[p].previous_orientation[i] =
            particles[p].previous_orientation[i - 1];
          particles[p].previous_omega[i] = particles[p].previous_omega[i - 1];
        }

      particles[p].previous_positions[0]   = particles[p].position;
      particles[p].previous_velocity[0]    = particles[p].velocity;
      particles[p].previous_omega[0]       = particles[p].omega;
      particles[p].previous_orientation[0] = particles[p].orientation;

      particles[p].previous_fluid_forces = particles[p].fluid_forces;
      particles[p].previous_fluid_viscous_forces =
        particles[p].fluid_viscous_forces;
      particles[p].previous_fluid_pressure_forces =
        particles[p].fluid_pressure_forces;
      particles[p].previous_fluid_torque = particles[p].fluid_torque;

      particles[p].velocity_iter              = particles[p].velocity;
      particles[p].impulsion_iter             = particles[p].impulsion;
      particles[p].omega_iter                 = particles[p].omega;
      particles[p].omega_impulsion_iter       = particles[p].omega_impulsion;
      particles[p].previous_velocity_residual = 0;
      particles[p].previous_omega_residual    = 0;

      if (some_particles_are_coupled &&
          this->simulation_parameters.particlesParameters->print_dem)
        {
          this->pcout << "particle " << p << " position "
                      << particles[p].position << std::endl;
          if constexpr (dim == 2)
            {
              this->pcout << "particle " << p << " velocity "
                          << tensor_nd_to_2d(particles[p].velocity)
                          << std::endl;
            }
          else
            {
              this->pcout << "particle " << p << " velocity "
                          << particles[p].velocity << std::endl;
            }
        }
      table_p[p].add_value("particle_ID", p);
      table_all_p.add_value("particle_ID", p);
      if (this->simulation_parameters.simulation_control.method !=
          Parameters::SimulationControl::TimeSteppingMethod::steady)
        table_p[p].add_value("time",
                             this->simulation_control->get_current_time());
      table_all_p.add_value("time",
                            this->simulation_control->get_current_time());
      if constexpr (dim == 3)
        {
          table_p[p].add_value("T_x", particles[p].fluid_torque[0]);
          table_p[p].set_precision(
            "T_x",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.add_value("T_x", particles[p].fluid_torque[0]);
          table_all_p.set_precision(
            "T_x",
            this->simulation_parameters.simulation_control.log_precision);
          if (some_particles_are_coupled)
            {
              table_p[p].add_value("omega_x", particles[p].omega[0]);
              table_p[p].set_precision(
                "omega_x",
                this->simulation_parameters.simulation_control.log_precision);
              table_p[p].add_value("theta_x", particles[p].orientation[0]);
              table_p[p].set_precision(
                "theta_x",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.add_value("omega_x", particles[p].omega[0]);
              table_all_p.set_precision(
                "omega_x",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.add_value("theta_x", particles[p].orientation[0]);
              table_all_p.set_precision(
                "theta_x",
                this->simulation_parameters.simulation_control.log_precision);
            }

          table_p[p].add_value("T_y", particles[p].fluid_torque[1]);
          table_p[p].set_precision(
            "T_y",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.add_value("T_y", particles[p].fluid_torque[1]);
          table_all_p.set_precision(
            "T_y",
            this->simulation_parameters.simulation_control.log_precision);
          if (some_particles_are_coupled)
            {
              table_p[p].add_value("omega_y", particles[p].omega[1]);
              table_p[p].set_precision(
                "omega_y",
                this->simulation_parameters.simulation_control.log_precision);
              table_p[p].add_value("theta_y", particles[p].orientation[1]);
              table_p[p].set_precision(
                "theta_y",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.add_value("omega_y", particles[p].omega[1]);
              table_all_p.set_precision(
                "omega_y",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.add_value("theta_y", particles[p].orientation[1]);
              table_all_p.set_precision(
                "theta_y",
                this->simulation_parameters.simulation_control.log_precision);
            }
        }

      table_p[p].add_value("T_z", particles[p].fluid_torque[2]);
      table_p[p].set_precision(
        "T_z", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.add_value("T_z", particles[p].fluid_torque[2]);
      table_all_p.set_precision(
        "T_z", this->simulation_parameters.simulation_control.log_precision);
      if (some_particles_are_coupled)
        {
          table_p[p].add_value("omega_z", particles[p].omega[2]);
          table_p[p].set_precision(
            "omega_z",
            this->simulation_parameters.simulation_control.log_precision);
          table_p[p].add_value("theta_z", particles[p].orientation[2]);
          table_p[p].set_precision(
            "theta_z",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.add_value("omega_z", particles[p].omega[2]);
          table_all_p.set_precision(
            "omega_z",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.add_value("theta_z", particles[p].orientation[2]);
          table_all_p.set_precision(
            "theta_z",
            this->simulation_parameters.simulation_control.log_precision);
        }

      table_p[p].add_value("f_x", particles[p].fluid_forces[0]);
      table_all_p.add_value("f_x", particles[p].fluid_forces[0]);
      table_p[p].set_precision(
        "f_x", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.set_precision(
        "f_x", this->simulation_parameters.simulation_control.log_precision);
      if (some_particles_are_coupled)
        {
          table_p[p].add_value("v_x", particles[p].velocity[0]);
          table_p[p].add_value("p_x", particles[p].position[0]);
          table_all_p.add_value("v_x", particles[p].velocity[0]);
          table_all_p.add_value("p_x", particles[p].position[0]);
          table_p[p].set_precision(
            "v_x",
            this->simulation_parameters.simulation_control.log_precision);

          table_p[p].set_precision(
            "p_x",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "v_x",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "p_x",
            this->simulation_parameters.simulation_control.log_precision);
        }

      table_p[p].add_value("f_y", particles[p].fluid_forces[1]);
      table_p[p].set_precision(
        "f_y", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.add_value("f_y", particles[p].fluid_forces[1]);
      table_all_p.set_precision(
        "f_y", this->simulation_parameters.simulation_control.log_precision);
      if (some_particles_are_coupled)
        {
          table_p[p].add_value("v_y", particles[p].velocity[1]);
          table_p[p].add_value("p_y", particles[p].position[1]);
          table_all_p.add_value("v_y", particles[p].velocity[1]);
          table_all_p.add_value("p_y", particles[p].position[1]);
          table_p[p].set_precision(
            "v_y",
            this->simulation_parameters.simulation_control.log_precision);
          table_p[p].set_precision(
            "p_y",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "v_y",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "p_y",
            this->simulation_parameters.simulation_control.log_precision);
        }

      if constexpr (dim == 3)
        {
          table_p[p].add_value("f_z", particles[p].fluid_forces[2]);
          table_p[p].set_precision(
            "f_z",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.add_value("f_z", particles[p].fluid_forces[2]);
          table_all_p.set_precision(
            "f_z",
            this->simulation_parameters.simulation_control.log_precision);
          if (some_particles_are_coupled)
            {
              table_p[p].add_value("v_z", particles[p].velocity[2]);
              table_p[p].add_value("p_z", particles[p].position[2]);
              table_all_p.add_value("v_z", particles[p].velocity[2]);
              table_all_p.add_value("p_z", particles[p].position[2]);
              table_p[p].set_precision(
                "v_z",
                this->simulation_parameters.simulation_control.log_precision);
              table_p[p].set_precision(
                "p_z",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.set_precision(
                "v_z",
                this->simulation_parameters.simulation_control.log_precision);
              table_all_p.set_precision(
                "p_z",
                this->simulation_parameters.simulation_control.log_precision);
            }
        }

      table_p[p].add_value("f_xv", particles[p].fluid_viscous_forces[0]);
      table_all_p.add_value("f_xv", particles[p].fluid_viscous_forces[0]);
      table_p[p].add_value("f_yv", particles[p].fluid_viscous_forces[1]);
      table_all_p.add_value("f_yv", particles[p].fluid_viscous_forces[1]);

      table_p[p].set_precision(
        "f_xv", this->simulation_parameters.simulation_control.log_precision);
      table_p[p].set_precision(
        "f_yv", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.set_precision(
        "f_xv", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.set_precision(
        "f_yv", this->simulation_parameters.simulation_control.log_precision);
      if constexpr (dim == 3)
        {
          table_p[p].add_value("f_zv", particles[p].fluid_viscous_forces[2]);
          table_all_p.add_value("f_zv", particles[p].fluid_viscous_forces[2]);
          table_p[p].set_precision(
            "f_zv",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "f_zv",
            this->simulation_parameters.simulation_control.log_precision);
        }

      table_p[p].add_value("f_xp", particles[p].fluid_pressure_forces[0]);
      table_all_p.add_value("f_xp", particles[p].fluid_pressure_forces[0]);
      table_p[p].add_value("f_yp", particles[p].fluid_pressure_forces[1]);
      table_all_p.add_value("f_yp", particles[p].fluid_pressure_forces[1]);
      table_p[p].set_precision(
        "f_xp", this->simulation_parameters.simulation_control.log_precision);
      table_p[p].set_precision(
        "f_yp", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.set_precision(
        "f_xp", this->simulation_parameters.simulation_control.log_precision);
      table_all_p.set_precision(
        "f_yp", this->simulation_parameters.simulation_control.log_precision);
      if constexpr (dim == 3)
        {
          table_p[p].add_value("f_zp", particles[p].fluid_pressure_forces[2]);
          table_all_p.add_value("f_zp", particles[p].fluid_pressure_forces[2]);
          table_p[p].set_precision(
            "f_zp",
            this->simulation_parameters.simulation_control.log_precision);
          table_all_p.set_precision(
            "f_zp",
            this->simulation_parameters.simulation_control.log_precision);
        }
    }
  if (this->this_mpi_process == 0)
    {
      if (this->simulation_parameters.forces_parameters.verbosity ==
          Parameters::Verbosity::verbose)
        {
          std::cout << "+------------------------------------------+"
                    << std::endl;
          std::cout << "|  Force summary particles  "
                    << "               |" << std::endl;
          std::cout << "+------------------------------------------+"
                    << std::endl;
          table_all_p.write_text(std::cout);
        }
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::fill_particle_properties(
  const IBParticle<dim> &particle,
  std::vector<double>   &properties) const
{
  using PropertiesIndex = DEM::CFDDEMProperties::PropertiesIndex;

  properties.assign(PropertiesIndex::n_properties, 0.);

  properties[PropertiesIndex::type] = 0.;
  properties[PropertiesIndex::dp]   = 2.0 * particle.radius;
  properties[PropertiesIndex::mass] = particle.mass;

  properties[PropertiesIndex::v_x]     = particle.velocity[0];
  properties[PropertiesIndex::v_y]     = particle.velocity[1];
  properties[PropertiesIndex::omega_x] = particle.omega[0];
  properties[PropertiesIndex::omega_y] = particle.omega[1];
  properties[PropertiesIndex::omega_z] = particle.omega[2];

  properties[PropertiesIndex::fem_force_two_way_coupling_x] =
    particle.fluid_forces[0];
  properties[PropertiesIndex::fem_force_two_way_coupling_y] =
    particle.fluid_forces[1];
  properties[PropertiesIndex::fem_torque_x] = particle.fluid_torque[0];
  properties[PropertiesIndex::fem_torque_y] = particle.fluid_torque[1];
  properties[PropertiesIndex::fem_torque_z] = particle.fluid_torque[2];

  if constexpr (dim == 2)
    {
      properties[PropertiesIndex::v_z]                          = 0.;
      properties[PropertiesIndex::fem_force_two_way_coupling_z] = 0.;
    }
  else
    {
      properties[PropertiesIndex::v_z] = particle.velocity[2];
      properties[PropertiesIndex::fem_force_two_way_coupling_z] =
        particle.fluid_forces[2];
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::build_ib_particle_handler(
  Particles::ParticleHandler<dim, dim> &particle_handler,
  typename DEM::dem_data_structures<dim>::particle_index_iterator_map
                                   &particle_container,
  std::unordered_set<unsigned int> &local_particle_ids) const
{
  particle_handler.clear();
  particle_container.clear();
  local_particle_ids.clear();

  Point<dim>          ref_point;
  std::vector<double> properties;

  // Rebuild a DEM particle handler from the Sharp-IB particles so that DEM
  // output and post-processing can reuse the same code path.
  for (const auto &particle : particles)
    {
      typename DoFHandler<dim>::active_cell_iterator cell;
      try
        {
          cell = LetheGridTools::find_cell_around_point_with_tree(
            *this->dof_handler,
            particle.position,
            sharp_ib_periodic_boundaries.is_periodic_enabled());
        }
      catch (...)
        {
          continue;
        }

      if (!cell->is_locally_owned())
        continue;

      fill_particle_properties(particle, properties);
      particle_handler.insert_particle(
        particle.position, ref_point, particle.particle_id, cell, properties);
      local_particle_ids.insert(particle.particle_id);
    }

  particle_handler.sort_particles_into_subdomains_and_cells();
  particle_handler.exchange_ghost_particles(true);

  update_particle_container(particle_container, &particle_handler);
}

template <int dim>
void
FluidDynamicsSharp<dim>::build_contact_containers(
  const typename DEM::dem_data_structures<dim>::particle_index_iterator_map
                                         &particle_container,
  const std::unordered_set<unsigned int> &local_particle_ids,
  typename DEM::dem_data_structures<dim>::adjacent_particle_pairs
    &local_adjacent_particles,
  typename DEM::dem_data_structures<dim>::adjacent_particle_pairs
    &ghost_adjacent_particles) const
{
  local_adjacent_particles.clear();
  ghost_adjacent_particles.clear();

  // Convert the Sharp-IB contact map into the DEM containers expected by the
  // existing DEM post-processing utilities.
  const auto &pp_contact_map = ib_dem.get_particle_particle_contact_map();
  for (const auto &[particle_id, contacts] : pp_contact_map)
    {
      // After ghost exchange, a contact near a subdomain boundary can be
      // visible on more than one rank. To avoid duplicate DEM
      // post-processing entries, each rank only contributes contacts whose
      // first particle is locally owned here.
      if (local_particle_ids.find(particle_id) == local_particle_ids.end())
        continue;

      const auto particle_one_it = particle_container.find(particle_id);
      if (particle_one_it == particle_container.end())
        continue;

      for (const auto &[particle_two_id, contact_info] : contacts)
        {
          const auto particle_two_it = particle_container.find(particle_two_id);
          if (particle_two_it == particle_container.end())
            continue;

          particle_particle_contact_info<dim> dem_contact_info;
          dem_contact_info.particle_two = particle_two_it->second;
          dem_contact_info.tangential_displacement =
            contact_info.tangential_displacement;
          dem_contact_info.rolling_resistance_spring_torque =
            contact_info.rolling_resistance_spring_torque;

          // Keep the same convention as for DEM, contacts between two owned
          // particles stay in the local container, while contacts involving a
          // ghost neighbor go in the ghost container. The iterator to particle
          // one is shared by all its contacts and stored once in the adjacency
          // value.
          if (local_particle_ids.find(particle_two_id) !=
              local_particle_ids.end())
            {
              auto &adjacency        = local_adjacent_particles[particle_id];
              adjacency.particle_one = particle_one_it->second;
              adjacency.second_particles[particle_two_id] = dem_contact_info;
            }
          else
            {
              auto &adjacency        = ghost_adjacent_particles[particle_id];
              adjacency.particle_one = particle_one_it->second;
              adjacency.second_particles[particle_two_id] = dem_contact_info;
            }
        }
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::handle_dem_particle_output_and_postprocessing()
{
  const auto &dem_parameters = cfd_dem_parameters.dem_parameters;

  // Mirror DEM output cadence for Sharp-IB particles and any DEM-style
  // post-processing derived from them.
  const bool output_iteration = this->simulation_control->is_output_iteration();
  const bool write_particle_output = output_iteration;
  const bool write_force_chains =
    output_iteration && dem_parameters.post_processing.force_chains;

  if (!(write_particle_output || write_force_chains))
    return;

  Particles::ParticleHandler<dim, dim> particle_handler(
    *this->triangulation,
    *this->mapping,
    DEM::CFDDEMProperties::PropertiesIndex::n_properties);
  typename DEM::dem_data_structures<dim>::particle_index_iterator_map
                                   particle_container;
  std::unordered_set<unsigned int> local_particle_ids;

  build_ib_particle_handler(particle_handler,
                            particle_container,
                            local_particle_ids);

  typename DEM::dem_data_structures<dim>::adjacent_particle_pairs
    local_adjacent_particles;
  typename DEM::dem_data_structures<dim>::adjacent_particle_pairs
    ghost_adjacent_particles;

  if (write_force_chains)
    build_contact_containers(particle_container,
                             local_particle_ids,
                             local_adjacent_particles,
                             ghost_adjacent_particles);

  if (write_particle_output)
    {
      const std::string folder =
        this->simulation_parameters.simulation_control.output_folder;
      const std::string particles_solution_name =
        this->simulation_parameters.particlesParameters->ib_particles_pvd_file;
      const unsigned int iter =
        this->simulation_control->get_iteration_number();
      const double       time = this->simulation_control->get_current_time();
      const unsigned int group_files =
        this->simulation_parameters.simulation_control.group_files;

      Visualization<dim, DEM::CFDDEMProperties::PropertiesIndex>
        particle_data_out;
      particle_data_out.build_patches(
        particle_handler,
        DEM::ParticleProperties<dim, DEM::CFDDEMProperties::PropertiesIndex>::
          get_properties_name());

      write_vtu_and_pvd<0, dim>(ib_particles_pvdhandler,
                                particle_data_out,
                                folder,
                                particles_solution_name,
                                time,
                                iter,
                                group_files,
                                this->mpi_communicator);

      if (write_force_chains)
        {
          auto particles_force_chains_object =
            set_force_chains_contact_force_model<
              dim,
              DEM::CFDDEMProperties::PropertiesIndex>(dem_parameters);
          particles_force_chains_object->write_force_chains(
            dem_parameters,
            ib_particles_pvdhandler_force_chains,
            this->mpi_communicator,
            dem_parameters.simulation_control.output_folder,
            iter,
            time,
            local_adjacent_particles,
            ghost_adjacent_particles);
        }
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::sharp_edge()
{
  // SHARP-TEMPORARY-DEBUG-TOP
  // Optional experiment: synchronize any outstanding additive RHS writes
  // collectively before Sharp replaces selected entries.
  if (std::getenv("SHARP_DEBUG_SYNC_RHS") != nullptr)
    this->system_rhs.compress(VectorOperation::add);
  const auto debug_previous_pressure_reference = debug_pressure_reference;
  debug_pressure_reference.clear();
  const auto debug_previous_rows = debug_row_kind;
  debug_row_kind.clear();
  debug_stencil_counts.fill(0.);
  // SHARP-TEMPORARY-DEBUG-BOTTOM

  // This function defines an Immersed Boundary based on the sharp edge method
  // on a solid of dim=2 or dim=3

  TimerOutput::Scope t(this->computing_timer, "Assemble sharp");
  // SHARP-TEMPORARY-DEBUG-TOP
  try
    {
  // SHARP-TEMPORARY-DEBUG-BOTTOM
  using dealii::numbers::PI;
  std::vector<typename DoFHandler<dim>::active_cell_iterator> active_neighbors;
  std::vector<typename DoFHandler<dim>::active_cell_iterator>
    active_neighbors_set;
  std::vector<typename DoFHandler<dim>::active_cell_iterator>
                                   active_neighbors_2;
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<double> time_steps_vector =
    this->simulation_control->get_time_steps_vector();
  // Define a map to all DOFs and their support points
  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(*this->mapping, *this->dof_handler);

  // Initalize fe value objects in order to do calculation with it later
  QGauss<dim>        q_formula(this->number_quadrature_points);
  FEValues<dim>      fe_values(*this->fe,
                          q_formula,
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;

  const unsigned int degree =
    this->simulation_parameters.particlesParameters->stencil_degree;
  double length_ratio =
    this->simulation_parameters.particlesParameters->length_ratio;

  IBStencil<dim>      stencil;
  std::vector<double> ib_coef = stencil.coefficients(degree, length_ratio);

  unsigned int n_q_points = q_formula.size();

  // Define multiple local_dof_indices one for the cell iterator one for the
  // cell with the second point for the sharp edge stencil and one for
  // manipulation on the neighbour’s cell.
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_2(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_3(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_4(dofs_per_cell);
  std::set<unsigned int>               clear_line;

  // Define minimal cell length
  double dr = GridTools::minimal_cell_diameter(*this->triangulation) /
              std::numbers::sqrt2;

  // Define cell iterator
  const auto &cell_iterator = this->dof_handler->active_cell_iterators();
  double      dt            = time_steps_vector[0];
  if (Parameters::SimulationControl::TimeSteppingMethod::steady ==
      this->simulation_parameters.simulation_control.method)
    dt = 1;

  if (this->simulation_parameters.particlesParameters
        ->assemble_navier_stokes_inside == true)
    {
      // impose pressure reference in each of the particle
      for (unsigned int p = 0; p < particles.size(); ++p)
        {
          Point<dim> pressure_reference_location =
            particles[p].pressure_location + particles[p].position;

          if (sharp_ib_periodic_boundaries.is_periodic_enabled())
            {
              // Select the same interior pressure node on every rank. The
              // first owned vertex of the reference cell can lie outside the
              // particle, and changes when the mesh is repartitioned.
              // Order candidates by squared distance, then coordinates and
              // global DoF index to break ties consistently across ranks.
              using Candidate =
                std::pair<double,
                          std::pair<Point<dim>, types::global_dof_index>>;
              Candidate  best{std::numeric_limits<double>::max(),
                              {Point<dim>(), numbers::invalid_dof_index}};
              const auto less = [](const Candidate &a, const Candidate &b) {
                if (a.first != b.first)
                  return a.first < b.first;
                for (int d = 0; d < dim; ++d)
                  if (a.second.first[d] != b.second.first[d])
                    return a.second.first[d] < b.second.first[d];
                return a.second.second < b.second.second;
              };
              const auto pressure_frames = get_periodic_particle_shifts(
                particles[p].position, particle_support_radius(particles[p]));
              for (const auto &candidate_cell : cell_iterator)
                if (candidate_cell->is_locally_owned())
                  {
                    candidate_cell->get_dof_indices(local_dof_indices);
                    for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                      {
                        const auto index = local_dof_indices[i];
                        if (this->fe->system_to_component_index(i).first !=
                              dim ||
                            !this->locally_owned_dofs.is_element(index) ||
                            this->zero_constraints.is_constrained(index) ||
                            this->nonzero_constraints.is_constrained(index))
                          continue;
                        for (const auto &shift : pressure_frames)
                          {
                            const Point<dim> point =
                              support_points.at(index) - shift;
                            if (particles[p].get_levelset(point) >= 0.0)
                              continue;
                            Candidate candidate{point.distance_square(
                                                  pressure_reference_location),
                                                {support_points.at(index),
                                                 index}};
                            if (less(candidate, best))
                              best = candidate;
                          }
                      }
                  }
              const auto candidates =
                Utilities::MPI::all_gather(this->mpi_communicator, best);
              for (const auto &candidate : candidates)
                if (less(candidate, best))
                  best = candidate;
              AssertThrow(best.second.second != numbers::invalid_dof_index,
                          ExcMessage(
                            "No unconstrained pressure node lies inside "
                            "the periodic particle. Refine the mesh."));
              const auto index = best.second.second;
              if (this->locally_owned_dofs.is_element(index))
                {
                  const double scale = std::pow(dr, dim) / dt;
                  this->system_matrix.clear_row(index);
                  this->system_matrix.add(index, index, scale);
                  this->system_rhs(index) =
                    -this->evaluation_point(index) * scale;
                }
              continue;
            }


          const auto &cell = LetheGridTools::find_cell_around_point_with_tree(
            *this->dof_handler,
            pressure_reference_location,
            sharp_ib_periodic_boundaries.is_periodic_enabled());

          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);
              double sum_line = 0;
              double volume   = 0;
              fe_values.reinit(cell);
              std::vector<int> set_pressure_cell;
              set_pressure_cell.resize(particles.size());

              // Define the order of magnitude for the stencil.
              for (unsigned int qf = 0; qf < n_q_points; ++qf)
                volume += fe_values.JxW(qf);

              sum_line = volume / dt;
              // Clear the line in the matrix
              unsigned int inside_index = local_dof_indices[dim];
              // Check on which DOF of the cell to impose the pressure. If the
              // dof is on a hanging node, it is already constrained and the
              // pressure cannot be imposed there. So we just go to the next
              // pressure DOF of the cell.

              for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                {
                  const int component_i =
                    this->fe->system_to_component_index(i).first;
                  if (this->zero_constraints.is_constrained(
                        local_dof_indices[i]) == false &&
                      this->locally_owned_dofs.is_element(
                        local_dof_indices[i]) &&
                      component_i == dim)
                    {
                      inside_index = local_dof_indices[i];
                      break;
                    }
                }
              this->system_matrix.clear_row(inside_index);
              // this->system_matrix.clear_row(inside_index)
              // is not reliable on edge case

              // Set the new equation for the first pressure DOFs of the
              // cell. this is the new reference pressure inside a
              // particle. As for the other pressure DOFs, the rhs is divided
              // by the pressure_scaling_factor since the pressure DOFs are
              // rescaled in the newton correction.
              this->system_matrix.add(inside_index, inside_index, sum_line);
              this->system_rhs(inside_index) =
                -this->evaluation_point(inside_index) * sum_line /
                this->simulation_parameters.stabilization
                  .pressure_scaling_factor;
            }
        }
    }

  ib_done.clear();
  // Loop on all the cell to define if the sharp edge cut them
  for (const auto &cell_cut : cell_iterator)
    {
      if (cell_cut->is_locally_owned() || cell_cut->is_ghost())
        {
          // Check if the cell_cut is cut or not by the IB and what the particle
          // the cut the cell_cut. If the particle is cut
          const auto &cut_cell_info = cut_cells_map[cell_cut];
          const auto &overconstrained_cell_info =
            overconstrained_fluid_cell_map[cell_cut];
          bool cell_is_cut = cut_cell_info.is_cut;
          bool cell_is_overconstrained =
            overconstrained_cell_info.is_overconstrained;
          unsigned int ib_particle_id =
            cell_is_overconstrained ? overconstrained_cell_info.particle_id :
                                      cut_cell_info.primary_particle_id;
          std::vector<unsigned int> cutting_particle_ids;
          cutting_particle_ids.reserve(
            cut_cell_info.cutting_particle_frames.size());
          for (const auto &particle_frame :
               cut_cell_info.cutting_particle_frames)
            {
              if (std::ranges::none_of(cutting_particle_ids,
                                       [&particle_frame](
                                         const unsigned int particle_id) {
                                         return particle_id ==
                                                particle_frame.particle_id;
                                       }))
                {
                  cutting_particle_ids.push_back(particle_frame.particle_id);
                }
            }

          if (cell_is_cut || cell_is_overconstrained)
            {
              // Reuse the periodic frame stored during cut-cell
              // classification instead of re-guessing the wrapped particle
              // image from the cell barycenter.
              const Point<dim> saved_particle_pos_for_periodic =
                particles[ib_particle_id].position;
              const Tensor<1, dim> periodic_shift_for_cell =
                cell_is_overconstrained ?
                  overconstrained_cell_info.periodic_shift :
                  get_periodic_shift_for_particle_frame(cut_cell_info,
                                                        ib_particle_id);
              const int periodic_frame_index =
                get_periodic_frame_index(periodic_shift_for_cell);
              set_particle_position_from_periodic_shift(
                ib_particle_id,
                saved_particle_pos_for_periodic,
                periodic_shift_for_cell);

              double sum_line = 0;
              fe_values.reinit(cell_cut);

              double volume = 0;
              // Define the order of magnitude for the stencil.
              for (unsigned int qf = 0; qf < n_q_points; ++qf)
                volume += fe_values.JxW(qf);

              sum_line = volume / dt;
              cell_cut->get_dof_indices(local_dof_indices);
              // If we are here, the cell_cut is cut by the IB.
              // Loops on the dof that represents the velocity  component
              // and pressure separately
              for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                {
                  unsigned int global_index_overwrite = local_dof_indices[i];
                  // SHARP-TEMPORARY-DEBUG-TOP
                  debug_row_kind.try_emplace(global_index_overwrite, 1);
                  // SHARP-TEMPORARY-DEBUG-BOTTOM

                  const unsigned int cache_particle_id =
                    periodic_frame_index != 0 ? ib_particle_id : 0;
                  const PeriodicFrameDofCacheKey stencil_cache_key = {
                    global_index_overwrite,
                    cache_particle_id,
                    periodic_frame_index};
                  if (ib_done.find(stencil_cache_key) == ib_done.end())
                    {
                      const int component_i =
                        this->fe->system_to_component_index(i).first;
                      bool dof_is_inside =
                        particles[ib_particle_id].get_levelset(
                          support_points[local_dof_indices[i]], cell_cut) <= 0;

                      // If multiple particles cut the cell_cut, we treat the
                      // dof of pressure as a dummy dof. We don't use them to
                      // set the boundary condition for the Poisson problem
                      // inside the particle.
                      bool use_ib_for_pressure =
                        (dof_is_inside) && (component_i == dim) &&
                        (this->simulation_parameters.particlesParameters
                           ->assemble_navier_stokes_inside == false) &&
                        cutting_particle_ids.size() < 2;

                      // Reapply the zero and nonzero constraints.
                      if ((this->zero_constraints.is_constrained(
                             local_dof_indices[i]) ||
                           this->nonzero_constraints.is_constrained(
                             local_dof_indices[i])) &&
                          this->locally_owned_dofs.is_element(
                            global_index_overwrite))
                        {
                          // SHARP-TEMPORARY-DEBUG-TOP
                          const auto *debug_entries =
                            this->nonzero_constraints.get_constraint_entries(
                              global_index_overwrite);
                          if (debug_entries == nullptr)
                            debug_entries =
                              this->zero_constraints.get_constraint_entries(
                                global_index_overwrite);
                          debug_row_kind[global_index_overwrite] =
                            debug_entries != nullptr &&
                                !debug_entries->empty() ?
                              5 :
                              6;
                          // SHARP-TEMPORARY-DEBUG-BOTTOM
                          if (this->zero_constraints.is_constrained(
                                local_dof_indices[i]))
                            {
                              // Clear the line if there is something on it
                              this->system_matrix.clear_row(
                                global_index_overwrite);
                              // Get the constraint equations
                              auto local_entries =
                                *this->zero_constraints.get_constraint_entries(
                                  local_dof_indices[i]);
                              double interpolation = 0;
                              // Write the equation
                              for (unsigned int j = 0; j < local_entries.size();
                                   ++j)
                                {
                                  unsigned int col = local_entries[j].first;
                                  double entries   = local_entries[j].second;

                                  interpolation +=
                                    this->evaluation_point(col) * entries;

                                  // Expand constrained columns explicitly so
                                  // Sharp-IB equations remain valid when the
                                  // column DoF is a periodic slave.
                                  add_matrix_entry_with_periodic_expansion(
                                    local_dof_indices[i],
                                    col,
                                    entries * sum_line);
                                }
                              this->system_matrix.add(local_dof_indices[i],
                                                      local_dof_indices[i],
                                                      sum_line);

                              // Write the RHS
                              this->system_rhs(local_dof_indices[i]) =
                                -this->evaluation_point(local_dof_indices[i]) *
                                  sum_line +
                                interpolation * sum_line +
                                this->zero_constraints.get_inhomogeneity(
                                  local_dof_indices[i]) *
                                  sum_line;
                            }
                          if (this->nonzero_constraints.is_constrained(
                                local_dof_indices[i]))
                            {
                              // Clear the line if there is something on it
                              this->system_matrix.clear_row(
                                global_index_overwrite);
                              // Get the constraint equations

                              auto local_entries_non_zero =
                                *this->nonzero_constraints
                                   .get_constraint_entries(
                                     local_dof_indices[i]);

                              double interpolation = 0;
                              // Write the equation
                              for (unsigned int j = 0;
                                   j < local_entries_non_zero.size();
                                   ++j)
                                {
                                  unsigned int col_non_zero =
                                    local_entries_non_zero[j].first;
                                  double entries_non_zero =
                                    local_entries_non_zero[j].second;

                                  interpolation +=
                                    this->evaluation_point(col_non_zero) *
                                    entries_non_zero;

                                  add_matrix_entry_with_periodic_expansion(
                                    local_dof_indices[i],
                                    col_non_zero,
                                    entries_non_zero * sum_line);
                                }
                              this->system_matrix.add(local_dof_indices[i],
                                                      local_dof_indices[i],
                                                      sum_line);

                              // Write the RHS
                              this->system_rhs(local_dof_indices[i]) =
                                -this->evaluation_point(local_dof_indices[i]) *
                                  sum_line +
                                interpolation * sum_line +
                                this->nonzero_constraints.get_inhomogeneity(
                                  local_dof_indices[i]) *
                                  sum_line;
                            }
                        }

                      // Check if the DOfs is owned and if it's not a hanging
                      // node.
                      if (((component_i < dim) || use_ib_for_pressure) &&
                          this->locally_owned_dofs.is_element(
                            global_index_overwrite) &&
                          (this->zero_constraints.is_constrained(
                             local_dof_indices[i]) ||
                           this->nonzero_constraints.is_constrained(
                             local_dof_indices[i])) == false)
                        {
                          // We are working on the DOFs of cells cut by a
                          // particle. We clear the current equation associated
                          // with the current DOF and replace it with the sharp
                          // IB constraints.

                          // Clear the current line of this dof
                          this->system_matrix.clear_row(global_index_overwrite);

                          // Define the points for the IB stencil based on the
                          // degree, particle position, and DOF position. The
                          // definition of the output variable "point" changes
                          // depending on the degree. In the case of stencil
                          // degrees 1 to 4, the variable point returns the
                          // position of the DOF directly. In the case of higher
                          // degree stencil (5 or more), it returns the position
                          // of the point that is on the IB. This is because
                          // stencil degrees higher than four are not
                          // implemented. The function extrapolates the element
                          // at the particle's surface in these cases. To do so,
                          // we use the point at the surface of the particle.
                          // The variable "interpolation points" return the
                          // points used to define the cell_cut used for the
                          // stencil definition and the locations of the points
                          // used in the stencil calculation.

                          auto [point, interpolation_points] =
                            stencil.support_points_for_interpolation(
                              degree,
                              length_ratio,
                              particles[ib_particle_id],
                              support_points[local_dof_indices[i]],
                              cell_cut);
                          // Find the cell_cut used for the stencil definition.
                          auto point_to_find_cell =
                            stencil.point_for_cell_detection(
                              particles[ib_particle_id],
                              support_points[local_dof_indices[i]],
                              cell_cut);
                          typename DoFHandler<dim>::active_cell_iterator
                                         stencil_cell;
                          bool           particle_close_to_wall = false;
                          Tensor<1, dim> stencil_periodic_shift;
                          (void)particle_close_to_wall;
                          if (sharp_ib_periodic_boundaries
                                .is_periodic_enabled())
                            stencil_cell = find_periodic_stencil_cell(
                              cell_cut,
                              point_to_find_cell,
                              stencil_periodic_shift);
                          else
                            {
                              try
                                {
                                  stencil_cell = LetheGridTools::
                                    find_cell_around_point_with_neighbors<dim>(
                                      *this->dof_handler,
                                      vertices_to_cell,
                                      cell_cut,
                                      point_to_find_cell);
                                }
                              catch (...)
                                {
                                  stencil_cell = this->dof_handler->end();
                                }
                            }
                          // SHARP-TEMPORARY-DEBUG-TOP
                          ++debug_stencil_counts[0];
                          if (stencil_periodic_shift.norm_square() > 0.)
                            ++debug_stencil_counts[1];
                          if (stencil_cell == this->dof_handler->end())
                            {
                              ++debug_stencil_counts[2];
                              if (std::getenv("SHARP_DEBUG_STENCIL_FALLBACK") !=
                                  nullptr)
                                std::cerr
                                  << "[SHARP-TEMP] stencil_fallback rank="
                                  << Utilities::MPI::this_mpi_process(
                                       this->mpi_communicator)
                                  << " row=" << global_index_overwrite
                                  << " particle=" << ib_particle_id
                                  << " particle_position="
                                  << particles[ib_particle_id].position
                                  << " support_point="
                                  << support_points[local_dof_indices[i]]
                                  << " search_point=" << point_to_find_cell
                                  << std::endl;
                            }
                          // SHARP-TEMPORARY-DEBUG-BOTTOM
                          if (stencil_cell == this->dof_handler->end())
                            {
                              // Preserve the nonperiodic wall fallback.
                              particle_close_to_wall = true;
                              stencil_cell           = cell_cut;
                              if (this->zero_constraints.is_constrained(
                                    global_index_overwrite) ||
                                  this->nonzero_constraints.is_constrained(
                                    global_index_overwrite))
                                continue;
                            }

                          stencil_cell->get_dof_indices(local_dof_indices_2);
                          ib_done[stencil_cache_key] = {stencil_cell,
                                                        stencil_periodic_shift};

                          bool skip_stencil = false;
                          // SHARP-TEMPORARY-DEBUG-TOP
                          debug_row_kind[global_index_overwrite] = 2;
                          // SHARP-TEMPORARY-DEBUG-BOTTOM


                          // Check if the DOF intersect the IB
                          bool dof_on_ib = false;

                          // Check if this dof is a dummy dof or directly on IB
                          // and Check if the point used to define the cell_cut
                          // used for the definition of the stencil
                          // ("stencil_cell") is on a face between the cell_cut
                          // that is cut ("cell_cut") and the "stencil_cell".
                          bool point_in_cell;
                          // The extrapolation can be disabled for debugging
                          // purposes, although in most cases it shouldn't since
                          // it is a core part of this solver
                          if (this->simulation_parameters.particlesParameters
                                ->enable_extrapolation)
                            point_in_cell = cell_cut->point_inside(
                              interpolation_points
                                [stencil.number_of_interpolation_support_points(
                                   degree) -
                                 1]);
                          else
                            point_in_cell = cell_cut->point_inside(
                              support_points[local_dof_indices[i]]);

                          bool       dof_is_dummy = false;
                          const bool cell2_is_cut =
                            cut_cells_map[stencil_cell].is_cut;
                          bool cell2_is_overconstrained =
                            overconstrained_fluid_cell_map[stencil_cell]
                              .is_overconstrained;
                          if (cell2_is_cut || point_in_cell)
                            {
                              dof_is_dummy = true;
                            }

                          if (dof_is_dummy || use_ib_for_pressure ||
                              cell2_is_overconstrained ||
                              particle_close_to_wall)
                            {
                              // Give the DOF an approximated value. This help
                              // with pressure shock when the DOF passes from
                              // part of the boundary to the fluid.

                              this->system_matrix.add(global_index_overwrite,
                                                      global_index_overwrite,
                                                      sum_line);
                              skip_stencil = true;
                              // SHARP-TEMPORARY-DEBUG-TOP
                              debug_row_kind[global_index_overwrite] = 3;
                              ++debug_stencil_counts[3];
                              if (use_ib_for_pressure)
                                ++debug_stencil_counts[4];
                              if (cell2_is_overconstrained)
                                ++debug_stencil_counts[5];
                              // SHARP-TEMPORARY-DEBUG-BOTTOM


                              // Tolerance to define an intersection of
                              // the DOF and IB
                              if (abs(particles[ib_particle_id].get_levelset(
                                    support_points[local_dof_indices[i]],
                                    cell_cut)) <= 1e-12 * dr)
                                {
                                  dof_on_ib = true;
                                }
                            }
                          // Define the variable used for the
                          // extrapolation of the actual solution at the
                          // boundaries in order to define the correction

                          // Define the unit cell_cut points for the points
                          // used in the stencil. If the stencil cell was
                          // found via periodic wrapping, shift the
                          // interpolation points by the same offset so
                          // they are in the stencil cell's coordinate frame.
                          std::vector<Point<dim>>
                            unite_cell_interpolation_points(ib_coef.size());
                          {
                            Point<dim> shifted_point;
                            for (int d = 0; d < dim; ++d)
                              shifted_point[d] =
                                point[d] + stencil_periodic_shift[d];
                            unite_cell_interpolation_points[0] =
                              this->mapping->transform_real_to_unit_cell(
                                stencil_cell, shifted_point);
                          }
                          for (unsigned int j = 1; j < ib_coef.size(); ++j)
                            {
                              // The extrapolation can be disabled for debugging
                              // purposes, although in most cases it shouldn't
                              // since it is a core part of this solver
                              if (this->simulation_parameters
                                    .particlesParameters->enable_extrapolation)
                                {
                                  Point<dim> shifted_interpolation_point;
                                  for (int d = 0; d < dim; ++d)
                                    shifted_interpolation_point[d] =
                                      interpolation_points[j - 1][d] +
                                      stencil_periodic_shift[d];
                                  unite_cell_interpolation_points[j] =
                                    this->mapping->transform_real_to_unit_cell(
                                      stencil_cell,
                                      shifted_interpolation_point);
                                }
                              else
                                {
                                  Point<dim> shifted_support_point;
                                  for (int d = 0; d < dim; ++d)
                                    shifted_support_point[d] =
                                      support_points[local_dof_indices[i]][d] +
                                      stencil_periodic_shift[d];
                                  unite_cell_interpolation_points[j] =
                                    this->mapping->transform_real_to_unit_cell(
                                      stencil_cell, shifted_support_point);
                                }
                            }

                          // SHARP-TEMPORARY-DEBUG-TOP
                          for (const auto &debug_point :
                               unite_cell_interpolation_points)
                            debug_stencil_counts[6] =
                              std::max(debug_stencil_counts[6],
                                       GeometryInfo<dim>::distance_to_unit_cell(
                                         debug_point));
                          for (const auto debug_weight : ib_coef)
                            debug_stencil_counts[7] =
                              std::max(debug_stencil_counts[7],
                                       std::abs(debug_weight));
                          // SHARP-TEMPORARY-DEBUG-BOTTOM
                          std::vector<double> local_interp_sol(ib_coef.size());

                          // Define the new matrix entry for this dof
                          if (skip_stencil == false)
                            {
                              for (unsigned int j = 0;
                                   j < local_dof_indices_2.size();
                                   ++j)
                                {
                                  const int component_j =
                                    this->fe->system_to_component_index(j)
                                      .first;
                                  if (component_j == component_i)
                                    {
                                      //  Define the solution at each point used
                                      //  for the stencil and applied the
                                      //  stencil for the specific DOF. For
                                      //  stencils of degree 4 or higher, the
                                      //  stencil is defined through direct
                                      //  extrapolation of the cell_cut. This
                                      //  can only be done when using a
                                      //  structured mesh as this required a
                                      //  mapping of a point outside of a
                                      //  cell_cut.

                                      // Define the local matrix entries of this
                                      // DOF based on its contribution of each
                                      // of the points used in the stencil
                                      // definition and the coefficient
                                      // associated with this point. This loop
                                      // defined the current solution at the
                                      // boundary using the same stencil. This
                                      // is needed to define the residual.
                                      double local_matrix_entry = 0;
                                      for (unsigned int k = 0;
                                           k < ib_coef.size();
                                           ++k)
                                        {
                                          local_matrix_entry +=
                                            this->fe->shape_value(
                                              j,
                                              unite_cell_interpolation_points
                                                [k]) *
                                            ib_coef[k];
                                          local_interp_sol[k] +=
                                            this->fe->shape_value(
                                              j,
                                              unite_cell_interpolation_points
                                                [k]) *
                                            this->evaluation_point(
                                              local_dof_indices_2[j]);
                                        }
                                      // Update the matrix using the expanded
                                      // column constraint, if any.
                                      add_matrix_entry_with_periodic_expansion(
                                        global_index_overwrite,
                                        local_dof_indices_2[j],
                                        local_matrix_entry * sum_line);
                                    }
                                }
                            }

                          // Define the RHS of the equation.
                          double rhs_add = 0;
                          // Different boundary conditions depending
                          // on the component index of the DOF and
                          // the dimension.
                          double v_ib = stencil.ib_velocity(
                            particles[ib_particle_id],
                            support_points[local_dof_indices[i]],
                            component_i,
                            cell_cut);

                          //  If the pressure is imposed trough IB inside the
                          //  particle we use an approximation of the pressure
                          //  outside of the IB in this cell_cut.
                          if (component_i == dim)
                            {
                              for (unsigned int k = 0;
                                   k < local_dof_indices.size();
                                   ++k)
                                {
                                  // Check if the dof is inside or outside of
                                  // the particle.
                                  bool dof_is_inside_p =
                                    particles[ib_particle_id].get_levelset(
                                      support_points[local_dof_indices[k]],
                                      cell_cut) <= 0;
                                  const int component_k =
                                    this->fe->system_to_component_index(k)
                                      .first;
                                  if (component_k == dim &&
                                      dof_is_inside_p == false)
                                    {
                                      v_ib = this->evaluation_point(
                                        local_dof_indices[k]);
                                      // SHARP-TEMPORARY-DEBUG-TOP
                                      debug_pressure_reference
                                        [global_index_overwrite] =
                                          local_dof_indices[k];
                                      // SHARP-TEMPORARY-DEBUG-BOTTOM
                                      add_matrix_entry_with_periodic_expansion(
                                        global_index_overwrite,
                                        local_dof_indices[k],
                                        -1 * sum_line);
                                      break;
                                    }
                                }
                            }

                          for (unsigned int k = 0; k < ib_coef.size(); ++k)
                            {
                              rhs_add +=
                                -local_interp_sol[k] * ib_coef[k] * sum_line;
                            }
                          // The rhs contribution where the component would
                          // correspond to a pressure dof are divided by the
                          // pressure_scaling_factor. This is because when the
                          // pressure DOFs are rescaled in the newton
                          // correction, there is no distinction between
                          // dummy and non-dummy pressure dofs. Since this
                          // case is only applicable to this solver, the
                          // choice was made to modify the rhs itself.
                          const double pressure_scaling_factor =
                            this->simulation_parameters.stabilization
                              .pressure_scaling_factor;
                          this->system_rhs(global_index_overwrite) =
                            component_i == dim ? (v_ib * sum_line + rhs_add) /
                                                   pressure_scaling_factor :
                                                 v_ib * sum_line + rhs_add;

                          if (dof_on_ib)
                            // Dof is on the immersed boundary
                            this->system_rhs(global_index_overwrite) =
                              component_i == dim ?
                                (v_ib * sum_line - this->evaluation_point(
                                                     global_index_overwrite) *
                                                     sum_line) /
                                  pressure_scaling_factor :
                                (v_ib * sum_line - this->evaluation_point(
                                                     global_index_overwrite) *
                                                     sum_line);

                          if (skip_stencil && dof_on_ib == false)
                            // Impose the value of the dummy dof. This help
                            // with pressure variation when the IB is
                            // moving.
                            this->system_rhs(global_index_overwrite) =
                              component_i == dim ?
                                (sum_line * v_ib - this->evaluation_point(
                                                     global_index_overwrite) *
                                                     sum_line) /
                                  pressure_scaling_factor :
                                sum_line * v_ib - this->evaluation_point(
                                                    global_index_overwrite) *
                                                    sum_line;
                        }

                      if (component_i == dim &&
                          this->locally_owned_dofs.is_element(
                            global_index_overwrite))
                        {
                          // Applied equation on dof that have no equation
                          // defined for them. those DOF become Dummy dof. This
                          // is useful for high degree cells or when a dof is
                          // only element of cells that are cut.
                          unsigned int global_index_overwrite =
                            local_dof_indices[i];
                          bool dummy_dof = true;

                          // To check if the pressure dof is a dummy. first
                          // check if the matrix entry is close to 0.
                          if (abs(this->system_matrix.el(
                                global_index_overwrite,
                                global_index_overwrite)) <= 1e-16 * dr)
                            {
                              // If the matrix entry on the diagonal of this DOF
                              // is close to zero, check if all the cells close
                              // are cut. If it's the case, the DOF is a dummy
                              // DOF.
                              active_neighbors_set =
                                LetheGridTools::find_cells_around_cell<dim>(
                                  vertices_to_cell, cell_cut);
                              for (unsigned int m = 0;
                                   m < active_neighbors_set.size();
                                   m++)
                                {
                                  const auto &neighbor_cell =
                                    active_neighbors_set[m];
                                  neighbor_cell->get_dof_indices(
                                    local_dof_indices_3);
                                  for (const auto o : local_dof_indices_3)
                                    {
                                      if (global_index_overwrite == o)
                                        {
                                          // neighbor_cell contain the same dof
                                          // check if this cell_cut is cut if
                                          // it's not cut this dof must not
                                          // be overwritten
                                          const bool cell_is_cut =
                                            cut_cells_map[neighbor_cell].is_cut;
                                          const bool cell_is_overconstrained =
                                            overconstrained_fluid_cell_map
                                              [neighbor_cell]
                                                .is_overconstrained;
                                          if (cell_is_cut == false &&
                                              cell_is_overconstrained == false)
                                            {
                                              dummy_dof = false;
                                              break;
                                            }
                                        }
                                    }
                                  if (dummy_dof == false)
                                    break;
                                }

                              if (dummy_dof)
                                {
                                  // The DOF is dummy
                                  this->system_matrix.add(
                                    global_index_overwrite,
                                    global_index_overwrite,
                                    sum_line);
                                  auto &system_rhs = this->system_rhs;
                                  system_rhs(global_index_overwrite) = 0;
                                  // SHARP-TEMPORARY-DEBUG-TOP
                                  debug_row_kind[global_index_overwrite] = 4;
                                  // SHARP-TEMPORARY-DEBUG-BOTTOM
                                }
                            }
                        }
                    }
                }

              if (particles[ib_particle_id].position !=
                  saved_particle_pos_for_periodic)
                particles[ib_particle_id].set_position(
                  saved_particle_pos_for_periodic);
            }
        }
    }

  // SHARP-TEMPORARY-DEBUG-TOP
  std::array<unsigned int, 9> debug_local_row_counts{};
  for (const auto &[index, kind] : debug_row_kind)
    if (this->locally_owned_dofs.is_element(index) &&
        kind < debug_local_row_counts.size())
      ++debug_local_row_counts[kind];
  std::ostringstream debug_rank_rows;
  debug_rank_rows << "[SHARP-TEMP] pre_rhs_compress rank="
                  << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                  << " owned=" << this->locally_owned_dofs.n_elements()
                  << " row_kind_counts=";
  for (const auto count : debug_local_row_counts)
    debug_rank_rows << count << ',';
  std::cerr << debug_rank_rows.str() << std::endl;
  if (std::getenv("SHARP_DEBUG_PRECOMPRESS_BARRIER") != nullptr)
    {
      MPI_Barrier(this->mpi_communicator);
      std::cerr << "[SHARP-TEMP] pre_rhs_barrier_passed rank="
                << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                << std::endl;
    }
  // SHARP-TEMPORARY-DEBUG-BOTTOM
  this->system_rhs.compress(VectorOperation::insert);
  this->system_matrix.compress(VectorOperation::add);
  // SHARP-TEMPORARY-DEBUG-TOP
  // Distinguish condensed constraint rows from physical bulk equations.
  for (const auto row : this->locally_owned_dofs)
    if (this->zero_constraints.is_constrained(row) &&
        (!debug_row_kind.count(row) || debug_row_kind.at(row) < 5))
      {
        const auto *entries =
          this->zero_constraints.get_constraint_entries(row);
        debug_row_kind[row] = entries != nullptr && !entries->empty() ? 7 : 8;
      }
  std::vector<double> debug_reference_changes(3, 0.);
  for (const auto &[row, reference] : debug_pressure_reference)
    if (this->locally_owned_dofs.is_element(row))
      {
        const auto old = debug_previous_pressure_reference.find(row);
        if (old == debug_previous_pressure_reference.end())
          ++debug_reference_changes[0];
        else if (old->second != reference)
          ++debug_reference_changes[1];
      }
  for (const auto &[row, reference] : debug_previous_pressure_reference)
    if (this->locally_owned_dofs.is_element(row) &&
        !debug_pressure_reference.count(row))
      ++debug_reference_changes[2];
  Utilities::MPI::sum(debug_reference_changes,
                      this->mpi_communicator,
                      debug_reference_changes);
  this->pcout << "[SHARP-TEMP] pressure_reference_added_switched_removed="
              << debug_reference_changes[0] << ',' << debug_reference_changes[1]
              << ',' << debug_reference_changes[2] << std::endl;
  debug_stencils();
  std::vector<double> debug_transitions(81, 0.);
  for (const auto row : this->locally_owned_dofs)
    {
      const unsigned int old_kind =
        debug_previous_rows.count(row) ? debug_previous_rows.at(row) : 0;
      const unsigned int new_kind =
        debug_row_kind.count(row) ? debug_row_kind.at(row) : 0;
      if (old_kind != new_kind)
        ++debug_transitions[9 * old_kind + new_kind];
    }
  Utilities::MPI::sum(debug_transitions,
                      this->mpi_communicator,
                      debug_transitions);
  this->pcout << "[SHARP-TEMP] row_kind_changes rhs_call=" << debug_rhs_calls;
  for (unsigned int old_kind = 0; old_kind < 9; ++old_kind)
    for (unsigned int new_kind = 0; new_kind < 9; ++new_kind)
      if (debug_transitions[9 * old_kind + new_kind] > 0.)
        this->pcout << " " << old_kind << "->" << new_kind << ':'
                    << debug_transitions[9 * old_kind + new_kind];
  this->pcout << std::endl;
  // SHARP-TEMPORARY-DEBUG-BOTTOM
  // SHARP-TEMPORARY-DEBUG-TOP
    }
  catch (const std::exception &exception)
    {
      std::cerr << "[SHARP-TEMP] sharp_exception rank="
                << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                << " message=" << exception.what() << std::endl;
      throw;
    }
  // SHARP-TEMPORARY-DEBUG-BOTTOM
}


template <int dim>
void
FluidDynamicsSharp<dim>::setup_assemblers()
{
  // The sharp-interface immersed boundary solver only supports BDF methods.
  AssertThrow(
    !time_stepping_is_sdirk(this->simulation_control->get_assembly_method()),
    ExcMessage("SDIRK time-stepping methods are not supported by the "
               "sharp-interface immersed boundary solver. Use a BDF method "
               "instead."));

  this->assemblers.clear();
  assemblers_inside_ib.clear();

  if (this->check_existance_of_bc(
        BoundaryConditions::BoundaryType::function_weak))
    {
      this->assemblers.push_back(
        std::make_shared<WeakDirichletBoundaryCondition<dim>>(
          this->simulation_control,
          this->simulation_parameters.boundary_conditions));
    }
  if (this->check_existance_of_bc(BoundaryConditions::BoundaryType::pressure))
    {
      this->assemblers.push_back(
        std::make_shared<PressureBoundaryCondition<dim>>(
          this->simulation_control,
          this->simulation_parameters.boundary_conditions));
    }
  if (this->check_existance_of_bc(BoundaryConditions::BoundaryType::outlet))
    {
      this->assemblers.push_back(std::make_shared<OutletBoundaryCondition<dim>>(
        this->simulation_control,
        this->simulation_parameters.boundary_conditions));
    }
  if (this->check_existance_of_bc(
        BoundaryConditions::BoundaryType::partial_slip))
    {
      this->assemblers.push_back(
        std::make_shared<PartialSlipDirichletBoundaryCondition<dim>>(
          this->simulation_control,
          this->simulation_parameters.boundary_conditions));
    }
  if (this->simulation_parameters.multiphysics.CLS)
    {
      // Time-stepping schemes
      if (time_stepping_is_bdf(
            this->simulation_control->get_assembly_method()) &&
          this->simulation_parameters.physical_properties_manager
            .density_is_constant())
        {
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesCLSAssemblerBDF<dim>>(
              this->simulation_control));
        }
      else if (time_stepping_is_bdf(
                 this->simulation_control->get_assembly_method()))
        {
          this->assemblers.push_back(
            std::make_shared<
              GLSIsothermalCompressibleNavierStokesCLSAssemblerBDF<dim>>(
              this->simulation_control));
        }

      // Core assemblers
      if (this->simulation_parameters.physical_properties_manager
            .is_non_newtonian())
        {
          // Core assembler with Non newtonian viscosity
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesCLSAssemblerNonNewtonianCore<dim>>(
              this->simulation_control, this->simulation_parameters));
        }
      else if (!this->simulation_parameters.physical_properties_manager
                  .density_is_constant())
        {
          this->assemblers.push_back(
            std::make_shared<
              GLSIsothermalCompressibleNavierStokesCLSAssemblerCore<dim>>(
              this->simulation_control, this->simulation_parameters));
        }
      else
        {
          // Core assembler
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesCLSAssemblerCore<dim>>(
              this->simulation_control, this->simulation_parameters));
        }
    }
  else
    {
      // Time-stepping schemes
      if (time_stepping_is_bdf(this->simulation_control->get_assembly_method()))
        {
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesAssemblerBDF<dim>>(
              this->simulation_control));
        }

      // Velocity sources term
      if (this->simulation_parameters.velocity_sources.rotating_frame_type ==
          Parameters::VelocitySource::RotatingFrameType::srf)
        {
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesAssemblerSRF<dim>>(
              this->simulation_parameters.velocity_sources));
        }

      // Core assemblers
      if (this->simulation_parameters.physical_properties_manager
            .is_non_newtonian())
        {
          // Core assembler with Non newtonian viscosity
          this->assemblers.push_back(
            std::make_shared<GLSNavierStokesAssemblerNonNewtonianCore<dim>>(
              this->simulation_control));
        }
      else
        {
          // Core assembler
          this->assemblers.push_back(
            std::make_shared<PSPGSUPGNavierStokesAssemblerCore<dim>>(
              this->simulation_control));
        }
    }

  assemblers_inside_ib.push_back(
    std::make_shared<LaplaceAssembly<dim>>(this->simulation_control));
}

template <int dim>
void
FluidDynamicsSharp<dim>::assemble_local_system_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  NavierStokesScratchData<dim>                         &scratch_data,
  StabilizedMethodsTensorCopyData<dim>                 &copy_data)
{
  copy_data.cell_is_local = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  // We check if the cell is cut by the IB and if it is overconstrained. If the
  // cell is cut or overconstrained, we adjust the copy data so that the
  // assemblers don't execute work on this cell.
  bool cell_is_cut = cut_cells_map[cell].is_cut;
  bool cell_is_overconstrained =
    overconstrained_fluid_cell_map[cell].is_overconstrained;

  copy_data.cell_is_cut = cell_is_cut || cell_is_overconstrained;

  if (copy_data.cell_is_cut)
    return;

  scratch_data.reinit(
    cell,
    this->evaluation_point,
    *this->previous_solutions,
    this->sdirk_vectors.sum_over_previous_stages,
    this->forcing_function,
    this->flow_control.get_beta(),
    this->simulation_parameters.stabilization.pressure_scaling_factor);

  if (this->simulation_parameters.multiphysics.CLS)
    {
      const DoFHandler<dim> &dof_handler_cls =
        this->multiphysics->get_dof_handler(PhysicsID::CLS);
      typename DoFHandler<dim>::active_cell_iterator phase_cell(
        &(*(this->triangulation)),
        cell->level(),
        cell->index(),
        &dof_handler_cls);

      scratch_data.reinit_cls(
        phase_cell,
        this->multiphysics->get_solution(PhysicsID::CLS),
        this->multiphysics->get_filtered_solution(PhysicsID::CLS),
        this->multiphysics->get_previous_solutions(PhysicsID::CLS));
    }

  scratch_data.calculate_physical_properties();
  copy_data.reset();

  // check if we assemble the NS equation inside the particle or the Laplacian
  // of the variables
  bool cell_is_inside = cells_inside_map[cell].is_inside;
  if (cell_is_inside && this->simulation_parameters.particlesParameters
                            ->assemble_navier_stokes_inside == false)
    {
      for (auto &assembler : this->assemblers_inside_ib)
        {
          assembler->assemble_matrix(scratch_data, copy_data);
        }
    }
  else
    {
      for (auto &assembler : this->assemblers)
        {
          assembler->assemble_matrix(scratch_data, copy_data);
        }
    }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void
FluidDynamicsSharp<dim>::copy_local_matrix_to_global_matrix(
  const StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  if (!copy_data.cell_is_local || copy_data.cell_is_cut)
    return;

  const AffineConstraints<double> &constraints_used = this->zero_constraints;
  constraints_used.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              this->system_matrix);
}

template <int dim>
void
FluidDynamicsSharp<dim>::assemble_local_system_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  NavierStokesScratchData<dim>                         &scratch_data,
  StabilizedMethodsTensorCopyData<dim>                 &copy_data)
{
  copy_data.cell_is_local = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  // We check if the cell is cut by the IB and if it is overconstrained. If the
  // cell is cut or overconstrained, we adjust the copy data so that the
  // assemblers don't execute work on this cell.
  bool cell_is_cut = cut_cells_map[cell].is_cut;

  bool cell_is_overconstrained =
    overconstrained_fluid_cell_map[cell].is_overconstrained;

  copy_data.cell_is_cut = cell_is_cut || cell_is_overconstrained;

  if (copy_data.cell_is_cut)
    return;

  scratch_data.reinit(
    cell,
    this->evaluation_point,
    *this->previous_solutions,
    this->sdirk_vectors.sum_over_previous_stages,
    this->forcing_function,
    this->flow_control.get_beta(),
    this->simulation_parameters.stabilization.pressure_scaling_factor);

  if (this->simulation_parameters.multiphysics.CLS)
    {
      const DoFHandler<dim> &dof_handler_cls =
        this->multiphysics->get_dof_handler(PhysicsID::CLS);
      typename DoFHandler<dim>::active_cell_iterator phase_cell(
        &(*(this->triangulation)),
        cell->level(),
        cell->index(),
        &dof_handler_cls);

      scratch_data.reinit_cls(
        phase_cell,
        this->multiphysics->get_solution(PhysicsID::CLS),
        this->multiphysics->get_filtered_solution(PhysicsID::CLS),
        this->multiphysics->get_previous_solutions(PhysicsID::CLS));
    }

  scratch_data.calculate_physical_properties();
  copy_data.reset();

  // check if we assemble the NS equation inside the particle or the Laplacian
  // of the variables
  bool cell_is_inside = cells_inside_map[cell].is_inside;
  if (cell_is_inside && this->simulation_parameters.particlesParameters
                            ->assemble_navier_stokes_inside == false)
    {
      for (auto &assembler : this->assemblers_inside_ib)
        {
          assembler->assemble_rhs(scratch_data, copy_data);
        }
    }
  else
    {
      for (auto &assembler : this->assemblers)
        {
          assembler->assemble_rhs(scratch_data, copy_data);
        }
    }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void
FluidDynamicsSharp<dim>::copy_local_rhs_to_global_rhs(
  const StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  if (!copy_data.cell_is_local || copy_data.cell_is_cut)
    return;


  const AffineConstraints<double> &constraints_used = this->zero_constraints;
  constraints_used.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              this->system_rhs);
}

template <int dim>
void
FluidDynamicsSharp<dim>::write_checkpoint()
{
  this->FluidDynamicsMatrixBased<dim>::write_checkpoint();

  // Write a table with all the relevant properties of the particle in a table.
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    {
      std::string prefix =
        this->simulation_parameters.simulation_control.output_folder +
        this->simulation_parameters.restart_parameters.filename;

      TableHandler  particles_information_table;
      std::string   filename = prefix + ".ib_particles";
      std::ofstream output(filename.c_str());
      if (sharp_ib_periodic_boundaries.is_periodic_enabled())
        {
          std::ofstream contacts_output(prefix + ".ib_contacts");
          AssertThrow(contacts_output,
                      ExcMessage("Cannot write Sharp-IB contact checkpoint."));
          ib_dem.write_contact_history(contacts_output);
        }

      this->simulation_control->save(prefix);
      ib_particles_pvdhandler.save(prefix + ".ib_particles");

      this->pvdhandler.save(prefix);
      for (unsigned int i_particle = 0; i_particle < particles.size();
           ++i_particle)
        {
          for (unsigned int j = 0;
               j < particles[i_particle].previous_positions.size();
               ++j)
            {
              particles_information_table.add_value("ID", i_particle);
              particles_information_table.add_value(
                "p_x", particles[i_particle].previous_positions[j][0]);
              particles_information_table.set_precision("p_x", 16);
              particles_information_table.add_value(
                "p_y", particles[i_particle].previous_positions[j][1]);
              particles_information_table.set_precision("p_y", 16);
              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "p_z", particles[i_particle].previous_positions[j][2]);
                  particles_information_table.set_precision("p_z", 16);
                }

              particles_information_table.add_value(
                "v_x", particles[i_particle].previous_velocity[j][0]);
              particles_information_table.set_precision("v_x", 16);
              particles_information_table.add_value(
                "v_y", particles[i_particle].previous_velocity[j][1]);
              particles_information_table.set_precision("v_y", 16);
              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "v_z", particles[i_particle].previous_velocity[j][2]);
                  particles_information_table.set_precision("v_z", 16);
                }

              particles_information_table.add_value(
                "f_x", particles[i_particle].previous_fluid_forces[0]);
              particles_information_table.set_precision("f_x", 16);
              particles_information_table.add_value(
                "f_y", particles[i_particle].previous_fluid_forces[1]);
              particles_information_table.set_precision("f_y", 16);

              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "f_z", particles[i_particle].previous_fluid_forces[2]);
                  particles_information_table.set_precision("f_z", 16);
                }

              particles_information_table.add_value(
                "f_xv", particles[i_particle].previous_fluid_viscous_forces[0]);
              particles_information_table.set_precision("f_xv", 16);
              particles_information_table.add_value(
                "f_yv", particles[i_particle].previous_fluid_viscous_forces[1]);
              particles_information_table.set_precision("f_yv", 16);

              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "f_zv",
                    particles[i_particle].previous_fluid_viscous_forces[2]);
                  particles_information_table.set_precision("f_zv", 16);
                }

              particles_information_table.add_value(
                "f_xp",
                particles[i_particle].previous_fluid_pressure_forces[0]);
              particles_information_table.set_precision("f_xp", 16);
              particles_information_table.add_value(
                "f_yp",
                particles[i_particle].previous_fluid_pressure_forces[1]);
              particles_information_table.set_precision("f_yp", 16);

              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "f_zp",
                    particles[i_particle].previous_fluid_pressure_forces[2]);
                  particles_information_table.set_precision("f_zp", 16);
                }

              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "omega_x", particles[i_particle].previous_omega[j][0]);
                  particles_information_table.set_precision("omega_x", 16);
                  particles_information_table.add_value(
                    "omega_y", particles[i_particle].previous_omega[j][1]);
                  particles_information_table.set_precision("omega_y", 16);
                  particles_information_table.add_value(
                    "theta_x",
                    particles[i_particle].previous_orientation[j][0]);
                  particles_information_table.set_precision("theta_x", 16);
                  particles_information_table.add_value(
                    "theta_y",
                    particles[i_particle].previous_orientation[j][1]);
                  particles_information_table.set_precision("theta_y", 16);
                }
              particles_information_table.add_value(
                "omega_z", particles[i_particle].previous_omega[j][2]);
              particles_information_table.set_precision("omega_z", 16);
              particles_information_table.add_value(
                "theta_z", particles[i_particle].previous_orientation[j][2]);
              particles_information_table.set_precision("theta_z", 16);
              if constexpr (dim == 3)
                {
                  particles_information_table.add_value(
                    "T_x", particles[i_particle].previous_fluid_torque[0]);
                  particles_information_table.set_precision("T_x", 16);
                  particles_information_table.add_value(
                    "T_y", particles[i_particle].previous_fluid_torque[1]);
                  particles_information_table.set_precision("T_y", 16);
                }
              particles_information_table.add_value(
                "T_z", particles[i_particle].previous_fluid_torque[2]);
              particles_information_table.set_precision("T_z", 16);
            }
        }
      // Write the table in the checkpoint file.
      particles_information_table.write_text(output);
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::read_checkpoint()
{
  this->FluidDynamicsMatrixBased<dim>::read_checkpoint();

  TimerOutput::Scope t(this->computing_timer,
                       "Reset Sharp-Edge particle information");

  std::string prefix =
    this->simulation_parameters.simulation_control.output_folder +
    this->simulation_parameters.restart_parameters.filename;

  std::string filename = prefix + ".ib_particles";

  ib_particles_pvdhandler.read(prefix + ".ib_particles");
  // refill the table from checkpoint
  for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
    {
      std::string filename_table =
        this->simulation_parameters.simulation_control.output_folder +
        this->simulation_parameters.particlesParameters->ib_force_output_file +
        "." + Utilities::int_to_string(p_i, 2) + ".dat";
      fill_table_from_file(table_p[p_i], filename_table);
    }

  // Read the data of each particle and put the relevant information in a
  // vector.
  std::map<std::string, std::vector<double>> restart_data;
  fill_vectors_from_file(restart_data, filename);

  // Implement the data in the particles.
  if constexpr (dim == 2)
    {
      unsigned int row = 0;
      for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
        {
          unsigned int j = 0;
          while (row < restart_data["ID"].size() &&
                 restart_data["ID"][row] == p_i)
            {
              if (j == 0)
                {
                  particles[p_i].position[0]     = restart_data["p_x"][row];
                  particles[p_i].position[1]     = restart_data["p_y"][row];
                  particles[p_i].velocity[0]     = restart_data["v_x"][row];
                  particles[p_i].velocity[1]     = restart_data["v_y"][row];
                  particles[p_i].fluid_forces[0] = restart_data["f_x"][row];
                  particles[p_i].fluid_forces[1] = restart_data["f_y"][row];
                  particles[p_i].fluid_viscous_forces[0] =
                    restart_data["f_xv"][row];
                  particles[p_i].fluid_viscous_forces[1] =
                    restart_data["f_yv"][row];
                  particles[p_i].fluid_pressure_forces[0] =
                    restart_data["f_xp"][row];
                  particles[p_i].fluid_pressure_forces[1] =
                    restart_data["f_yp"][row];
                  particles[p_i].omega[2]        = restart_data["omega_z"][row];
                  particles[p_i].orientation[2]  = restart_data["theta_z"][row];
                  particles[p_i].fluid_torque[2] = restart_data["T_z"][row];

                  // fill previous time step
                  particles[p_i].previous_positions[j][0] =
                    restart_data["p_x"][row];
                  particles[p_i].previous_positions[j][1] =
                    restart_data["p_y"][row];
                  particles[p_i].previous_velocity[j][0] =
                    restart_data["v_x"][row];
                  particles[p_i].previous_velocity[j][1] =
                    restart_data["v_y"][row];
                  particles[p_i].previous_omega[j][2] =
                    restart_data["omega_z"][row];
                  particles[p_i].previous_orientation[j][2] =
                    restart_data["theta_z"][row];

                  particles[p_i].set_position(particles[p_i].position);
                  particles[p_i].set_orientation(particles[p_i].orientation);
                }
              else
                {
                  // fill previous time step
                  particles[p_i].previous_positions[j][0] =
                    restart_data["p_x"][row];
                  particles[p_i].previous_positions[j][1] =
                    restart_data["p_y"][row];
                  particles[p_i].previous_velocity[j][0] =
                    restart_data["v_x"][row];
                  particles[p_i].previous_velocity[j][1] =
                    restart_data["v_y"][row];
                  particles[p_i].previous_omega[j][2] =
                    restart_data["omega_z"][row];
                  particles[p_i].previous_orientation[j][2] =
                    restart_data["theta_z"][row];
                }
              row += 1;
              j += 1;
            }
        }
    }
  if constexpr (dim == 3)
    {
      unsigned int row = 0;
      for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
        {
          unsigned int j = 0;
          while (row < restart_data["ID"].size() &&
                 restart_data["ID"][row] == p_i)
            {
              if (j == 0)
                {
                  particles[p_i].position[0]     = restart_data["p_x"][row];
                  particles[p_i].position[1]     = restart_data["p_y"][row];
                  particles[p_i].position[2]     = restart_data["p_z"][row];
                  particles[p_i].velocity[0]     = restart_data["v_x"][row];
                  particles[p_i].velocity[1]     = restart_data["v_y"][row];
                  particles[p_i].velocity[2]     = restart_data["v_z"][row];
                  particles[p_i].fluid_forces[0] = restart_data["f_x"][row];
                  particles[p_i].fluid_forces[1] = restart_data["f_y"][row];
                  particles[p_i].fluid_forces[2] = restart_data["f_z"][row];
                  particles[p_i].fluid_viscous_forces[0] =
                    restart_data["f_xv"][row];
                  particles[p_i].fluid_viscous_forces[1] =
                    restart_data["f_yv"][row];
                  particles[p_i].fluid_viscous_forces[2] =
                    restart_data["f_zv"][row];
                  particles[p_i].fluid_pressure_forces[0] =
                    restart_data["f_xp"][row];
                  particles[p_i].fluid_pressure_forces[1] =
                    restart_data["f_yp"][row];
                  particles[p_i].fluid_pressure_forces[2] =
                    restart_data["f_zp"][row];
                  particles[p_i].orientation[0]  = restart_data["theta_x"][row];
                  particles[p_i].orientation[1]  = restart_data["theta_y"][row];
                  particles[p_i].orientation[2]  = restart_data["theta_z"][row];
                  particles[p_i].omega[0]        = restart_data["omega_x"][row];
                  particles[p_i].omega[1]        = restart_data["omega_y"][row];
                  particles[p_i].omega[2]        = restart_data["omega_z"][row];
                  particles[p_i].fluid_torque[0] = restart_data["T_x"][row];
                  particles[p_i].fluid_torque[1] = restart_data["T_y"][row];
                  particles[p_i].fluid_torque[2] = restart_data["T_z"][row];

                  // fill previous time step
                  particles[p_i].previous_positions[j][0] =
                    restart_data["p_x"][row];
                  particles[p_i].previous_positions[j][1] =
                    restart_data["p_y"][row];
                  particles[p_i].previous_positions[j][2] =
                    restart_data["p_z"][row];
                  particles[p_i].previous_velocity[j][0] =
                    restart_data["v_x"][row];
                  particles[p_i].previous_velocity[j][1] =
                    restart_data["v_y"][row];
                  particles[p_i].previous_velocity[j][2] =
                    restart_data["v_z"][row];
                  particles[p_i].previous_omega[j][0] =
                    restart_data["omega_x"][row];
                  particles[p_i].previous_omega[j][1] =
                    restart_data["omega_y"][row];
                  particles[p_i].previous_omega[j][2] =
                    restart_data["omega_z"][row];
                  particles[p_i].previous_orientation[j][0] =
                    restart_data["theta_x"][row];
                  particles[p_i].previous_orientation[j][1] =
                    restart_data["theta_y"][row];
                  particles[p_i].previous_orientation[j][2] =
                    restart_data["theta_z"][row];

                  particles[p_i].set_position(particles[p_i].position);
                  particles[p_i].set_orientation(particles[p_i].orientation);
                }
              else
                {
                  // fill previous time step
                  particles[p_i].previous_positions[j][0] =
                    restart_data["p_x"][row];
                  particles[p_i].previous_positions[j][1] =
                    restart_data["p_y"][row];
                  particles[p_i].previous_positions[j][2] =
                    restart_data["p_z"][row];
                  particles[p_i].previous_velocity[j][0] =
                    restart_data["v_x"][row];
                  particles[p_i].previous_velocity[j][1] =
                    restart_data["v_y"][row];
                  particles[p_i].previous_velocity[j][2] =
                    restart_data["v_z"][row];
                  particles[p_i].previous_omega[j][0] =
                    restart_data["omega_x"][row];
                  particles[p_i].previous_omega[j][1] =
                    restart_data["omega_y"][row];
                  particles[p_i].previous_omega[j][2] =
                    restart_data["omega_z"][row];
                  particles[p_i].previous_orientation[j][0] =
                    restart_data["theta_x"][row];
                  particles[p_i].previous_orientation[j][1] =
                    restart_data["theta_y"][row];
                  particles[p_i].previous_orientation[j][2] =
                    restart_data["theta_z"][row];
                }
              row += 1;
              j += 1;
            }
        }
    }
  // Check if particles are spheres
  check_whether_all_particles_are_sphere();

  // Create the list of contact candidates
  ib_dem.initialize(
    this->simulation_parameters.particlesParameters,
    std::make_shared<Parameters::Lagrangian::FloatingWalls<dim>>(
      cfd_dem_parameters.dem_parameters.floating_walls),
    std::make_shared<Parameters::Lagrangian::BCDEM>(
      cfd_dem_parameters.dem_parameters.boundary_conditions),
    this->mpi_communicator,
    particles);
  if (sharp_ib_periodic_boundaries.is_periodic_enabled())
    {
      // Older checkpoints have no spring-history file and retain legacy empty
      // history.
      std::ifstream contacts_input(prefix + ".ib_contacts");
      if (contacts_input)
        ib_dem.read_contact_history(contacts_input);
    }
  ib_dem.update_contact_candidates();
  // Finish the time step of the particles.
  for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
    {
      particles[p_i].velocity_iter        = particles[p_i].velocity;
      particles[p_i].impulsion_iter       = particles[p_i].impulsion;
      particles[p_i].omega_iter           = particles[p_i].omega;
      particles[p_i].omega_impulsion_iter = particles[p_i].omega_impulsion;
    }
  // Finish the time step of the particle.

  update_precalculations_for_ib();
  for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
    {
      TimerOutput::Scope t(this->computing_timer, "Remove RBF nodes");
      particles[p_i].remove_superfluous_data(
        *this->dof_handler, particles[p_i].mesh_based_precalculations);
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::load_particles_from_file()
{
  using dealii::numbers::PI;
  TimerOutput::Scope t(this->computing_timer,
                       "Reset Sharp-Edge particle information");
  this->pcout << "Loading particles from a file" << std::endl;
  std::string filename =
    this->simulation_parameters.particlesParameters->particles_file;

  // Read the data of each particle and put the relevant information in a
  // vector.
  std::map<std::string, std::vector<std::string>> particles_data;
  fill_string_vectors_from_file(particles_data, filename, ";");
  particles.resize(particles_data["type"].size());

  this->pcout << "Particles found: " << particles.size() << std::endl;

  // The inertia is given either as a single value, when it is uniform around
  // each axis, or as the 9 components of the inertia matrix separated by ":".
  const auto parse_inertia = [](const std::string &inertia_str) {
    const std::vector<double> inertia_list = Utilities::string_to_double(
      Utilities::split_string_list(inertia_str, ":"));
    Tensor<2, 3> inertia;
    if (inertia_list.size() == 9)
      {
        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = 0; j < 3; ++j)
            inertia[i][j] = inertia_list[3 * i + j];
      }
    else if (inertia_list.size() == 1)
      {
        for (unsigned int i = 0; i < 3; ++i)
          inertia[i][i] = inertia_list[0];
      }
    else
      {
        throw(std::runtime_error(
          " Invalid inertia matrix. The inertia is given as a 3 by 3 matrix or a single value if the inertia is uniform around each axis."));
      }
    return inertia;
  };

  // Implement the data  in the particles.
  if constexpr (dim == 2)
    {
      // Loop over each line of the file and filling the particles properties.
      for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
        {
          particles[p_i].initialize_all();

          particles[p_i].particle_id = p_i;

          particles[p_i].position[0] =
            Utilities::string_to_double(particles_data["p_x"][p_i]);
          particles[p_i].position[1] =
            Utilities::string_to_double(particles_data["p_y"][p_i]);
          particles[p_i].velocity[0] =
            Utilities::string_to_double(particles_data["v_x"][p_i]);
          particles[p_i].velocity[1] =
            Utilities::string_to_double(particles_data["v_y"][p_i]);
          particles[p_i].orientation[2] =
            Utilities::string_to_double(particles_data["orientation_z"][p_i]);
          // Initialize shape of particles according to the type parameter in
          // the file

          std::string shape_type = particles_data["type"][p_i];
          std::string shape_arguments_str =
            particles_data["shape_argument"][p_i];
          // Here shape is supposed to be separated by : but the initializer
          // take ; as separator so we replace : by ;.
          std::string toReplace   = ":";
          std::string replacement = ";";

          // Check if the substring is found
          size_t pos = shape_arguments_str.find(toReplace);
          while (pos != std::string::npos)
            {
              // Replace the substring
              shape_arguments_str.replace(pos, toReplace.length(), replacement);

              // Find next occurrence of the substring
              pos =
                shape_arguments_str.find(toReplace, pos + replacement.length());
            }
          particles[p_i].initialize_shape(shape_type, shape_arguments_str);


          particles[p_i].radius = particles[p_i].shape->effective_radius;

          double volume =
            Utilities::string_to_double(particles_data["volume"][p_i]);
          if (volume == 0)
            {
              // value is automatically defined.
              volume = particles[p_i].shape->displaced_volume();
              if (volume == 0)
                {
                  if constexpr (dim == 2)
                    {
                      volume =
                        PI * particles[p_i].radius * particles[p_i].radius;
                    }
                  else if constexpr (dim == 3)
                    {
                      volume = 4.0 / 3.0 * PI * particles[p_i].radius *
                               particles[p_i].radius * particles[p_i].radius;
                    }
                }
            }
          particles[p_i].volume = volume;
          particles[p_i].mass =
            particles[p_i].volume *
            Utilities::string_to_double(particles_data["density"][p_i]);

          particles[p_i].omega[2] =
            Utilities::string_to_double(particles_data["omega_z"][p_i]);

          particles[p_i].inertia =
            parse_inertia(particles_data["inertia"][p_i]);

          particles[p_i].pressure_location[0] =
            Utilities::string_to_double(particles_data["pressure_x"][p_i]);
          particles[p_i].pressure_location[1] =
            Utilities::string_to_double(particles_data["pressure_y"][p_i]);

          particles[p_i].youngs_modulus =
            Utilities::string_to_double(particles_data["youngs_modulus"][p_i]);
          particles[p_i].restitution_coefficient = Utilities::string_to_double(
            particles_data["restitution_coefficient"][p_i]);
          particles[p_i].friction_coefficient = Utilities::string_to_double(
            particles_data["friction_coefficient"][p_i]);
          particles[p_i].poisson_ratio =
            Utilities::string_to_double(particles_data["poisson_ratio"][p_i]);
          particles[p_i].rolling_friction_coefficient =
            Utilities::string_to_double(
              particles_data["rolling_friction_coefficient"][p_i]);
          particles[p_i].initialize_previous_solution();
          if (particles_data["integrate_motion"][p_i] == "false")
            {
              particles[p_i].integrate_motion = false;
            }
          else
            {
              particles[p_i].integrate_motion = true;
            }
          particles[p_i].set_position(particles[p_i].position);
          particles[p_i].set_orientation(particles[p_i].orientation);
        }
    }
  if constexpr (dim == 3)
    {
      // Loop over each line of the file and filling the particles properties.
      for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
        {
          particles[p_i].initialize_all();

          particles[p_i].particle_id = p_i;
          particles[p_i].position[0] =
            Utilities::string_to_double(particles_data["p_x"][p_i]);
          particles[p_i].position[1] =
            Utilities::string_to_double(particles_data["p_y"][p_i]);
          particles[p_i].position[2] =
            Utilities::string_to_double(particles_data["p_z"][p_i]);
          particles[p_i].velocity[0] =
            Utilities::string_to_double(particles_data["v_x"][p_i]);
          particles[p_i].velocity[1] =
            Utilities::string_to_double(particles_data["v_y"][p_i]);
          particles[p_i].velocity[2] =
            Utilities::string_to_double(particles_data["v_z"][p_i]);
          particles[p_i].orientation[0] =
            Utilities::string_to_double(particles_data["orientation_x"][p_i]);
          particles[p_i].orientation[1] =
            Utilities::string_to_double(particles_data["orientation_y"][p_i]);
          particles[p_i].orientation[2] =
            Utilities::string_to_double(particles_data["orientation_z"][p_i]);

          std::string shape_type = particles_data["type"][p_i];
          std::string shape_arguments_str =
            particles_data["shape_argument"][p_i];
          // Here shape is supposed to be separated by : but the initializer
          // takes ; as separator so we replace : by ;.
          std::string toReplace   = ":";
          std::string replacement = ";";

          // Check if the substring is found
          size_t pos = shape_arguments_str.find(toReplace);
          while (pos != std::string::npos)
            {
              // Replace the substring
              shape_arguments_str.replace(pos, toReplace.length(), replacement);

              // Find next occurrence of the substring
              pos =
                shape_arguments_str.find(toReplace, pos + replacement.length());
            }
          particles[p_i].initialize_shape(shape_type, shape_arguments_str);

          particles[p_i].radius = particles[p_i].shape->effective_radius;
          double volume =
            Utilities::string_to_double(particles_data["volume"][p_i]);
          if (volume == 0)
            {
              // value is automatically defined.
              volume = particles[p_i].shape->displaced_volume();
              if (volume == 0)
                {
                  if constexpr (dim == 2)
                    {
                      volume =
                        PI * particles[p_i].radius * particles[p_i].radius;
                    }
                  else if constexpr (dim == 3)
                    {
                      volume = 4.0 / 3.0 * PI * particles[p_i].radius *
                               particles[p_i].radius * particles[p_i].radius;
                    }
                }
            }
          particles[p_i].volume = volume;
          particles[p_i].mass =
            particles[p_i].volume *
            Utilities::string_to_double(particles_data["density"][p_i]);

          particles[p_i].omega[0] =
            Utilities::string_to_double(particles_data["omega_x"][p_i]);
          particles[p_i].omega[1] =
            Utilities::string_to_double(particles_data["omega_y"][p_i]);
          particles[p_i].omega[2] =
            Utilities::string_to_double(particles_data["omega_z"][p_i]);

          particles[p_i].inertia =
            parse_inertia(particles_data["inertia"][p_i]);

          particles[p_i].pressure_location[0] =
            Utilities::string_to_double(particles_data["pressure_x"][p_i]);
          particles[p_i].pressure_location[1] =
            Utilities::string_to_double(particles_data["pressure_y"][p_i]);
          particles[p_i].pressure_location[2] =
            Utilities::string_to_double(particles_data["pressure_z"][p_i]);

          particles[p_i].youngs_modulus =
            Utilities::string_to_double(particles_data["youngs_modulus"][p_i]);
          particles[p_i].restitution_coefficient = Utilities::string_to_double(
            particles_data["restitution_coefficient"][p_i]);
          particles[p_i].friction_coefficient = Utilities::string_to_double(
            particles_data["friction_coefficient"][p_i]);
          particles[p_i].poisson_ratio =
            Utilities::string_to_double(particles_data["poisson_ratio"][p_i]);
          particles[p_i].rolling_friction_coefficient =
            Utilities::string_to_double(
              particles_data["rolling_friction_coefficient"][p_i]);
          particles[p_i].initialize_previous_solution();
          if (particles_data["integrate_motion"][p_i] == "false")
            {
              particles[p_i].integrate_motion = false;
            }
          else
            {
              particles[p_i].integrate_motion = true;
            }
          particles[p_i].set_position(particles[p_i].position);
          particles[p_i].set_orientation(particles[p_i].orientation);
        }
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::update_precalculations_for_ib()
{
  TimerOutput::Scope t(this->computing_timer,
                       "Update precalculated shape distance");
  for (unsigned int p_i = 0; p_i < particles.size(); ++p_i)
    {
      particles[p_i].update_precalculations(
        *this->dof_handler, particles[p_i].mesh_based_precalculations);
    }
}

template <int dim>
void
FluidDynamicsSharp<dim>::solve()
{
  AssertThrow(
    !cfd_dem_parameters.dem_parameters.post_processing
       .lagrangian_post_processing_enabled,
    ExcMessage(
      "DEM lagrangian post-processing is not supported for lethe-fluid-sharp. "
      "Disable it in the DEM post-processing subsection."));

  AssertThrow(!cfd_dem_parameters.dem_parameters.post_processing
                 .particle_wall_collision_statistics,
              ExcMessage(
                "DEM particle-wall collision statistics are not supported for "
                "lethe-fluid-sharp. Disable them in the DEM post-processing "
                "subsection."));

  read_mesh_and_manifolds(
    *this->triangulation,
    this->simulation_parameters.mesh,
    this->simulation_parameters.manifolds_parameters,
    this->simulation_parameters.restart_parameters.restart,
    this->simulation_parameters.boundary_conditions.periodic_boundaries,
    &this->computing_timer);

  define_particles();
  initialize_sharp_ib_periodic_configuration();
  this->setup_dofs();
  {
    TimerOutput::Scope timer(this->computing_timer,
                             "Initialize Sharp periodic geometry");
    update_sharp_ib_periodic_geometry();
    if (sharp_ib_periodic_boundaries.is_periodic_enabled())
      for (auto &particle : particles)
        {
          const Point<dim> normalized =
            sharp_ib_periodic_boundaries.wrap_point(particle.position,
                                                    periodic_domain_lower,
                                                    periodic_domain_upper);
          const Tensor<1, dim> shift = normalized - particle.position;
          if (shift.norm_square() != 0.)
            {
              particle.set_position(normalized);
              sharp_ib_periodic_boundaries.translate_history(
                particle.previous_positions, shift);
            }
        }
  }

  if (this->simulation_parameters.restart_parameters.restart == false)
    {
      refinement_control(true);
      vertices_cell_mapping();

      if (all_spheres)
        optimized_generate_cut_cells_map();
      else
        generate_cut_cells_map();
    }

  this->set_initial_condition(
    this->simulation_parameters.initial_condition->type,
    this->simulation_parameters.restart_parameters.restart);
  this->update_multiphysics_time_average_solution();
  if (this->simulation_parameters.restart_parameters.restart == false &&
      this->simulation_control->is_output_iteration())
    {
      // Match the fluid solver by writing the initial Sharp-IB particle output
      // on step 0 whenever the current iteration is an output iteration.
      handle_dem_particle_output_and_postprocessing();
    }

  while (this->simulation_control->integrate())
    {
      this->simulation_control->print_progression(this->pcout);
      this->forcing_function->set_time(
        this->simulation_control->get_current_time());

      // We allow the physics to update their boundary conditions
      // according to their own parameters
      this->update_boundary_conditions();
      this->multiphysics->update_boundary_conditions();

      this->dynamic_flow_control();

      if (some_particles_are_coupled == false)
        integrate_particles();


      if (this->simulation_control->is_at_start())
        {
          vertices_cell_mapping();
          update_precalculations_for_ib();

          ib_dem.update_particles_boundary_contact(this->particles,
                                                   *this->dof_handler,
                                                   *this->face_quadrature,
                                                   *this->mapping);
          ib_dem.update_contact_candidates();
          this->iterate();
        }
      else
        {
          ib_done.clear();
          refinement_control(false);
          vertices_cell_mapping();

          ib_dem.update_particles_boundary_contact(this->particles,
                                                   *this->dof_handler,
                                                   *this->face_quadrature,
                                                   *this->mapping);
          if (this->simulation_control->get_iteration_number() == 0 ||
              this->simulation_control->get_iteration_number() %
                  this->simulation_parameters.particlesParameters
                    ->contact_search_frequency ==
                0)
            ib_dem.update_contact_candidates();

          // add initialization
          this->iterate();
        }

      this->postprocess(false);
      handle_dem_particle_output_and_postprocessing();

      if (this->simulation_parameters.particlesParameters->calculate_force_ib)
        force_on_ib();
      finish_time_step_particles();
      write_force_ib();
      this->finish_time_step();
    }

  this->finish_simulation();
}


template <int dim>
void
FluidDynamicsSharp<dim>::setup_dofs_fd()
{
  // SHARP-TEMPORARY-DEBUG-TOP
  debug_force_baseline_valid = false;
  debug_base_valid           = false;
  debug_components_ready     = false;
  debug_components.clear();
  debug_pressure_reference.clear();
  debug_row_kind.clear();
  // SHARP-TEMPORARY-DEBUG-BOTTOM

  // DoF redistribution also invalidates cached cells without a mesh change.
  periodic_stencil_search.reinit(*this->dof_handler);
  // Call the base class to do all standard setup (DoF distribution,
  // constraints, vector reinit, sparsity pattern, matrix reinit).
  FluidDynamicsMatrixBased<dim>::setup_dofs_fd();

  // If periodic boundaries are not enabled, the standard sparsity
  // pattern (built with keep_constrained_dofs=false) is sufficient.
  if (!sharp_ib_periodic_boundaries.is_periodic_enabled())
    return;

  // Sharp stencils may use DoFs on ghost cells. Import their constraints from
  // their owners before constructing the sparsity pattern or expanding stencil
  // columns, so all ranks apply the same periodic and boundary constraints.
  this->zero_constraints.make_consistent_in_parallel(this->locally_owned_dofs,
                                                     this->locally_relevant_dofs,
                                                     this->mpi_communicator);
  this->nonzero_constraints.make_consistent_in_parallel(
    this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);

  // The base Setup DOFs scope has ended; report this rebuild separately.
  TimerOutput::Scope timer(this->computing_timer,
                           "Setup Sharp periodic sparsity");

  // Retain condensed element couplings and add only the slave-to-master
  // entries needed by the Sharp constraint equations.
  auto &nonzero_constraints = this->get_nonzero_constraints();

  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(*this->dof_handler,
                                  dsp,
                                  nonzero_constraints,
                                  false);

  if (this->simulation_parameters.mortar_parameters.enable)
    this->mortar_coupling_operator->add_sparsity_pattern_entries(dsp);

  // Constrained diagonals already exist; add their master columns.
  for (auto dof = this->locally_relevant_dofs.begin();
       dof != this->locally_relevant_dofs.end();
       ++dof)
    {
      const auto *entries = this->zero_constraints.get_constraint_entries(*dof);
      if (entries == nullptr)
        entries = this->nonzero_constraints.get_constraint_entries(*dof);

      if (entries != nullptr && !entries->empty())
        {
          // Off-diagonal (slave, master) entries so the constraint
          // equation A(slave, master) = weight can be written.
          for (const auto &[master_dof, weight] : *entries)
            {
              (void)weight;
              dsp.add(*dof, master_dof);
            }
        }
    }

  // SHARP-TEMPORARY-DEBUG-TOP
  // Small-case isolation experiment only: a complete locally relevant graph
  // tests whether absent stencil entries cause the matrix insertion failure.
  // This is intentionally not a scalable production sparsity construction.
  if (std::getenv("SHARP_DEBUG_DENSE_RELEVANT_GRAPH") != nullptr)
    {
      AssertThrow(this->dof_handler->n_dofs() <= 10000,
                  ExcMessage("Dense diagnostic graph is restricted to small cases."));
      for (const auto row : this->locally_owned_dofs)
        for (const auto column : this->locally_relevant_dofs)
          dsp.add(row, column);
    }
  // SHARP-TEMPORARY-DEBUG-BOTTOM
  SparsityTools::distribute_sparsity_pattern(
    dsp,
    this->dof_handler->locally_owned_dofs(),
    this->mpi_communicator,
    this->locally_relevant_dofs);

  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);
}


template <int dim>
void
FluidDynamicsSharp<dim>::initialize_sharp_ib_periodic_configuration()
{
  sharp_ib_periodic_boundaries.initialize(
    cfd_dem_parameters.dem_parameters.boundary_conditions,
    this->simulation_parameters.boundary_conditions,
    this->pcout);
  if (sharp_ib_periodic_boundaries.is_periodic_enabled())
    for (const auto &particle : particles)
      {
        const auto *composite =
          dynamic_cast<const CompositeShape<dim> *>(particle.shape.get());
        AssertThrow(
          !dynamic_cast<const RBFShape<dim> *>(particle.shape.get()) &&
            !(composite && composite->contains_rbf()),
          ExcMessage(
            "RBF shapes, including RBF constituents of composite shapes, "
            "are not yet supported with periodic Sharp-IB boundaries."));
      }

  // The DEM exporter recomputes overlap from primary positions and therefore
  // omits contacts across the seam. Reject misleading force-chain output.
  AssertThrow(
    !sharp_ib_periodic_boundaries.is_periodic_enabled() ||
      !cfd_dem_parameters.dem_parameters.post_processing.force_chains,
    ExcMessage(
      "Force-chain output is not supported for periodic Sharp-IB particles."));
}


template <int dim>
typename DoFHandler<dim>::active_cell_iterator
FluidDynamicsSharp<dim>::find_periodic_stencil_cell(
  const typename DoFHandler<dim>::active_cell_iterator &seed,
  const Point<dim>                                     &point,
  Tensor<1, dim>                                       &shift)
{
  const Point<dim> wrapped =
    sharp_ib_periodic_boundaries.wrap_point(point,
                                            periodic_domain_lower,
                                            periodic_domain_upper);
  shift = wrapped - point;
  const auto cell =
    periodic_stencil_search.find(wrapped, seed, vertices_to_cell);
  AssertThrow(cell != this->dof_handler->end() || wrapped == point,
              ExcMessage(
                "Periodic Sharp stencil is unavailable on this rank."));
  return cell;
}

template <int dim>
void
FluidDynamicsSharp<dim>::update_sharp_ib_periodic_geometry()
{
  if (!sharp_ib_periodic_boundaries.is_periodic_enabled())
    return;

  const unsigned int periodic_direction =
    sharp_ib_periodic_boundaries.get_periodic_direction();

  // Sharp-IB currently supports one periodic pair on opposite extremal faces
  // of a box-like domain. Under that constraint, the wrap length is the global
  // min/max coordinate span in the periodic direction, so it can be recovered
  // from the mesh-wide vertex extrema below.
  //
  // If support is ever extended to periodic faces that are not the geometric
  // extrema in that direction, these bounds must instead be computed directly
  // from the configured periodic boundary ids.
  periodic_domain_lower = std::numeric_limits<double>::max();
  periodic_domain_upper = std::numeric_limits<double>::lowest();

  for (const auto &cell : this->triangulation->active_cell_iterators())
    {
      for (unsigned int vertex_index = 0;
           vertex_index < GeometryInfo<dim>::vertices_per_cell;
           ++vertex_index)
        {
          const Point<dim> &vertex = cell->vertex(vertex_index);
          periodic_domain_lower =
            std::min(periodic_domain_lower, vertex[periodic_direction]);
          periodic_domain_upper =
            std::max(periodic_domain_upper, vertex[periodic_direction]);
        }
    }

  periodic_domain_lower =
    Utilities::MPI::min(periodic_domain_lower, this->mpi_communicator);
  periodic_domain_upper =
    Utilities::MPI::max(periodic_domain_upper, this->mpi_communicator);

  const double length = periodic_domain_upper - periodic_domain_lower;
  AssertThrow(std::isfinite(length) && length > 0.,
              ExcMessage("Sharp-IB requires a positive finite periodic span."));
  const auto   ids = sharp_ib_periodic_boundaries.get_periodic_boundary_ids();
  const double tolerance         = 1e-10 * length;
  std::array<double, 2> face_min = {
    {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}};
  std::array<double, 2> face_max = {{std::numeric_limits<double>::lowest(),
                                     std::numeric_limits<double>::lowest()}};
  for (const auto &cell : this->triangulation->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          for (unsigned int side = 0; side < 2; ++side)
            if (cell->face(face)->boundary_id() ==
                (side == 0 ? ids.first : ids.second))
              for (const auto vertex : cell->face(face)->vertex_indices())
                {
                  const double x =
                    cell->face(face)->vertex(vertex)[periodic_direction];
                  face_min[side] = std::min(face_min[side], x);
                  face_max[side] = std::max(face_max[side], x);
                }
  for (unsigned int side = 0; side < 2; ++side)
    {
      face_min[side] =
        Utilities::MPI::min(face_min[side], this->mpi_communicator);
      face_max[side] =
        Utilities::MPI::max(face_max[side], this->mpi_communicator);
      AssertThrow(
        face_max[side] >= face_min[side] &&
          face_max[side] - face_min[side] <= tolerance,
        ExcMessage(
          "Sharp-IB periodic faces must exist and be planar normal to the periodic direction."));
    }
  AssertThrow(
    std::abs(std::min(face_min[0], face_min[1]) - periodic_domain_lower) <=
        tolerance &&
      std::abs(std::max(face_max[0], face_max[1]) - periodic_domain_upper) <=
        tolerance,
    ExcMessage(
      "Sharp-IB periodic faces must be opposite extremal faces of the mesh."));

  Tensor<1, dim> periodic_offset;
  periodic_offset[periodic_direction] =
    periodic_domain_upper - periodic_domain_lower;
  sharp_ib_periodic_boundaries.set_periodic_offset(periodic_offset);
  ib_dem.set_periodic_offset(periodic_offset);

  this->pcout << "  Periodic domain bounds [" << periodic_domain_lower << ", "
              << periodic_domain_upper << "] in direction "
              << periodic_direction << std::endl;
  this->pcout << "  Periodic offset: " << periodic_offset << std::endl;
}


template <int dim>
void
FluidDynamicsSharp<dim>::add_matrix_entry_with_periodic_expansion(
  const types::global_dof_index row_dof,
  const types::global_dof_index col_dof,
  const double                  value)
{
  // Sharp-IB equations are assembled row-wise. Never write to a non-owned row:
  // nonlocal row inserts are not guaranteed and can make parallel matrices
  // rank-dependent.
  if (!this->locally_owned_dofs.is_element(row_dof))
    return;

  if (!sharp_ib_periodic_boundaries.is_periodic_enabled())
    {
      // Retain upstream nonperiodic assembly, including its handling of
      // entries absent from the condensed sparsity pattern.
      try
        {
          this->system_matrix.add(row_dof, col_dof, value);
        }
      catch (...)
        {}
      return;
    }

  // When sharp_edge() writes IB stencil or constraint equations, the column
  // DoF may be a periodic slave whose entry is not in the sparsity pattern
  // (built with keep_constrained_dofs=false). We therefore expand COLUMN
  // constraints explicitly so the custom Sharp-IB row writes the same master
  // columns that deal.II would use for standard constrained assembly.
  //
  // Keep the requested row: sharp_edge() also writes constraint equations on
  // slave rows. All resulting entries must already exist in the matrix graph.

  // Check if the column DoF is constrained (periodic or hanging node)
  bool col_is_constrained = this->zero_constraints.is_constrained(col_dof) ||
                            this->nonzero_constraints.is_constrained(col_dof);

  if (!col_is_constrained)
    {
      // SHARP-TEMPORARY-DEBUG-TOP
      try
        {
      // SHARP-TEMPORARY-DEBUG-BOTTOM
      this->system_matrix.add(row_dof, col_dof, value);
      // SHARP-TEMPORARY-DEBUG-TOP
        }
      catch (const std::exception &)
        {
          std::cerr << "[SHARP-TEMP] missing_matrix_entry rank="
                    << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                    << " row=" << row_dof << " requested_column=" << col_dof
                    << " actual_column=" << col_dof << " value=" << value
                    << std::endl;
          throw;
        }
      // SHARP-TEMPORARY-DEBUG-BOTTOM
    }
  else
    {
      // For x_col = sum_i(weight_i * x_master_i), add weight_i * value to
      // each master column. This helper only updates the matrix; sharp_edge()
      // constructs the residual separately.

      const auto *entries =
        this->zero_constraints.get_constraint_entries(col_dof);

      if (entries == nullptr)
        entries = this->nonzero_constraints.get_constraint_entries(col_dof);

      if (entries != nullptr)
        {
          for (const auto &entry : *entries)
            {
              const types::global_dof_index master_col = entry.first;
              const double                  weight     = entry.second;
              // SHARP-TEMPORARY-DEBUG-TOP
              try
                {
              // SHARP-TEMPORARY-DEBUG-BOTTOM
              this->system_matrix.add(row_dof, master_col, weight * value);
              // SHARP-TEMPORARY-DEBUG-TOP
                }
              catch (const std::exception &)
                {
                  std::cerr << "[SHARP-TEMP] missing_matrix_entry rank="
                            << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                            << " row=" << row_dof << " requested_column=" << col_dof
                            << " actual_column=" << master_col << " value=" << value
                            << std::endl;
                  throw;
                }
              // SHARP-TEMPORARY-DEBUG-BOTTOM
            }
        }
      else
        {
          // SHARP-TEMPORARY-DEBUG-TOP
          try
            {
          // SHARP-TEMPORARY-DEBUG-BOTTOM
          this->system_matrix.add(row_dof, col_dof, value);
          // SHARP-TEMPORARY-DEBUG-TOP
            }
          catch (const std::exception &)
            {
              std::cerr << "[SHARP-TEMP] missing_matrix_entry rank="
                        << Utilities::MPI::this_mpi_process(this->mpi_communicator)
                        << " row=" << row_dof << " requested_column=" << col_dof
                        << " actual_column=" << col_dof << " value=" << value
                        << std::endl;
              throw;
            }
          // SHARP-TEMPORARY-DEBUG-BOTTOM
        }
    }
}


template <int dim>
std::vector<Point<dim>>
FluidDynamicsSharp<dim>::get_periodic_particle_images(
  const Point<dim> &particle_position,
  const double      particle_radius) const
{
  return sharp_ib_periodic_boundaries.generate_periodic_image_points(
    particle_position,
    particle_radius,
    periodic_domain_lower,
    periodic_domain_upper);
}


template <int dim>
std::vector<Tensor<1, dim>>
FluidDynamicsSharp<dim>::get_periodic_particle_shifts(
  const Point<dim> &particle_position,
  const double      support_radius) const
{
  std::vector<Tensor<1, dim>> periodic_shifts(1, Tensor<1, dim>());

  if (!sharp_ib_periodic_boundaries.is_periodic_enabled())
    return periodic_shifts;

  for (const auto &image_position :
       get_periodic_particle_images(particle_position, support_radius))
    {
      periodic_shifts.push_back(image_position - particle_position);
    }

  return periodic_shifts;
}


template <int dim>
int
FluidDynamicsSharp<dim>::get_periodic_frame_index(
  const Tensor<1, dim> &periodic_shift) const
{
  if (periodic_shift.norm_square() == 0.0 ||
      !sharp_ib_periodic_boundaries.is_periodic_enabled())
    return 0;

  const unsigned int periodic_direction =
    sharp_ib_periodic_boundaries.get_periodic_direction();

  return static_cast<int>(
    std::lround(periodic_shift[periodic_direction] /
                (periodic_domain_upper - periodic_domain_lower)));
}


template <int dim>
void
FluidDynamicsSharp<dim>::set_particle_position_from_periodic_shift(
  const unsigned int    particle_id,
  const Point<dim>     &primary_particle_position,
  const Tensor<1, dim> &periodic_shift)
{
  Point<dim> shifted_position = primary_particle_position;
  shifted_position += periodic_shift;
  if (particles[particle_id].position != shifted_position)
    particles[particle_id].set_position(shifted_position);
}


template <int dim>
Tensor<1, dim>
FluidDynamicsSharp<dim>::get_periodic_shift_for_particle_frame(
  const CutCellInfo &cut_cell_info,
  const unsigned int particle_id) const
{
  for (const auto &particle_frame : cut_cell_info.cutting_particle_frames)
    {
      if (particle_frame.particle_id == particle_id)
        return particle_frame.periodic_shift;
    }

  AssertThrow(false,
              ExcMessage(
                "Missing periodic frame metadata for a Sharp-IB cut cell. "
                "The periodic shift used during cut-cell classification must "
                "be available during assembly and force evaluation."));
  return Tensor<1, dim>();
}


template <int dim>
double
FluidDynamicsSharp<dim>::particle_support_radius(
  const IBParticle<dim> &particle) const
{
  // Rotations preserve this enclosing radius. The center offset is essential
  // for composites whose reference point is not the bounding-box center.
  return sharp_ib_periodic_boundaries.enclosing_radius(*particle.shape);
}

template <int dim>
bool
FluidDynamicsSharp<dim>::PeriodicFrameDofCacheKey::operator<(
  const PeriodicFrameDofCacheKey &other) const
{
  return std::tie(dof_index, particle_id, periodic_frame_index) <
         std::tie(other.dof_index,
                  other.particle_id,
                  other.periodic_frame_index);
}

// Pre-compile the 2D and 3D versopm solver to ensure that the library is
// valid before we actually compile the final solver
template class FluidDynamicsSharp<2>;
template class FluidDynamicsSharp<3>;
