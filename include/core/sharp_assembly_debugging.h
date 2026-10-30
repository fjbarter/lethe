// SHARP-ASSEMBLY-DEBUGGING-TOP
// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
#ifndef lethe_sharp_assembly_debugging_h
#  define lethe_sharp_assembly_debugging_h

#  include <deal.II/base/mpi.h>

#  include <array>
#  include <chrono>
#  include <exception>
#  include <iomanip>
#  include <iostream>
#  include <sstream>

// Temporary, single-thread instrumentation. It is active only while the calling
// thread executes sharp_edge(); ordinary grid searches are not recorded.
namespace SharpAssemblyDebugging
{
  using Clock = std::chrono::steady_clock;
  enum Metric : unsigned int
  {
    total,
    neighbors,
    tree,
    failed_tree,
    periodic_retry,
    clear_row,
    matrix_add,
    compression,
    level_cells,
    child_cells,
    neighbor_cells,
    cache_hits,
    cache_misses,
    index_build,
    index_query,
    index_candidates,
    geometry_cache_hits,
    geometry_cache_misses,
    metric_count
  };
  struct Measurements
  {
    bool                             active = false;
    std::array<double, metric_count> seconds{};
    std::array<double, metric_count> calls{};
  };
  inline thread_local Measurements  measurements;
  inline thread_local unsigned long assembly_number = 0;

  inline void
  count(const Metric metric)
  {
    if (measurements.active)
      ++measurements.calls[metric];
  }

  // Timers are inclusive: e.g. neighbor time includes its fallback tree search,
  // and periodic-retry time includes its tree search. Do not sum those rows.
  class Scope
  {
  public:
    explicit Scope(const Metric metric)
      : metric(metric)
      , active(measurements.active)
      , exceptions(std::uncaught_exceptions())
      , start(active ? Clock::now() : Clock::time_point{})
    {}
    ~Scope()
    {
      finish();
    }
    void
    finish()
    {
      if (!active)
        return;
      active = false;
      const double elapsed =
        std::chrono::duration<double>(Clock::now() - start).count();
      measurements.seconds[metric] += elapsed;
      ++measurements.calls[metric];
      if (metric == tree && std::uncaught_exceptions() > exceptions)
        {
          measurements.seconds[failed_tree] += elapsed;
          ++measurements.calls[failed_tree];
        }
    }

  private:
    Metric            metric;
    bool              active;
    int               exceptions;
    Clock::time_point start;
  };

  class Assembly
  {
  public:
    Assembly(const MPI_Comm     communicator,
             const unsigned int step,
             const double       time)
      : communicator(communicator)
      , start(Clock::now())
    {
      measurements        = {};
      measurements.active = true;
      ++assembly_number;
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
        {
          std::ostringstream line;
          line << "[SHARP-ASSEMBLY-DEBUGGING] BEGIN assembly="
               << assembly_number << " step=" << step
               << " time=" << std::setprecision(10) << time;
          std::cout << line.str() << std::endl;
        }
    }
    ~Assembly()
    {
      // No collectives on exception unwinding: another rank may still be busy.
      measurements.active = false;
    }
    void
    report()
    {
      measurements.seconds[total] =
        std::chrono::duration<double>(Clock::now() - start).count();
      measurements.calls[total]     = 1.;
      measurements.active           = false;
      constexpr unsigned int   size = 2 * metric_count;
      std::array<double, size> local{}, minimum{}, sum{}, maximum{};
      for (unsigned int i = 0; i < metric_count; ++i)
        {
          local[i]                = measurements.seconds[i];
          local[metric_count + i] = measurements.calls[i];
        }
      // Three reductions per completed assembly, never per point or cell.
      // Report overhead is excluded from the measured total.
      MPI_Reduce(local.data(),
                 minimum.data(),
                 size,
                 MPI_DOUBLE,
                 MPI_MIN,
                 0,
                 communicator);
      MPI_Reduce(
        local.data(), sum.data(), size, MPI_DOUBLE, MPI_SUM, 0, communicator);
      MPI_Reduce(local.data(),
                 maximum.data(),
                 size,
                 MPI_DOUBLE,
                 MPI_MAX,
                 0,
                 communicator);
      if (dealii::Utilities::MPI::this_mpi_process(communicator) != 0)
        return;
      const double ranks =
        dealii::Utilities::MPI::n_mpi_processes(communicator);
      const std::array<const char *, metric_count> names = {
        {"total",
         "neighbor_search_inclusive",
         "tree_search_inclusive",
         "failed_tree_subset",
         "periodic_retry_inclusive",
         "matrix_clear_row",
         "matrix_add",
         "compression",
         "level_cells_examined",
         "child_cells_examined",
         "neighbor_cells_examined",
         "stencil_cache_hits",
         "stencil_cache_misses",
         "spatial_index_build",
         "spatial_index_query",
         "spatial_index_candidates",
         "geometry_cache_hits",
         "geometry_cache_misses"}};
      std::ostringstream out;
      out << "[SHARP-ASSEMBLY-DEBUGGING] END assembly=" << assembly_number
          << " ranks=" << static_cast<unsigned int>(ranks)
          << " (inclusive times; cell/cache rows are counts only)\n";
      out << std::scientific << std::setprecision(6);
      for (unsigned int i = 0; i < metric_count; ++i)
        out << "[SHARP-ASSEMBLY-DEBUGGING] " << names[i]
            << " seconds[min/mean/max]=" << minimum[i] << '/' << sum[i] / ranks
            << '/' << maximum[i]
            << " count[min/mean/max]=" << minimum[metric_count + i] << '/'
            << sum[metric_count + i] / ranks << '/' << maximum[metric_count + i]
            << '\n';
      std::cout << out.str() << std::flush;
    }

  private:
    MPI_Comm          communicator;
    Clock::time_point start;
  };
} // namespace SharpAssemblyDebugging
#endif
// SHARP-ASSEMBLY-DEBUGGING-BOTTOM
