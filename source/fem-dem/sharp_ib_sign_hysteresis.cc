// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#include <fem-dem/sharp_ib_sign_hysteresis.h>

#include <deal.II/base/exceptions.h>

#include <cmath>
#include <iterator>
#include <utility>

template <int dim>
bool
SharpIBSignHysteresis<dim>::PointLess::operator()(
  const std::pair<unsigned int, dealii::Point<dim>> &a,
  const std::pair<unsigned int, dealii::Point<dim>> &b) const
{
  if (a.first != b.first)
    return a.first < b.first;
  for (unsigned int d = 0; d < dim; ++d)
    if (a.second[d] != b.second[d])
      return a.second[d] < b.second[d];
  return false;
}

template <int dim>
void
SharpIBSignHysteresis<dim>::reinit(const double new_band)
{
  AssertThrow(new_band >= 0. && std::isfinite(new_band),
              dealii::ExcMessage(
                "The Sharp-IB sign hysteresis band must be finite and "
                "non-negative."));
  band = new_band;
  pass = 0;
  memory.clear();
}

template <int dim>
void
SharpIBSignHysteresis<dim>::start_new_pass()
{
  ++pass;
  // Entries not refreshed during the previous pass can no longer be used;
  // removing them keeps the memory proportional to the points near surfaces.
  for (auto it = memory.begin(); it != memory.end();)
    it = (it->second.pass + 1 < pass) ? memory.erase(it) : std::next(it);
}

template <int dim>
bool
SharpIBSignHysteresis<dim>::is_inside(const unsigned int        particle_id,
                                      const dealii::Point<dim> &point,
                                      const double              levelset)
{
  const bool sign_inside = levelset <= 0.;
  // Points away from this surface, including evaluations in a distant periodic
  // frame, use the sign and leave the memory untouched.
  if (band <= 0. || std::abs(levelset) > 2. * band)
    return sign_inside;

  const auto key    = std::make_pair(particle_id, point);
  auto       entry  = memory.find(key);
  bool       inside = sign_inside;
  if (std::abs(levelset) <= band && entry != memory.end() &&
      entry->second.pass + 1 >= pass)
    inside = entry->second.inside;

  if (entry == memory.end())
    memory.emplace(key, Entry{inside, pass});
  else
    entry->second = Entry{inside, pass};
  return inside;
}

template class SharpIBSignHysteresis<2>;
template class SharpIBSignHysteresis<3>;
