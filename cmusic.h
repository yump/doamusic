/*   Copyright 2013 Russell Haley
 *   (Please add yourself if you make changes)
 *
 *   This file is part of doamusic.
 *
 *   doamusic is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   doamusic is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with doamusic.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include <assert.h>

#ifndef CPMUSIC_H
#define CPMUSIC_H

double cpmusic(
               double complex *metric,                              // sz NxN
               double complex *antennas,                            // sz Nx3
               double complex *work, // workspace buffer               sz 2N+3
               size_t N,             // number of antennas
               double theta,         // inclination angle
               double phi            // azimuth angle
              );

#endif
