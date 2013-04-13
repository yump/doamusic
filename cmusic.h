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
