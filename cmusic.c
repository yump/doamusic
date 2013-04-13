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
#include <stdio.h>
#include <string.h>

double cpmusic(
               double complex *metric,                              // sz NxN
               double complex *antennas,                            // sz Nx3
               double complex *work, // workspace buffer               sz 2N+3
               size_t N,             // number of antennas
               double theta,         // inclination angle
               double phi            // azimuth angle
              )
{
        // Constants for scale factors passed to BLAS as pointers
        double complex one = 1.0;
        double complex zero = 0.0;

        // Scalar output of final dot product. Equal to 1/pmusic
        double complex invout;

        // Partition the workspace buffer.
        double complex *propvec  = &work[0];   //propogation vec: sz 3
        double complex *steervec = &work[3];   //steering vector: sz N
        double complex *temp     = &work[3+N]; //intermediate:    sz N
        
        /*
         * generate the steering vector
         */
        propvec[0] = -sin(theta)*cos(phi); // x
        propvec[1] = -sin(theta)*sin(phi); // y
        propvec[2] = -cos(theta);          // z
        // steervec <- antennas * propvec
        cblas_zgemv(CblasRowMajor,
                    CblasNoTrans,
                    N,3,&one,antennas,3, //3 because [x,y,z]
                    propvec,1,
                    &zero,steervec,1);
        //convert from phase to steering vector exp(j*steervec)
        for (int i=0; i<N; ++i){
                steervec[i] = cexp(I*steervec[i]);
        }

        /*
         * MUSIC pseudospectrum 1/(steer.H*metric*steer)
         */
        // temp <- metric * steervec
        cblas_zgemv(CblasRowMajor,
                    CblasNoTrans,
                    N,N,&one,metric,N,
                    steervec,1,
                    &zero,temp,1);
        // invout <- steervec.H*temp
        cblas_zdotc_sub(N,
                        steervec,1,
                        temp,1,
                        &invout);

#ifdef DEBUG
        printf(
                "Pv:<%g,%g,%g>\tSv:(%g+j%g)\ttemp:(%g+j%g)\tinvout:(%g+j%g)\n",
                creal(propvec[0]),creal(propvec[1]),creal(propvec[2]),
                creal(steervec[0]),cimag(steervec[0]),
                creal(temp[0]),cimag(temp[0]),
                creal(invout),cimag(invout)
                );
#endif
        //return
        return 1.0 / creal(invout);
}

                

