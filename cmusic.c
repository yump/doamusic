#include <math.h>
#include <complex.h>
#include <cblas.h>
#include <assert.h>

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
        double complex zone = 1.0;
        double complex zzero = 0.0;

        // Scalar output of final dot product. Equal to 1/pmusic
        double complex invout;

        // Partition the workspace buffer
        double complex *propvec  = &work[0];   //propogation vec: sz 3
        double complex *steervec = &work[3];   //steering vector: sz N
        double complex *temp     = &work[3+N]; //intermediate:    sz N
        
        /*
         * generate the steering vector
         */
        propvec[0] = -I*sin(theta)*cos(phi); // x
        propvec[1] = -I*sin(theta)*sin(phi); // y
        propvec[2] = -I*cos(theta);          // z
        // steervec <- antennas * propvec
        cblas_zgemv(CblasRowMajor,
                    CblasTrans,
                    N,3,
                    &zone,antennas,3,
                    propvec,1,
                    &zzero,steervec,1);
        //convert from phase to steering vector exp(steervec) (j from propvec)
        for (int i=0; i<N; ++i){
                steervec[i] = cexp(steervec[i]);
        }

        /*
         * MUSIC pseudospectrum 1/(steer.H*metric*steer)
         */
        // temp <- metric * steervec
        cblas_zgemv(CblasRowMajor,
                    CblasTrans,
                    N,N,
                    &zone,metric,N,
                    steervec,1,
                    &zzero,temp,1);
        // invout <- steervec.H*temp
        cblas_zdotc_sub(N,
                        steervec,1,
                        temp,1,
                        &invout);

        //return
        return 1.0 / creal(invout);
}

                

