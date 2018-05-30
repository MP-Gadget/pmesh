static int
mkname(_has_mode)(PMeshWhiteNoiseGenerator * self, ptrdiff_t * iabs)
{
    ptrdiff_t irel[3];
    int d;
    irel[2] = iabs[2] - self->start[2];
    if(irel[2] >= 0 && irel[2] < self->size[2]) return 1;
    return 0;
}

static void
mkname(_set_mode)(PMeshWhiteNoiseGenerator * self, ptrdiff_t * iabs, char * delta_k, FLOAT re, FLOAT im)
{
    ptrdiff_t ip = 0;
    ptrdiff_t irel[3];
    int d;
    for(d = 0; d < 3; d ++) {
        irel[d] = iabs[d] - self->start[d];
        ip += self->strides[d] * irel[d];
    }

    if(irel[2] >= 0 && irel[2] < self->size[2]) {

        ((FLOAT*) (delta_k + ip))[0] = re;
        ((FLOAT*) (delta_k + ip))[1] = im;
    }
}

static void
mkname(_generic_fill)(PMeshWhiteNoiseGenerator * self, void * delta_k, int seed)
{

#if 0
    clock_t start, end;
    double cpu_time_used;

    start = clock();
#endif
    /* Fill delta_k with gadget scheme */
    int i, j, k;

    int signs[3];

    {
        int compressed = 1;
        ptrdiff_t iabs[3] = {self->start[0], self->start[1], 0};

        /* if no negative k modes are requested, do not work with negative sign;
         * this saves half of the computing time. */

        for(k = self->Nmesh[2] / 2 + 1; k < self->Nmesh[2]; k ++) {
            iabs[2] = k;
            if (mkname(_has_mode)(self, iabs)) {
                compressed = 0;
                break;
            }
        }
        // printf("compressed = %d\n", compressed);
        if (compressed) {
            /* only half of the fourier space is requested, ignore the conjugates */
            signs[0] = 1;
            signs[1] = 0;
            signs[2] = 0;
        } else {
            /* full fourier space field is requested */
            /* do negative then positive. ordering is import to makesure the positive overwrites nyquist. */
            signs[0] = -1;
            signs[1] = 1;
            signs[2] = 0;
        }
    }

    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(rng, seed);

    for(i = 0; i < self->Nmesh[0] / 2; i++) {
        for(j = 0; j < i; j++)
            SETSEED(self, i, j, rng);
        for(j = 0; j < i + 1; j++)
            SETSEED(self, j, i, rng);
        for(j = 0; j < i; j++)
            SETSEED(self, self->Nmesh[0] - 1 - i, j, rng);
        for(j = 0; j < i + 1; j++)
            SETSEED(self, self->Nmesh[1] - 1 - j, i, rng);
        for(j = 0; j < i; j++)
            SETSEED(self, i, self->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++)
            SETSEED(self, j, self->Nmesh[0] - 1 - i, rng);
        for(j = 0; j < i; j++)
            SETSEED(self, self->Nmesh[0] - 1 - i, self->Nmesh[1] - 1 - j, rng);
        for(j = 0; j < i + 1; j++)
            SETSEED(self, self->Nmesh[1] - 1 - j, self->Nmesh[0] - 1 - i, rng);
    }
    gsl_rng_free(rng);

#if 0
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("time used in seeds = %g\n", cpu_time_used);

    start = end;
#endif
    ptrdiff_t skipped = 0;
    ptrdiff_t used = 0;

    for(i = self->start[0];
        i < self->start[0] + self->size[0];
        i ++) {

        gsl_rng * lower_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng * this_rng = gsl_rng_alloc(gsl_rng_ranlxd1);

        int ci = self->Nmesh[0] - i;
        if(ci >= self->Nmesh[0]) ci -= self->Nmesh[0];

        for(j = self->start[1];
            j < self->start[1] + self->size[1];
            j ++) {
            /* always pull the whitenoise from the lower quadrant plane for k = 0
             * plane and k == Nmesh / 2 plane*/
            int d1 = 0, d2 = 0;
            int cj = self->Nmesh[1] - j;
            if(cj >= self->Nmesh[1]) cj -= self->Nmesh[1];

            /* d1, d2 points to the conjugate quandrant */
            if( (ci == i && cj < j)
             || (ci < i && cj != j)
             || (ci < i && cj == j)) {
                d1 = 1;
                d2 = 1;
            }

            int isign;
            for(isign = 0; signs[isign] != 0; isign ++) {
                int sign = signs[isign];

                unsigned int seed_lower, seed_this;

                /* the lower quadrant generator */
                seed_lower = GETSEED(self, i, j, d1, d2);
                gsl_rng_set(lower_rng, seed_lower);

                if(sign == 1) {
                    seed_this = GETSEED(self, i, j, 0, 0);
                } else {
                    /* the negative half of k, sample from the conjugate quadrant */
                    seed_this = GETSEED(self, i, j, 1, 1);
                }
                gsl_rng_set(this_rng, seed_this);

                for(k = 0; k <= self->Nmesh[2] / 2; k ++) {
                    int use_conj = (d1 != 0 || d2 != 0) && (k == 0 || k == self->Nmesh[2] / 2);

                    double ampl, phase;
                    if(use_conj) {
                        /* on k = 0 and Nmesh/2 plane, we use the lower quadrant generator, 
                         * then hermit transform the result if it is nessessary */
                        SAMPLE(this_rng, &ampl, &phase);
                        SAMPLE(lower_rng, &ampl, &phase);
                    } else {
                        SAMPLE(lower_rng, &ampl, &phase);
                        SAMPLE(this_rng, &ampl, &phase);
                    }

                    ptrdiff_t iabs[3] = {i, j, k};

                    /* mode is not there, skip it */
                    if(!mkname(_has_mode)(self, iabs)) {
                        skipped ++;
                        continue;
                    } else {
                        used ++;
                    }

                    /* we want two numbers that are of std ~ 1/sqrt(2) */
                    if(self->unitary) {
                        /* Unitary gaussian, the norm of real and imag is fixed to 1/sqrt(2) */
                        ampl = 1.0;
                    } else {
                        /* box-mueller */
                        ampl = sqrt(- log(ampl));
                    }

                    FLOAT re = ampl * cos(phase);
                    FLOAT im = ampl * sin(phase);

                    /*
                    if(use_conj) {
                        printf("%d %d %d %d useconj=%d %d %d seed %d (otherseed %d) %g %g\n", i, j, k, sign, use_conj, d1, d2, seed_lower, seed_this, re, im);
                    } else {
                        printf("%d %d %d %d useconj=%d %d %d seed %d (otherseed %d) %g %g\n", i, j, k, sign, use_conj, d1, d2, seed_this, seed_lower, re, im);
                    }
                    */

                    if(sign == -1) {
                        iabs[2] = self->Nmesh[2] - k;
                        im = - im;
                    }

                    if(use_conj) {
                        im *= -1;
                    }

                    if((self->Nmesh[0] - iabs[0]) % self->Nmesh[0] == iabs[0] &&
                       (self->Nmesh[1] - iabs[1]) % self->Nmesh[1] == iabs[1] &&
                       (self->Nmesh[2] - iabs[2]) % self->Nmesh[2] == iabs[2]) {
                        /* The mode is self conjuguate, thus imaginary mode must be zero */
                        im = 0;
                        if(self->unitary)  /* real part must be 1 then*/
                            re = 1;
                    }

                    if(iabs[0] == 0 && iabs[1] == 0 && iabs[2] == 0) {
                        /* the mean is zero */
                        re = im = 0;
                    }

                    mkname(_set_mode)(self, iabs, delta_k, re, im);

                }
            }
        }
        gsl_rng_free(lower_rng);
        gsl_rng_free(this_rng);
    }
#if 0
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("time used in fill = %g\n", cpu_time_used);
    printf("skipped = %td used = %td\n", skipped, used);
#endif
}

/* Footnotes */ 

/* 1): 
 * We want delta(k) = delta_real + I delta_imag, where delta_real and
 * delta_imag are WhiteNoise random variables with variance given by
 * power spectrum, \sigma^2=P(k). We can obtain this equivalently as
 *
 *   delta(k) = A exp(i phase),
 *
 * where the phase is random (i.e. sampled from a uniform distribution)
 * and the amplitude A follows a Rayleigh distribution (see 
 * https://en.wikipedia.org/wiki/Rayleigh_distribution). To sample from 
 * Rayleigh distribution, use inverse transform sampling
 * (see https://en.wikipedia.org/wiki/Inverse_transform_sampling), i.e.
 * start from uniform random variable in [0,1] and then apply inverse of CDF
 * of Rayleigh distribution. From F(A)=CDF(A)=1-e^{-A^2/(2\sigma^2)} we get
 * A = \sigma \sqrt{-2 ln(1-CDF)}. So if x is uniform random number in [0,1], then 
 * A = \sigma \sqrt(-2 ln(x)) follows Rayleigh distribution as desired. 
 * Here we used x instead of 1-x because this does not make a difference for a 
 * uniform random number in [0,1]. In the code below, we start with \sigma=1 and 
 * multiply by sqrt(P(k)) later.
 */
