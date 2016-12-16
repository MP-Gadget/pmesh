static void
mkname(_generic_fill)(PMeshWhiteNoiseGenerator * self, void * delta_k, int seed)
{
    /* Fill delta_k with gadget scheme */
    int d;
    int i, j, k;

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

    ptrdiff_t irel[3];
    for(i = self->start[0];
        i < self->start[0] + self->size[0];
        i ++) {

        gsl_rng * random_generator0 = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng * random_generator1 = gsl_rng_alloc(gsl_rng_ranlxd1);

        for(j = self->start[1];
            j < self->start[1] + self->size[1];
            j ++) {
            /* always pull the whitenoise from the lower quadrant plane for k = 0
             * plane*/
            int hermitian = 0;
            int d1 = 0, d2 = 0;

            if(i == 0) {
                if(j > self->Nmesh[1] / 2) {
                    hermitian = 1; 
                    d2 = 1;
                } 
            } else {
                if(i > self->Nmesh[0] / 2) {
                    hermitian = 1;
                    d1 = 1;
                    d2 = j != 0;
                }  else {
                    /* no transpose */
                    d1 = d2 = 0;
                }
            } 

            unsigned int seed;
            seed = GETSEED(self, i, j, d1, d2);
            gsl_rng_set(random_generator0, seed);

            /* the lower quadrant generator */
            seed = GETSEED(self, i, j, 0, 0);
            gsl_rng_set(random_generator1, seed);

            /* this black magic matches two generators. */
            double skip = gsl_rng_uniform(random_generator1);
            do skip = gsl_rng_uniform(random_generator1);
            while(skip == 0);

            for(k = 0; k <= self->Nmesh[2] / 2; k ++) {
                gsl_rng * random_generator = k?random_generator1:random_generator0;
                /* on k = 0 plane, we use the lower quadrant generator, 
                 * then hermit transform the result if it is nessessary */
                double phase = gsl_rng_uniform(random_generator) * 2 * M_PI;
                double ampl = 0;
                do ampl = gsl_rng_uniform(random_generator); while(ampl == 0);

                ptrdiff_t iabs[3] = {i, j, k};
                ptrdiff_t ip = 0;
                for(d = 0; d < 3; d ++) {
                    irel[d] = iabs[d] - self->start[d];
                    ip += self->strides[d] * irel[d];
                }

                if(irel[2] < 0) continue;
                if(irel[2] >= self->size[2]) continue;

                /* we want two numbers that are of std ~ 1/sqrt(2) */
                ampl = sqrt(- log(ampl));

                if(iabs[0] == 0 && iabs[1] == 0 && iabs[2] == 0) {
                    /* the mean is zero */
                    ampl = 0;
                }

                ((FLOAT*) (delta_k + ip))[0] = ampl * cos(phase);
                ((FLOAT*) (delta_k + ip))[1] = ampl * sin(phase);

                if(hermitian && k == 0) {
                    ((FLOAT*) (delta_k + ip))[1] *= -1;
                }

                if((self->Nmesh[0] - iabs[0]) % self->Nmesh[0] == iabs[0] &&
                   (self->Nmesh[1] - iabs[1]) % self->Nmesh[1] == iabs[1] &&
                   (self->Nmesh[2] - iabs[2]) % self->Nmesh[2] == iabs[2]) {
                    /* The mode is self conjuguate, thus imaginary mode must be zero */
                    ((FLOAT*) (delta_k + ip))[1] = 0;
                }
            }
        }
        gsl_rng_free(random_generator0);
        gsl_rng_free(random_generator1);
    }
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
