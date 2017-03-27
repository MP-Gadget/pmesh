This directory contains a stripped version of GSL for random numbers.

We have to reimplement the GSL rng to preserve the exact GAGDET/NGEN-IC
3d white noise scheme. Because GSL is GPL, we directly embed the code
here to avoid the hassle. (pmesh is also GPL)

We avoid an explicit dependency on GSL by doing so.
