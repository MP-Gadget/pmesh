#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

static int
get_nbits(uint32_t v) {
/* https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel 
 * */
#define ONES(type) ((type)~(type)0)
    int c;
    v = v - ((v >> 1) & (ONES(uint32_t) /3));                           // temp
    v = (v & (ONES(uint32_t)/15*3)) + ((v >> 2) & (ONES(uint32_t)/15*3));      // temp
    v = (v + (v >> 4)) & (ONES(uint32_t)/255*15);                      // temp
    c = (uint32_t)(v * (ONES(uint32_t)/255)) >> ((sizeof(uint32_t) - 1) * 8); // count
    return c;
}

static ptrdiff_t
ipow(ptrdiff_t n, int p) {
    ptrdiff_t r = 1;
    while(p) {
        r *= n;
        p--;
    }
    return r;
}

static ptrdiff_t
absmax(int ndim, const ptrdiff_t x[])
{
    int d;
    if(ndim == 0) return 0;
    int shell = abs(x[0]);
    for(d = 0; d < ndim; d ++) {
        if(abs(x[d]) > shell) {
            shell = abs(x[d]);
        }
    }
    return shell;
}

/*
 * Returns the index in inside-out filling
 * order for integer mode vector x.
 *
 * cmask is a bitmask for axes that only the
 * non-negative half are saved. Usually it is 1 << (ndim-1).
 * or 0.
 *
 * max_length can terminate the calculation early. If the
 * index is g.e. max_length then the return value will be -1.
 *
 * -1 is returned if the mode does not exist. This
 * can happen when some axes have cmask set, or if the index
 * is g.e. max_length.
 *
 * */
ptrdiff_t pmesh_get_invariant_index(
    int ndim,
    const ptrdiff_t x[],
    const uint32_t cmask,
    ptrdiff_t max_length)
{
    int d;

    ptrdiff_t shell = absmax(ndim, x);
    /* (0, 0, 0....) */
    if (shell == 0) return 0;

    ptrdiff_t side = shell * 2 + 1; 

    /* partition the volume into pieces
     *
     * ((side-2) + 2)**ndim 
     * binomial expansion => 2**ndim capsets
     *
     * the capsets are indexed by icapset, an integer.
     * if there is a cut along d then (1<<d) is set.
     *
     * each capset is cut in n directions 
     * (n is number of active bits of axes)
     * each direction has two edges we can cut, pos or neg.
     * therefore we have 2**n caps in the capset;
     *
     * caps are indexed by icap. if the d-th axes is cut
     * along negative edge, then (1<<d) is set.
     * if icap has bits not set in icapset, then that icap
     * does not exist.
     *
     * if we look at the binomial expansion by n, it will
     * be like this: 
     *
     * C(ndim, 0) capsets x 2 ** 0 caps x size
     *      size = (side-2)**ndim
     *
     * C(ndim, 1) capsets x 2 ** 1 caps x size
     *      size = (side-2)**(ndim-1)
     *  ...
     * C(ndim, n) capsets x 2 ** n caps x size
     *      size = (side-2)**(ndim-n)
     *
     * but we do not iterate in this order; we iterate
     * first in icap, then in icapset.
     *
     * This is to ensure positive caps always come before negative
     * caps.
     *
     * */

    ptrdiff_t ind = 0;
    uint32_t icapset = 0;

    /* axes is a bit mask of which axes are projected
     * out in the expansion.
     * 0 is the inner region, and it is always iterated first.
     * this ensures the ordering is inside-out.
     * */
    
    /* which axes, and which edges did the query hit
     *
     * we will stop here.
     * then we will solve the subproblem in this cap.
     * */
    uint32_t icapset_x = 0;
    uint32_t icap_x = 0;

    for(d = 0; d < ndim; d ++) {
        /* if axis is compressed, negative side has no indices. */
        if((cmask & (1<<d)) && x[d] < 0) {
            return -1;
        }
    }

    /* the sub problem inside the hosting cap */
    ptrdiff_t x1[ndim];
    uint32_t cmask1 = 0;
    int ndim1;

    for(ndim1 = 0, d = 0; d < ndim; d ++) {
        if(abs(x[d]) == shell) {
            icapset_x |= (1<<d);
            icap_x |= (x[d] < 0) * (1<<d);
        } else {
            x1[ndim1] = x[d];
            cmask1 |= (1 << ndim1) * ((cmask >> d) & 1);
            ndim1 ++;
        }
    }

    int ncapsets = 1 << ndim;
    int ncapsmax = 1 << ndim;
    int icap;

    /* caches */
    ptrdiff_t size[ncapsets];
    memset(size, 0, sizeof(size[0]) * ncapsets);

    /* advance to the capset hosting the query */
    /* starting from all positive caps */
    for(icap = 0; icap < ncapsmax; icap ++) {
        for(icapset = 0; icapset < ncapsets; icapset++ ) {
            /* no such cap in this capset */
            if(icap & ~icapset) continue;
            /* cap landed in a ignored negative plane */
            if(icap & cmask) continue;

            /* hit the hosting cap */
            if(icap == icap_x && icapset == icapset_x) {
                /* subproblem */
                ptrdiff_t max_length1 = -1;
                if (max_length >= 0) {
                    max_length1 = max_length - ind;
                }
                ptrdiff_t subind = pmesh_get_invariant_index(ndim1, x1, cmask1, max_length1);
                if(subind == -1) {
                    return -1;
                }
                ind += subind;
                if(max_length >= 0 && ind >= max_length) {
                    return -1;
                }
                return ind;
            }
            /* advance to the end of the cap */
            if(size[icapset] == 0) {
                /* cache precomputed */
                int n = get_nbits(icapset);
                int nhalf = get_nbits(cmask & ~icapset);
                size[icapset] = ipow(side - 2, ndim - n - nhalf) * ipow(shell, nhalf);
            }
            ind += size[icapset];
            if(max_length >= 0 && ind >= max_length) {
                return -1;
            }
        }
    }
    return -1;
}

#ifdef PMESH_INVARIANT_DEBUG

#include <assert.h>
#define P(x, ...) ((ptrdiff_t []) { x, __VA_ARGS__ })

static void
visual_2d(int n)
{
    int i, j;
    int max = ipow(2*n, 2);
    char map[max];
    memset(map, 0, max);

#define INNERLOOP() \
    for(j = -n + 1; j <= n; j ++)

    printf("% 3s |", "");
    INNERLOOP() {
        printf("% 3d", j); \
    }
    printf("\n");

    for(i = -n + 1; i <= n; i ++) {
        printf("% 3d |", i);    
        INNERLOOP() {
            int r = pmesh_get_invariant_index(2, P(i, j), 0, -1);
            assert (r >= 0);
            assert (r < max);
            assert (map[r] == 0);
            map[r] = 1;
            printf("% 3d", r);
        }
        printf("\n");
    }
#undef INNERLOOP
}

static void
visual_2dc(int n)
{
    int i, j;
    int max = 2*n * (n + 1);
    char map[max];
    memset(map, 0, max);

#define INNERLOOP() \
    for(j = 0; j <= n; j ++)

    printf("% 3s |", "");
    INNERLOOP() {
        printf("% 3d", j); \
    }
    printf("\n");

    for(i = -n + 1; i <= n; i ++) {
        printf("% 3d |", i);    
        INNERLOOP() {
            int r = pmesh_get_invariant_index(2, P(i, j), 1, -1);
/*
            assert (r >= 0);
            assert (r < max);
            assert (map[r] == 0);
            map[r] = 1;
*/
            printf("% 3d", r);
        }
        printf("\n");
    }
#undef INNERLOOP
}

static void
visual_3d(int n)
{
    int i; int j; int k;
    int max = ipow(2*n, 3);
    char map[max];
    memset(map, 0, max);

#define INNERLOOP_START \
    for(j = -n + 1; j <= n; j ++) { \
        printf("% 2s", "/"); \
        for(k = -n + 1; k <= n; k ++) {

#define INNERLOOP_END \
        } \
        printf("% 2s", "/"); \
    } 

    printf("% 4s |", "j");
    INNERLOOP_START
        printf("% 3d", j);
    INNERLOOP_END
    printf("\n");
    printf("% 4s |", "k");
    INNERLOOP_START
        printf("% 3d", k);
    INNERLOOP_END
    printf("\n");

    for(i = -n + 1; i <= n; i ++) {
        printf("% 4d |", i);    
        INNERLOOP_START
        int r = pmesh_get_invariant_index(3, P(i, j, k), 0, -1);
        assert ( r >= 0);
        assert ( r <= max );
        assert ( map[r] == 0);
        map[r] = 1;
        printf("% 3d", r);
        INNERLOOP_END
        printf("\n");
    }
#undef INNERLOOP
}

static void
test_nd(int ndim, int n)
{
    printf("testing ndim = %d, n = %d\n", ndim, n);
    ptrdiff_t max = ipow(2*n, ndim);
    char map[max];
    memset(map, 0, max);
    ptrdiff_t x[ndim];
    uint32_t cmask = 0;
    ptrdiff_t i;
    int d;
    for(i = 0; i < max; i ++) {
        int ii = i;
        for(d = 0; d < ndim; d++) {
            x[d] = ii % (2*n) - (n - 1);
            ii /= (2*n);
        }
        int ind = pmesh_get_invariant_index(ndim, x, cmask, -1);
        assert (ind >= 0);
        assert (ind <= max);
        assert (map[ind] == 0);
        map[ind] = 1;
    }
}

static void
test_ndc(int ndim, int n, int cdim)
{
    printf("testing c ndim = %d, n = %d\n", ndim, n);
    ptrdiff_t max = ipow(2*n, ndim - 1) * (n + 1);
    char map[max];
    memset(map, 0, max);
    ptrdiff_t x[ndim];
    uint32_t cmask = 0;
    ptrdiff_t i;
    int d;
    for(i = 0; i < max; i ++) {
        int ii = i;
        for(d = 0; d < ndim; d++) {
            if(d != cdim) {
                x[d] = ii % (2*n) - (n - 1);
                ii /= (2*n);
            } else {
                x[d] = ii % (n + 1);
                ii /= (n + 1);
                cmask |= (1 << d);
            }
        }
        int ind = pmesh_get_invariant_index(ndim, x, cmask, -1);
        assert (ind >= 0);
        assert (ind <= max);
        assert (map[ind] == 0);
        map[ind] = 1;
    }
}

static void
check_index(ptrdiff_t expected, int ndim, const ptrdiff_t x[], uint32_t cmask)
{
    char xs[1000] = {0};
    char cs[1000] = {0};
    int d;
    for(d = 0; d < ndim; d ++) {
        sprintf(xs + strlen(xs), "% 2td ", x[d]);
        sprintf(cs + strlen(cs), "% 2td ", (cmask >>d) & 1);
    }
    ptrdiff_t value = pmesh_get_invariant_index(ndim, x, cmask, -1);
    char * status;
    if (value == expected) {
        status = "PASS";
    } else {
        status = "FAIL";
    }
    printf("[ %s ] ndim = %d, x = [%s], c = [%s], index = %td, expected = %td\n",
        status, 
        ndim, xs, cs, value, expected);
}

void
pmesh_test_get_invariant_index()
{
    assert (get_nbits(0) == 0);
    assert (get_nbits(1) == 1);
    assert (get_nbits(3) == 2);
    assert (get_nbits(0xffffffff) == 32);

    check_index(0, 0, P(0), 0);
    printf("----- test 1d no compresss -----\n");
    check_index(0, 1, P(0), 0);
    check_index(1, 1, P(1), 0);
    check_index(2, 1, P(-1), 0);
    check_index(3, 1, P(2), 0);
    check_index(4, 1, P(-2), 0);

    printf("----- test 1d compresss -----\n");
    check_index(0, 1, P(0), 1);
    check_index(1, 1, P(1), 1);
    check_index(2, 1, P(2), 1);

    visual_2dc(4);
    visual_2d(4);
    test_nd(1, 4);
    test_ndc(1, 4, 0);
    test_nd(2, 4);
    test_ndc(2, 4, 0);
    test_ndc(2, 4, 1);
    test_nd(3, 4);
    test_ndc(3, 4, 0);
    test_ndc(3, 4, 1);
    test_ndc(3, 4, 2);
    test_nd(4, 4);
    test_ndc(4, 4, 0);
    test_ndc(4, 4, 1);
    test_ndc(4, 4, 2);
    test_ndc(4, 4, 3);
    test_nd(3, 64);
}

int main() {
    pmesh_test_get_invariant_index();
    return 0;
}
#endif
