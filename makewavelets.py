import pywt
import numpy

def genfunc(name, support):
    D = pywt.Wavelet(name)
    phi, psi, x = D.wavefun(level=8)
    m = x < support
    x = x[m]
    phi = phi[m]

    numbers = ["%.8f, %.8f, %.8f, %.8f" % tuple(a) for a in phi.reshape(-1, 4)]
    step = numpy.diff(x).mean()

    template = """
    static double _%(funcname)s_table[] = %(table)s;
    static double _%(funcname)s_kernel(double x, double hinv)
    {
        x *= %(hsupport)g * hinv;
        x += %(hsupport)g;

        int i = x / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d) return 0;
        return _%(funcname)s_table[i];
    }
    static double _%(funcname)s_diff(double x, double hinv)
    {
        x *= %(hsupport)g * hinv;
        x += %(hsupport)g;

        int i = x / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d - 1) return 0;
        double f0 = _%(funcname)s_table[i];
        double f1 = _%(funcname)s_table[i + 1];
        return (f1 - f0) / %(step)e * %(hsupport)g * hinv;
    }
    """
    return template % {
            'table' : "{" + ",\n".join(numbers) + "}",
            'hsupport' : support * 0.5,
            'funcname' : name,
            'step' : step,
            'tablesize' : len(phi),
    }

with open('pmesh/_window_wavelets.h', 'wt') as f:
    f.write(genfunc('db12', 6))
    f.write(genfunc('db20', 8))
