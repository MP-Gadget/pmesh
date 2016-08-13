import pywt
import numpy

def genwavelet(name):
    D = pywt.Wavelet(name)
    phi, psi, x = D.wavefun(level=8)
    phi = (phi[1:] + phi[:-1]) * 0.5
    i = 0
    while abs(phi[i]) < 2e-3:
        i += 1
    phi = phi[i:]

    i = len(phi)
    while abs(phi[i - 1]) < 2e-3:
        i = i - 1
    support = int(numpy.ceil(x[i]))

    i = (x < support).sum()
    phi = phi[:i //4 * 4 + 4]
    print(name, support)
    numbers = ["%.8f, %.8f, %.8f, %.8f" % tuple(a) for a in phi.reshape(-1, 4)]
    step = numpy.diff(x).mean()

    template = """
    static double _%(funcname)s_table[] = %(table)s;
    static double _%(funcname)s_nativesupport = %(support)g;
    static double _%(funcname)s_kernel(double x)
    {
        x += %(hsupport)g;

        int i = x / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d) return 0;
        return _%(funcname)s_table[i];
    }
    static double _%(funcname)s_diff(double x)
    {
        x += %(hsupport)g;

        int i = x / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d - 1) return 0;
        double f0 = _%(funcname)s_table[i];
        double f1 = _%(funcname)s_table[i + 1];
        return (f1 - f0) / %(step)e;
    }
    """
    return template % {
            'table' : "{" + ",\n".join(numbers) + "}",
            'hsupport' : support * 0.5,
            'support' : support,
            'funcname' : name,
            'step' : step,
            'tablesize' : len(phi),
    }

with open('pmesh/_window_wavelets.h', 'wt') as f:
    f.write(genwavelet('sym6'))
    f.write(genwavelet('sym12'))
    f.write(genwavelet('sym20'))
    f.write(genwavelet('db6'))
    f.write(genwavelet('db12'))
    f.write(genwavelet('db20'))
