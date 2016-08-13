import numpy

def lanczos(n):
    x = numpy.linspace(0, n, 8192, endpoint=False)
    phi = numpy.sinc(x) * numpy.sinc(x/n)
    sum = 2 * numpy.trapz(phi, x)
    phi /= sum
    return phi, x

def genlanczos(n):
    phi, x = lanczos(n)
    name = 'lanczos%d' % n
    support = 2 * n

    numbers = ["%.8f, %.8f, %.8f, %.8f" % tuple(a) for a in phi.reshape(-1, 4)]
    step = numpy.diff(x).mean()

    template = """
    static double _%(funcname)s_table[] = %(table)s;
    static double _%(funcname)s_nativesupport = %(support)g;
    static double _%(funcname)s_kernel(double x)
    {
        int i = fabs(x) / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d) return 0;
        return _%(funcname)s_table[i];
    }
    static double _%(funcname)s_diff(double x)
    {
        double factor;
        if(x >= 0) {
            factor = 1;
        } else {
            factor = -1;
            x = -x;
        }
        
        int i = x / %(step)e;
        if (i < 0) return 0;
        if (i >= %(tablesize)d - 1) return 0;
        double f0 = _%(funcname)s_table[i];
        double f1 = _%(funcname)s_table[i + 1];
        return factor * (f1 - f0) / %(step)e;
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

with open('pmesh/_window_lanczos.h', 'wt') as f:
    f.write(genlanczos(2))
    f.write(genlanczos(3))
