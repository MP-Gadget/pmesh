import numpy
from .window import FindResampler

def lic(vectors, kernel, length, ds, resampler=None, texture=None, normalize=True):
    """ Line Intergral Convolution for visualizing vector fields.

        The vectors must be normalized to norm of 1.0 for LIC to give a reasonable
        image.

        The image looks bad if the vectors are noisy.

        Parameters
        ----------
        vectors : list of RealField
            vx, vy, vz, ... must be normalized.

        kernel : function kernel(s)
            the line integral kernel function. s is the line coordinate variable between
            -1 and 1.

        length : float
            the length of the line, in pixels.

        ds : float
            step size in the line integration. in pixels.

        resampler: string, ResamplerWindow
            the resampler window. See pmesh.window module for a full list. if None, use pm's
            default sampler, pm.resampler, where pm is infered from the first vector.

        texture : RealField, or None
            the texture to use. If None, a default gaussian texture is used.

        normalize : bool
            True if normalize the vectors to 1.0

        Returns
        -------
        lic : RealField
            the integration result.
    """
    pm = vectors[0].pm

    if normalize:
        vabs = sum(vi**2 for vi in vectors) ** 0.5
        mask = vabs[...] == 0.0
        vabs[mask] = 1.0
        vectors = [vi / vabs for vi in vectors]

    if texture is None:
        texture = pm.generate_whitenoise(seed=990919, type='real')

    Q = pm.generate_uniform_particle_grid(shift=0.0)

    if resampler is None:
        resampler = pm.resampler

    resampler = FindResampler(resampler)

    f = texture.readout(Q, resampler='nearest')
    vmax = max(abs(v[...]).max() for v in vectors)

    for sign in [-1, +1]:
        x = Q.copy()
        s = 0
        while s < length * 0.5:
            k = kernel(s * sign / (length * 0.5))
            dx = x * 0.0
            layout = pm.decompose(x, smoothing=vmax * ds * 0.5 + resampler.support * 0.5)
            for d, v in enumerate(vectors):
                dx[..., d] = v.readout(x, layout=layout, resampler=resampler) * ds
            x[...] += dx * 0.5 * sign
            f[...] += texture.readout(x, layout=layout, resampler=resampler) * k * ds
            x[...] += dx * 0.5 * sign

            s += ds

    return pm.paint(Q, f, resampler='nearest')
