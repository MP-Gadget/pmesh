"""
.. deprecated:: 0.1

"""
import warnings
warnings.warn("tools.py is deprecated", DeprecationWarning)
from mpi4py import MPI

class Rotator(object):
    def __init__(self, comm):
        self.comm = comm
    def __enter__(self):
        self.comm.Barrier()
        for i in range(self.comm.rank):
            self.comm.Barrier()
    def __exit__(self, type, value, tb):
        for i in range(self.comm.rank, self.comm.size):
            self.comm.Barrier()
        self.comm.Barrier()
def FromRoot(comm):
    def decorator(func):
        def wrapped(*args, **kwargs):
            if comm.rank == 0:
                rt = func(*args, **kwargs)
            else:
                rt = None
            rt = comm.bcast(rt)
            return rt
        return wrapped
    return decorator

class Timer(object):
    def __init__(self, comm):
        self.comm = comm
        self.t0 = 1.0 * MPI.Wtime()
        self.spent = 0.
    def __enter__(self):
        self.t0 = 1.0 * MPI.Wtime()

    def __exit__(self, *args, **kwargs):
        t1 = 1.0 * MPI.Wtime()
        self.spent += t1 - self.t0
class Timers(dict):
    def __init__(self, comm=None):
        self.comm = comm
    def __getitem__(self, key):
        if not dict.__contains__(self, key):
            self[key] = Timer(self.comm)
        return dict.__getitem__(self, key)
    def __str__(self):
        return '\n'.join(['%s: %g' % (key, self[key].spent) for key in self])
