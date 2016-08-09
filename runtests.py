from mpi4py_test import MPITester
import sys
import os.path

tester = MPITester(os.path.abspath(__file__), "pmesh")

tester.main(sys.argv[1:])
