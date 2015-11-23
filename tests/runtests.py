from __future__ import absolute_import

import sys
import os

from numpy.testing import Tester

from sys import argv

tester = Tester()
result = tester.test(extra_argv=['-w', '.'] + argv[1:])

if not result.wasSuccessful():
    raise Exception("Test Failed")
