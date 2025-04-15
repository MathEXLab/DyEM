"""pypardi init"""
__all__ = ['attractors', 'global_indices', 'local_indices', 'utils']

import os
import sys
PACKAGE_PARENTS = ['.', '..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))

__project__ = 'PyParDI'
__title__ = "PyParDI"
__author__ = "Adriano Gualandi, Gianmarco Mengaldo"
__email__ = 'adriano.gualandi@ingv.it, mpegim@nus.edu.sg'
__copyright__ = "Copyright 2020-2030 PyParDI authors and contributors"
__maintainer__ = __author__
__status__ = "Stable"
__license__ = "MIT"
__version__ = "0.0.1"
__url__ = "https://github.com/mathe-lab/PyParDI"
