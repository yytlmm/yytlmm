#import numpy
#from numpy.distutils.misc_util import Configuration

'''
def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    config.add_extension("_tree",
                         sources=["_tree.c"],
                         include_dirs=[numpy.get_include()])

    config.add_subpackage("tests")

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'dot_cython',
    sources=['_tree.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
