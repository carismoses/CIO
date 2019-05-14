try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='CIO',
    version='1.0',
    description='Contact Invariant Optimization',
    url='https://github.com/carismoses/CIO',
    author='Caris Moses',
    author_email='carism@mit.edu',
    packages=['cio',],
    python_requires='>3.0',
    install_requires=[
         'imageio',
         'scipy',
         'jupyter',
         'matplotlib'
      ],
    long_description=open('README.md').read(),
)
