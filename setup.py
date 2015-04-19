from setuptools import setup


try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''

setup(name='py-artm',
      version='0.1.2',
      description='',
      long_description = long_description,
      classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords='',
      url='',
      author='Alexander Plavin',
      author_email='alexander@plav.in',
      license='MIT',
      packages=['py_artm', 'py_artm.plsa', 'py_artm.regularizers', 'py_artm.regularizers.coefficients', 'py_artm.quantities', 'py_artm.stop_conditions'],
      install_requires=[ 'numpy', 'scipy', 'numexpr', 'ipy-progressbar' ],
      zip_safe=False)
