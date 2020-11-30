from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='plssmooth',
    version='0.1',
    packages=find_packages(include=['plssmooth', 'plssmooth.*']),
    url='https://github.com/JulianAustin1993/plssmooth',
    license='',
    author='Julian Austin',
    author_email='J.Austin3@newcastle.ac.uk',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Natural Language :: English',
    ],
    keywords='smooth',
    install_requires=[
        'numpy',
        'scipy'
    ],
    tests_require=['pytest',
                   'numpy',
                   'scipy',
                   'basis'],
    include_package_data=True,
    zip_safe=False)
