#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0',
                'numpy>=1.19',
                'pint',
                'scipy>=1.5',
                'matplotlib', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jeff Burghardt",
    author_email='jeffrey.burghardt@pnnl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="State of stress analysis tool",
    entry_points={
        'console_scripts': [
            'SOSAT=SOSAT.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='SOSAT',
    name='SOSAT',
    packages=find_packages(include=['SOSAT', 'SOSAT.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pnnl/SOSAT',
    version='0.7.0',
    zip_safe=False,
)
