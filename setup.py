from __future__ import print_function
from setuptools import setup, find_packages
import sys

setup(
    name="MPUtils",
    version="0.1.0",
    author="yinglang",
    author_email="y19941010@126.com",
    description="some analysis and visualize utils code for mxnet/pytorch coding",
    long_description=open("README.rst").read(),
    license="MIT",
    url="https://github.com/yinglang/MPUtils",
    packages=['MPUtils', 'MPUtils/__normal_utils', 'MPUtils/umxnet', 'MPUtils/umxnet/backgrad', 'MPUtils/upytorch'],
    classifiers=[
        "Environment :: CPU Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: mxnet :: utils',
        'Topic :: pytorch :: utils',
        "Topic :: Deep Learning",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe=True,
)

install_requires=[
        'six>=1.5.2',
    ]
