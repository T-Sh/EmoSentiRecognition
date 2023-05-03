#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    # Specify your application related info here.
    # Reference: http://pythonhosted.org/distribute/setuptools.html
    name="3-modal emotional bert",
    author="Voloshina Tatyana",
    author_email="tatyana.shimohina23@gmail.com",
    url="https://github.com/T-Sh/3-Modal-Cross-Bert",
    description="ML project for multimodal emotion recognition "
    "and sentiment analysis",
    packages=find_packages(exclude=["tests"]),
)
