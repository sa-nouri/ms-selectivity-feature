from setuptools import find_packages, setup

setup(
    name="ms-selectivity-feature",
    version="0.1.0",
    description="Microsaccade Selectivity as Discriminative Feature for Object Decoding",
    author="Salar Nouri",
    author_email="salr.nouri@gmail.com",
    url="https://github.com/sa-nouri/ms-selectivity-feature",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=2.1.1",
        "scikit-learn>=1.5.2",
        "scipy>=1.14.1",
        "matplotlib>=3.9.2",
        "pandas>=2.2.3",
        "seaborn>=0.13.2",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
