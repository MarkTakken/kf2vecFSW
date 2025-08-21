from setuptools import setup, find_packages


setup(
    name="kf2vec",                               # Package name
    version="1.0.62",                            # Initial version
    author="Eleonora Rachtman",
    author_email="noraracht@gmail.com",
    description="k-mer frequency to vector tool",
    url="https://github.com/noraracht/kf2vec",   # Project homepage
    packages=find_packages(),                    # Automatically find all packages and subpackages
    include_package_data=True,
    package_data={
        "kf2vec": ["data/*"],
    },
    python_requires='>=3.11,<3.12',                     # Minimum Python version
    install_requires=[              # Runtime dependencies,
        "numpy>=1.22,<1.27",
        "pandas",
        "scikit-learn",
        "torch",
        "treecluster>=1.0.3",
        "treeswift>=1.1.45"
    ],
    classifiers=[                                # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={                               # Optional: for command-line scripts
        "console_scripts": [
            "kf2vec=kf2vec.main:main",
        ],
    },
)

