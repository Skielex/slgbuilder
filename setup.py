from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="slgbuilder",
      version="0.1.2",
      author="Niels Jeppesen",
      author_email="niejep@dtu.dk",
      description="A Python package for building and cutting sparse layered s-t graphs.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Skielex/slgbuilder",
      packages=["slgbuilder"],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics"
      ],
      install_requires=["numpy", "scipy", "scikit-learn", "thinmaxflow", "thinqpbo"],
      extra_require={"ORTools": ["ortools"]},
      )
