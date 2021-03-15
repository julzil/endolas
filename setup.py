import setuptools

setuptools.setup(
    name="endolas",
    version="0.4.0",
    author="Julian Zilker",
    author_email="julian.zilker@gmx.de",
    description="pipeline for detecting and matching projected laser points",
    license="GPLv3+",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
	"tensorflow==2.4.0",
	"imageio==2.9.0",
	"matplotlib==3.3.3",
	"albumentations==0.5.1",
	"jupyter==1.0.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/engineerByNature/glabel",
    python_requires='>=3.7',
)
