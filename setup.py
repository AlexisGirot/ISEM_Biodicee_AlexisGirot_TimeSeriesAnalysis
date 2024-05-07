from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Tools for analysing easily ecological time series'
LONG_DESCRIPTION = 'Tools for analysing easily ecological time series, classify them and compute resilience indicators.'

# Setting up
setup(
        name="ISEM_Biodicee_AlexisGirot_TimeSeriesAnalysis", 
        version=VERSION,
        author="Alexis GIROT",
        author_email="<alexis.girot@normalesup.org>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'Time series analysis', 'Resilience', 'Classification'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)
