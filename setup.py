import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biorad-pkg-AHMED-ALBUNI",
    version="0.0.1",
    author="Ahmed Albuni",
    author_email="ahmed.albuni@gmail.com",
    description="Radiomics features extractions and analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedalbuni/biorad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
