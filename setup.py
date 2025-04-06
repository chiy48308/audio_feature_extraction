from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audio-feature-extraction-toolkit",
    version="0.1.0",
    author="Chris",
    author_email="chiy48308@gmail.com",
    description="一個用於音頻特徵提取的Python工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chiy48308/audio-feature-extraction-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "librosa>=0.9.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "soundfile>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
) 