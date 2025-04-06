from setuptools import setup, find_packages

setup(
    name="audio_feature_extraction",
    version="3.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "audio_noise_reducer>=2.0.0",
        "audio_vad_detector>=1.5.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="音頻特徵提取與對齊系統",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio_feature_extraction",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
) 