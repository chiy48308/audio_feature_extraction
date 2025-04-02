from setuptools import setup, find_packages

setup(
    name="audio_feature_extraction",
    version="0.1.0",
    author="Chris Yi",
    author_email="chiy48308@gmail.com",
    description="音頻特徵提取工具包，用於提取和分析音頻特徵，包括MFCC、F0、能量和過零率等。",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chiy48308/audio_feature_extraction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'librosa>=0.8.0',
        'scipy>=1.6.0',
        'pandas>=1.2.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.50.0'
    ]
) 