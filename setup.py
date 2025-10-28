from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="stream_music_gen",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description="Stream Music Gen",
    author="Yusong Wu, Mason Wang, Heidi Lei, Lancelot Blanchard, Shih-Lun Wu",
    author_email="wuyusongwys@gmail.com",
    url="https://github.com/lukewys/stream-music-gen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
