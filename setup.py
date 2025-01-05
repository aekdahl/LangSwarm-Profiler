from setuptools import setup, find_packages, find_namespace_packages

# Read dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    
setup(
    name="langswarm-profiler",
    version="0.0.1",
    description="...",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aekdahl/langswarm-profiler",
    author="Alexander Ekdahl",
    author_email="alexander.ekdahl@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_namespace_packages(include=["langswarm.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            # If your package includes CLI tools, specify them here.
            # e.g., "langswarm=core.cli:main",
        ],
    },
)
