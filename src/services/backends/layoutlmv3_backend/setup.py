from setuptools import setup, find_packages

setup(
    name="services-backends-layoutlmv3",
    version="0.1.0",
    author="Label Studio",
    description="Label Studio backend for LayoutLMv3",
    packages=find_packages(),
    install_requires=[
        "uvicorn",
        "pytest",
        "pytest-cov",
        "pytest-watch",
        "gunicorn==22.0.0",
        "services-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git",
        "transformers",
        "torch",
        "torchvision",
        "pillow",
        "opencv-python"
    ],
)