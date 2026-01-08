from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
      name="CubeShow"
    , version="1.0"
    , author="iawnix"
    , author_email="iawhaha@163.com"
    , description="A visualization software for cube file information obtained from xtb."
    , install_requires=read_requirements()
    , packages=find_packages()
    , entry_points={
        "console_scripts": [
            "CubeShow=src.run:main"]}
    , python_requires=">=3.12"
)
