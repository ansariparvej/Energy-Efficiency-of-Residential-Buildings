from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list: List[str] = []

    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    return requirement_list


setup(
    name="energy-efficiency",
    version="0.0.1",
    author="Ansari",
    author_email="ansarip82@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),  # ["pymongo==4.3.3"]
)
