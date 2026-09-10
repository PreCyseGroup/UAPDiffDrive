from glob import glob
from setuptools import find_packages, setup

package_name = "nn_trajectory_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/models", glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="UAP IL User",
    maintainer_email="user@example.com",
    description="ROS2 neural-network trajectory tracker for lemniscate tracking.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lemniscate_nn_tracker = nn_trajectory_tracker.lemniscate_nn_tracker:main",
        ],
    },
)
