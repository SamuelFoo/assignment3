from setuptools import setup

package_name = "assignment3"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yourname",
    maintainer_email="youremail@email.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector = assignment3.detector:main",
            "detector_yolo = assignment3.detector_yolo:main",
        ],
    },
)
