from setuptools import find_packages, setup

package_name = 'rktl_simulator'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy'],
    zip_safe=True,
    maintainer='sixfootsix50',
    maintainer_email='sixfootsix50@gmail.com',
    description='Creates a simulated environment to train the AI on vehicle handling and the rules of soccer.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator = rktl_simulator.simulator:main',
            'runPoint = rktl_simulator.simulatorPoint:main',
            'runTest = rktl_simulator.testingSimPoint:main',
            'sendOne = rktl_simulator.singleMessage:main'
        ],
    },
)
