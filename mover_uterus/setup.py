from setuptools import find_packages, setup

package_name = 'mover_uterus'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='jamesmorrisferguson@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_calibration = mover_uterus.run_calibration:main',
            'run_test_trajectory = mover_uterus.run_test_trajectory:main',
            'run_hyperparam_calibration = mover_uterus.run_hyperparam_calibration:main',
            'run_joint_calibration = mover_uterus.run_joint_calibration:main',
        ],
    },
)
