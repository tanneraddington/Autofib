from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sendy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Displacement diffusion model files (same models as diffusion_displacement)
        (os.path.join('share', package_name, 'models', 'diffusion_displacement'),
            glob('sendy/models/diffusion_displacement/*.py')),

        # Displacement diffusion checkpoints
        (os.path.join('share', package_name, 'models', 'diffusion_displacement', 'weights'),
            glob('sendy/models/diffusion_displacement/weights/*')),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tanner',
    maintainer_email='tanner.a.watts@vanderbilt.edu',
    description="Displacement diffusion models (from diffusion_displacement) applied to the uterine system's move/retract control loop",
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            f'uterine_diffusion_node = {package_name}.nodes.uterine_diffusion_node:main',
            f'uterine_move_node = {package_name}.nodes.uterine_move_node:main',
            f'uterine_move_node_r = {package_name}.nodes.uterine_move_node_r:main',
            f'uterine_task_publisher = {package_name}.nodes.task_publisher_node:main',
        ],
    },
)
