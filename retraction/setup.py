from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'retraction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all model files
        (os.path.join('share', package_name, 'models', 'PushVIBES'),
         glob('retraction/models/PushVIBES/*.py')),
        (os.path.join('share', package_name, 'models', 'PushVIBES', 'weights'),
         glob('retraction/models/PushVIBES/weights/*')),
        (os.path.join('share', package_name, 'models', 'RetractModels', 'weights'),
         glob('retraction/models/RetractModels/weights/*')),
        (os.path.join('share', package_name, 'models', 'RetractModels'),
         glob('retraction/models/RetractModels/*.py')),
        (os.path.join('share', package_name, 'models', 'RetractModels', 'weights'),
         glob('retraction/models/RetractModels/weights/*')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        
    ],
    install_requires=[
        'setuptools',
        'aliss_ros_msg',
    ],
    zip_safe=True,
    maintainer='james',
    maintainer_email='tanner.a.watts@vanderbilt.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collection_node = retraction.nodes.data_collection_node:main',
            'retract_fib_node = retraction.nodes.retract_fib_node:main',
            'move_node = retraction.nodes.move_node:main',
            'homing_mux = retraction.nodes.homing_mux:main',
            'retract_fib_node_boosted = retraction.nodes.retract_fib_node_boostslip:main',
            'diffusion_fib_node_boosted = retraction.nodes.retract_fib_diffusion_node:main',
            'camera_node = retraction.nodes.camera_node:main'
        ],
    },
)
