from setuptools import find_packages, setup

package_name = 'main_pkg'

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
    author='Myeongseok Ryu'
    author_email='msryu00@gmail.com'
    maintainer='dding',
    maintainer_email='msryu00@gmail.com',
    classifiers=[
        'Programming Language :: Python',
        'Topic :: RL application'
    ]
    description='autoDrone based on RL with Hierarchy NN',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
        ],
    },
)
