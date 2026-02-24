import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'turtlebot3_dqn'
authors_info = [
    ('Gilbert', 'kkjong@robotis.com'),
    ('Ryan Shim', 'N/A'),
    ('ChanHyeong Lee', 'dddoggi1207@gmail.com'),
]
authors = ', '.join(author for author, _ in authors_info)
author_emails = ', '.join(email for _, email in authors_info)

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    author=authors,
    author_email=author_emails,
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    description='ROS 2 packages for TurtleBot3 machine learning',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'loss_graph = turtlebot3_dqn.loss_graph:main',
            'action_graph = turtlebot3_dqn.action_graph:main',
            'dqn_agent = turtlebot3_dqn.dqn_agent:main',
            'rainbow_agent = turtlebot3_dqn.rainbow_agent:main',
            'per_agent = turtlebot3_dqn.per_agent:main',
            'nav2_agent = turtlebot3_dqn.nav2_agent:main',
            'dqn_environment = turtlebot3_dqn.dqn_environment:main',
            'custom_environment = turtlebot3_dqn.custom_environment:main',
            'new_environment = turtlebot3_dqn.new_environment:main',
            'dqn_gazebo = turtlebot3_dqn.dqn_gazebo:main',
            'dqn_test = turtlebot3_dqn.dqn_test:main',
            'result_graph = turtlebot3_dqn.result_graph:main',
            'lidar_subscriber = turtlebot3_dqn.Lidar_Subscriber:main',
            'odom_subscriber = turtlebot3_dqn.Odom_Subscriber:main',
            'evaluation = turtlebot3_dqn.evaluation:main',
            'eval_rosbag = turtlebot3_dqn.eval_rosbag:main',

        ],
    },
)
