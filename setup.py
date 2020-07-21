import os.path
import xml.etree.ElementTree
from pathlib import Path

from setuptools import setup, find_packages

ENTRY_POINTS = [('main', 'main', 'main')]
SHARE_DIRS = [('launch', '*.launch.py'), ('config', '*.rviz'), ('worlds', '*.world'), ('models', '*')]

ROOT = xml.etree.ElementTree.parse('package.xml').getroot()
PACKAGE_NAME = ROOT.findall('name')[-1].text

ALL_MAINTAINERS = ROOT.findall('maintainer')
MAINTAINERS = [m.text for m in ALL_MAINTAINERS]
MAINTAINER_EMAILS = [m.attrib['email'] for m in ALL_MAINTAINERS]

AUTHORS = ROOT.findall('author')
AUTHOR_NAMES = [m.text for m in AUTHORS]
AUTHOR_EMAILS = [m.attrib['email'] for m in AUTHORS]

DATA_FILES = [(f'share/{PACKAGE_NAME}', ['package.xml']),
              ('share/ament_index/resource_index/packages/', [f'resources/{PACKAGE_NAME}'])]
DATA_FILES += [(os.path.join('share', PACKAGE_NAME, str(directory)),
                [str(file) for file in directory.rglob(pattern) if not file.is_dir() and file.parent == directory])
               for folder, pattern in SHARE_DIRS for directory in Path(folder).rglob('**')]

INSTALL_REQUIRES = ['setuptools']
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as file:
        INSTALL_REQUIRES += [line.strip() for line in file.readlines()]

TESTS_REQUIRE = ['pytest']
if os.path.isfile('test-requirements.txt'):
    with open('test-requirements.txt') as file:
        TESTS_REQUIRE += [line.strip() for line in file.readlines()]


setup(
    name=PACKAGE_NAME,
    version=ROOT.findall('version')[-1].text,
    packages=find_packages(),
    data_files=DATA_FILES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    zip_safe=True,
    author=', '.join(AUTHOR_NAMES),
    author_email=', '.join(AUTHOR_EMAILS),
    maintainer=', '.join(MAINTAINERS),
    maintainer_email=', '.join(MAINTAINER_EMAILS),
    keywords=['ROS'],
    description=ROOT.findall('description')[-1].text,
    license=ROOT.findall('license')[-1].text,
    entry_points={'console_scripts': [f'{cmd} = {PACKAGE_NAME}.{file}:{func}' for cmd, file, func in ENTRY_POINTS]}
)
