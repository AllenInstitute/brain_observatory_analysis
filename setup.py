from setuptools import setup

setup(name='brain_observatory_analysis',
      version='0.0.1',
      packages=['brain_observatory_analysis'],
      include_package_data=True,
      description='Analysis tools for the Allen Institute Brain Observatory',
      url='https://github.com/AllenInstitute/brain_observatory_analysis',
      author='Allen Institute',
      author_email='matt.davis@alleninstitue.org',
      license='Allen Institute',
      install_requires=[
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'License :: Other/Proprietary License',  # Allen Institute Software License
          'Natural Language :: English',
          'Programming Language :: Python :: 3.8'
      ],
      )
