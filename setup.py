from setuptools import setup, find_packages

setup(
    # Name of the package
    name='nn_optimizer',

    # Version of the package
    version='0.1',

    # Automatically find all sub-packages
    packages=find_packages(),

    # List of dependencies installed via pip
    install_requires=[
        'numpy',   # For numerical operations
        'pandas',  # For data manipulation
    ],

    # Metadata
    author='Your Name',
    author_email='your.email@example.com',
    description='A neural network optimizer with various features like activation functions, dropout, batch normalization, and multiple optimizers.',
    keywords='neural network optimizer machine learning',
    url='http://github.com/yourusername/nn_optimizer',  # Replace with the URL of your repository
    license='MIT',

    # Classifiers help users find your project by categorizing it.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],

    # Entry points define the command-line scripts that should be created when installing the package
    entry_points={
        'console_scripts': [
            'nn-optimizer=nn_optimizer.command_line:main',
        ],
    },

    # Additional data to include in the package
    include_package_data=True,

    # Custom scripts or commands
    scripts=['scripts/nn_optimizer_run']
)
