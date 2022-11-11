"""Setup mmapy."""
import setuptools

setuptools.setup(
    name='mmapy',
    version='0.0.1',
    license='GPL-2.0',
    author='Aaditya Chandrasekhar',
    author_email='cs.aaditya@gmail.com',
    install_requires=[
        'dataclasses',
        'absl',
        'jax',
        'numpy',
        'scipy',
        'absl-py',
        'pytest',
    ],
    url='https://github.com/aadityacs/mmapy',
    packages=setuptools.find_packages(),
    python_requires='>=3',
)
