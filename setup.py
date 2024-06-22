from setuptools import setup, find_packages

setup(
    name='ringattention',
    version='0.1.1',
    license='Apache-2.0',
    description='RingAttention for Transformers with Arbitrarily Large Context.',
    url='https://github.com/lhao499/ringattention',
    packages=find_packages(include=['ringattention']),
    python_requires=">=3.10",
    install_requires=[
        'numpy',
        'jax>=0.4.29',
        'einops',
        'flax',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
