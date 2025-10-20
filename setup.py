from setuptools import setup, find_packages

setup(
    name='autonomous-research-agent',
    version='1.0',
    author='Jiawei Zhou',
    author_email='davidzjw@sjtu.edu.cn',
    description='An autonomous agent for generating and refining research ideas.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/valleysprings/autonomous-research-agent',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'networkx',
        'torch',
        'transformers',
        'pyyaml',
        'requests',
        # Core runtime deps used by the codebase
        'aiohttp',                 # async HTTP for LLM + Semantic Scholar
        'gradio',                  # Web UI
        'tokencost',               # token cost accounting
        'tiktoken',                # tokenization backend for tokencost
        'sentence-transformers',   # embedding models for distinctness analysis
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
