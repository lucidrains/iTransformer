from setuptools import setup, find_packages

setup(
  name = 'iTransformer',
  packages = find_packages(exclude=[]),
  version = '0.8.1',
  license='MIT',
  description = 'iTransformer - Inverted Transformer Are Effective for Time Series Forecasting',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/iTransformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'time series forecasting'
  ],
  install_requires=[
    'beartype',
    'einops>=0.8.0',
    'hyper-connections>=0.0.14',
    'gateloop-transformer>=0.2.3',
    'rotary-embedding-torch',
    'torch>=2.3',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
