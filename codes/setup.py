from setuptools import setup
from setuptools import find_packages

setup(
    name='TorchTTS',
    version='0.1',
    packages=find_packages(),
    url='https://dev.azure.com/msasg/TextToSpeech/_git/NeuralVoiceModelling?path=%2FTorchTTS',
    license='MIT License',
    author='yuczha',
    author_email='yuczha@microsoft.com',
    description='TorchTTS toolkit',
    python_requires='>=3.6'
)
