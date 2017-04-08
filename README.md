# Tensorflow Speech Recognition

A example showing how to get CTC (connectionist temporal classification) cost function working with Tensorflow for automatic speech recognition.

## Requirements

- python 2.7+
	- tensorflow 1.0+
	- python_speech_features
	- numpy
	- scipy
- sox (to convert MP3 to WAV)

## Speech Data

I'm trying to transcribe recitation of the Quran from various reciters. The verse-by-verse recitation can be downloaded [here](http://www.everyayah.com/data/getfile.php). Convert them into WAV format using `2wav.sh` script. Some WAV files from surah Al-Fatihah verse 2 are included in the `wav` directory to get started.

## Learning Materials

Some useful introductory materials to get started:

- [Deep Learning for Speech Recognition](https://www.youtube.com/watch?v=g-sndkf7mCs) (video)
- [CTC + Tensorflow + TIMIT](https://igormq.github.io/2016/07/19/ctc-tensorflow-timit/) 
- [Machine Learning is Fun Part 6](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a) 

## License

This project is licensed under the terms of the MIT license.

See README for more information.
