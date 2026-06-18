## Research on waveform, sample rate, spectrograms, MFCCs, and Librosa

## Introduction
Music genre classification is a machine learning task by which audio signals are analyzed and grouped into genres such as pop, hip-hop, classical, rock, jazz, and reggae.  Before a model can classify music, the audio must be converted into numerical features that the model can understand.

## Waveform
A waveform is the raw representation of sound over time. It shows how the amplitude of an audio signal changes from one moment to another. In audio machine learning, the waveform is the starting point for analysis because it contains the original sound information.

## Sample Rate
The sample rate refers to the number of audio samples captured per second. It is measured in Hertz. For example, a sample rate of 22,050 Hz means that 22,050 samples are taken every second. A consistent sample rate is important because machine learning models require audio inputs to be processed in the same format.

## Spectrogram
A spectrogram is a visual representation of sound that shows how frequencies change over time. It is created by converting the waveform from the time domain into the frequency domain. Spectrograms are useful because they reveal patterns in pitch, rhythm, and intensity that may help distinguish one music genre from another.

## MFCCs
Mel-Frequency Cepstral Coefficients, or MFCCs, are audio features commonly used in speech recognition and music classification. MFCCs summarize the frequency characteristics of sound based on how humans perceive audio. They are useful because they reduce complex audio signals into a smaller set of meaningful numerical features.

## Librosa
Librosa is a Python library used for audio and music analysis. It allows users to load audio files, extract features, create spectrograms, and calculate MFCCs. In music genre classification projects, Librosa is commonly used to prepare audio data before training machine learning models.

## Conclusion
Waveforms, sample rate, spectrograms, MFCCs, and Librosa are important concepts in audio machine learning. The waveform represents the original sound, the sample rate controls how the sound is digitally captured, spectrograms show frequency changes over time, and MFCCs provide compact features that help models recognize patterns. Librosa connects these concepts by providing tools to process and analyze audio data in Python.

## References
#### 1. Librosa Development Team. “Librosa Documentation.” Librosa, https://librosa.org/doc/.
#### 2.  Librosa Development Team. “librosa.feature.mfcc.” Librosa Documentation, https://librosa.org/doc/main/generated/librosa.feature.mfcc.html.
#### 3. Librosa Development Team. “Librosa: Audio and Music Processing in Python.” Librosa,   https://librosa.org/.
#### 4. MathWorks. “Spectrogram Using Short-Time Fourier Transform.” MathWorks Documentation.
#### 5. National Instruments. “STFT Spectrogram.” NI Documentation.
#### 6. Practical Cryptography. “Mel Frequency Cepstral Coefficient MFCC Tutorial.”
