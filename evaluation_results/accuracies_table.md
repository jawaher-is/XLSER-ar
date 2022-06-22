
| Base Model                        | Cross-Language Training Dataset     | Emotions Dataset | Base Model Features | Attention Mask | Training Epochs | Accuracy
| --------                          | --------              | --------    | --------    | -------- | -------- | -------- |
| Wav2Vec2-base                     | -                     | KSUEmotions | frozen      | No  | 1 | 79.9
| Wav2Vec2-base                     | -                     | KSUEmotions | frozen      | No  | 3 | 91.5
| Wav2Vec2-base                     | -                     | KSUEmotions | frozen      | No  | 5 | 95.4
| Wav2Vec2-base                     | -                     | KSUEmotions | fine-tuned  | No  | 1 | 74.7
| Wav2Vec2-base                     | -                     | KSUEmotions | fine-tuned  | No  | 3 | 85.1
| Wav2Vec2-base                     | -                     | KSUEmotions | fine-tuned  | No  | 5 | 91.2
| Wav2Vec2-Large-XLSR-53-Arabic     | Common Voice Arabic   | KSUEmotions | frozen      | No  | 1 | 89.9
| Wav2Vec2-Large-XLSR-53-Arabic     | Common Voice Arabic   | KSUEmotions | frozen      | No  | 3 | 94.8
| Wav2Vec2-Large-XLSR-53-Arabic     | Common Voice Arabic   | KSUEmotions | fine-tuned  | No  | 1 | 88.1
| Wav2Vec2-Large-XLSR-53-Arabic     | Common Voice Arabic   | KSUEmotions | fine-tuned  | No  | 3 | 95.1
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | KSUEmotions | frozen      | No  | 3 | 93.9
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | KSUEmotions | frozen      | Yes | 3 | 91.2
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | KSUEmotions | frozen      | Yes | 5 | 95.7
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | KSUEmotions | fine-tuned  | Yes | 1 | 91.5
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | KSUEmotions | fine-tuned  | Yes | 3 | 95.4
| Wav2Vec2-Large-XLS-R-1B-Arabic    | Common Voice Arabic   | KSUEmotions | frozen      | Yes | 1 | 51.5
| Wav2Vec2-Large-XLS-R-1B-Arabic    | Common Voice Arabic   | KSUEmotions | frozen      | Yes | 3 | 61.9
| Wav2Vec2-Large-XLSR-53-English    | Common Voice English  | KSUEmotions | frozen      | Yes | 1 | 84.1
| Wav2Vec2-Large-XLSR-53-English    | Common Voice English  | KSUEmotions | frozen      | Yes | 3 | 97.3
| Wav2Vec2-Large-XLSR-53-English    | Common Voice English  | KSUEmotions | frozen      | No  | 3 | 93.6
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | Ravdess     | frozen      | No  | 3 | 81.9
| Wav2Vec2-Large-XLS-R-300M-Arabic  | Common Voice Arabic   | Ravdess     | frozen      | No  | 5 | 88.2
