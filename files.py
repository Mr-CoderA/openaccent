from operator import index
import pandas as pd
import wave
import torch
import torchaudio

from pathlib import Path
from sklearn.model_selection import train_test_split

from rich.progress import track

class Files:
    def __init__(self, metadata_path="./data/metadata.csv",mels_path='./data/mels', train_path="./data/train.csv", eval_path="./data/eval.csv", random_seed=42):
        self.metadata = pd.read_csv(metadata_path, index_col=[0])
        self.train_data = pd.read_csv(train_path, index_col=[0])
        self.eval_data = pd.read_csv(eval_path, index_col=[0])
        self.files_name = self.metadata.name
        self.train_path = train_path
        self.eval_path = eval_path
        self.random_seed = random_seed
        self.mels_path = mels_path

    def verify(self):
        status = 200
        non_exists = []

        for i in range(0, len(self.files_name)):
            path = Path(f"./data/wavs/{self.files_name[i]}.wav")
            if not path.exists():
                non_exists.append(self.files_name[i])
                status = 400

        return {'status': status, 'non_exists': non_exists}

    def format_check(self):
        path = f"./data/wavs/{self.files_name[0]}.wav"  # convert path to str

        try:
            with wave.open(path, 'rb') as wav:
                details = {
                "channels": wav.getnchannels(),
                "sample_rate": wav.getframerate(),
                "bit_depth": wav.getsampwidth() * 8,
                "frames": wav.getnframes(),
                "duration": wav.getnframes() / wav.getframerate()}
                
                return details
        except wave.Error as e:
            print("Invalid WAV file:", e)
        except FileNotFoundError:
            print("File does not exist")

    def get_train_eval(self):
        df = self.metadata

        df_train, df_eval = train_test_split(df, test_size=0.1, random_state=self.random_seed)

        print(f"Original: {df.shape} - Train: {df_train.shape} - Eval: {df_eval.shape}")

        df_train = df_train.reset_index(drop=True)
        df_eval = df_eval.reset_index(drop=True)

        df_train.to_csv(self.train_path)
        df_eval.to_csv(self.eval_path)

        return {'train': self.train_path, 'eval': self.eval_path}

    def wav_to_mel(self, sample_rate):
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        mel_path = Path(self.mels_path)

        mel_path.mkdir(parents=True, exist_ok=True)
        for i in track(range(0, len(self.files_name)), description="Converting WAV to Mel..."):
            waveform, channels = torchaudio.load(f"./data/wavs/{self.files_name[i]}.wav")

            mel = mel_transform(waveform)
            out_path = mel_path / f"{self.files_name[i]}.pt"
       
            mel = mel.squeeze(0)

            torch.save(mel, out_path)

    def missing_files(self):
        return [f"./data/wavs/{self.metadata.iloc[i]['name']}.wav" for i, x in self.metadata['normalized_text'].isnull().items() if x is True]

    def clean(self):
        missing_files = self.missing_files()

        context = "Found missing values, removed their files and record successfully and updated metadata as well."
        if not missing_files:
            context = 'Nothing to clean'
        else:
            for file in missing_files:
                path = Path(file)
                path.unlink(missing_ok=True)

        self.metadata = self.metadata.dropna().reset_index(drop=True)
        self.files_name = self.metadata.name
        self.metadata.to_csv('./data/metadata.csv')

        missing_files = self.missing_files()

        return {
            "context": context if not missing_files else "Found NUN Files.",
            "status": 200 if not missing_files else 400,
            "missing_files": missing_files
            }