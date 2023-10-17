import json
import numpy as np
import logging
from pathlib import Path
from transformers import AutoTokenizer
from torchtts.data.core import features
from torchtts.data.core.dataset_builder import GeneratorBasedBuilder
from torchtts.data.core.dataset_info import DatasetInfo
from torchtts.utils.data_utils import get_bucket_scheme, lowercase_dict_keys

from data.audio.voice_tokenizer import VoiceBpeTokenizer
import torch         

logger = logging.getLogger(__name__)


class TortoiseDataset(GeneratorBasedBuilder):
    def _info(self):
        sample_rate = self._config.get("sample_rate", 22050)
        lazy_decode = self._config.get("lazy_decode", True)

        feature_dict = {
            "phone_id": features.Tensor(shape=(None,), dtype=np.int64),
            "duration": features.Tensor(shape=(None,), dtype=np.int64),
            "speech": features.Audio(sample_rate=sample_rate, lazy_decode=lazy_decode),
            "mel": features.Tensor(shape=(None, 80), dtype=np.float32),
            "f0": features.Tensor(shape=(None,), dtype=np.float32),
            "uv": features.Tensor(shape=(None,), dtype=np.float32),
            "speaker_id": features.Tensor(shape=(), dtype=np.int64),
            "locale_id": features.Tensor(shape=(), dtype=np.int64),
            "style_id": features.Tensor(shape=(), dtype=np.int64),
        }

        if self._config.get('with_stat_data', False):
            feature_dict.update({
                'stat': features.Tensor(shape=(None, 6), dtype=np.float32),
                'syl_phone_num': features.Tensor(shape=(None,), dtype=np.int64),
            })

        if self._config.get('with_text_data', False):
            feature_dict.update({"text": features.Text()})

        if self._config.get('with_context_info', False):
            feature_dict.update({
                'has_left_context': features.Tensor(shape=(), dtype=bool),
                'has_right_context': features.Tensor(shape=(), dtype=bool),
            })


        return DatasetInfo(
            builder=self, description="Tortoise dataset builder", features=features.FeaturesDict(feature_dict)
        )

    def _split_generators(self):
        path = self._config.get("raw_data", None)
        if path is None:
            raise ValueError("You should specify raw_data in dataset builder")
        return {k: self._raw_data_generator(split=v, path=Path(path)) for k, v in self.split_type.items()}

    def _raw_data_generator(self, split, path: Path):
        raise NotImplementedError("Not implemented yet, please use unified_tts dataset")

    def _data_pipeline(self, datapipe, shuffle=True):
        # On-the-fly context should be added before shuffle
        self.load_condition = self._config.get("load_condition", False)
        self.conditioning_candidates = self._config.get("conditioning_candidates", 1)
        self.conditioning_length = self._config.get("conditioning_length", 44100)
        self.load_aligned_codes = self._config.get("load_aligned_codes", False)
        self.aligned_codes_to_audio_ratio = self._config.get("aligned_codes_ratio", 443)
        datapipe = datapipe.map(self._process_wav)
        
        if self._config.get("use_t5_tokenizer", False):
            self.tokenizer = AutoTokenizer.from_pretrained(self._config.get("t5_model", "google/flan-t5-base"))
            datapipe = datapipe.map(self._tokenize_t5, fn_kwargs={"t5_tokenizer": self.tokenizer})
        else:
            self.tokenizer = VoiceBpeTokenizer(self._config["vocab_path"])
            datapipe = datapipe.map(self._tokenize, fn_kwargs={"tokenizer": self.tokenizer})
            datapipe = datapipe.filter(self._filter_unk_text)

        datapipe = datapipe.filter(self._filter_long_sentence, fn_kwargs={"max_text_tokens": self._config.get("max_text_tokens", 400),
                                                                          "max_audio_length": self._config.get("max_audio_length", 600 / 22 * 22050)})
        datapipe = datapipe.map(self._release_unnecessary_data)

        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=100)

        if self._config.get("break_map", None):
            # e.g., break_map.json: {"4144": "4145", "4146": "4145"}, 4144/4145/4146 are phone id of br0, br1, br2
            # in phone_set.json, this map means map br0/br2 to br1
            break_map = json.load(open(self._config["break_map"], "r"))
            datapipe = datapipe.map(self._break_replace, fn_kwargs={"break_map": break_map})

        if self._config.get("dynamic_batch", False):
            # Dynamic batching with bucket
            batch_size = self._config.get("batch_size", 2000)
            bucket_step = self._config.get("bucket_step", 1.1)
            bucket_scheme = get_bucket_scheme(batch_size, 8, bucket_step)
            datapipe = datapipe.dynamic_batch(
                group_key_fn=self.get_token_length,
                bucket_boundaries=bucket_scheme["boundaries"],
                batch_sizes=bucket_scheme["batch_sizes"],
            )
        else:
            datapipe = datapipe.batch(self._config["batch_size"])

        # Shuffle on batch
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=32)

        repeat_epoch = self._config.get("repeat_epoch", -1)
        if repeat_epoch != -1:
            datapipe = datapipe.repeat(repeat_epoch)  # this datapipe must be last data pipeline

        # Apply padding and then convert to pytorch tensor
        datapipe = datapipe.collate(
            fn_kwargs={
                "padding_axes": {
                    "text": -1,
                    "wav": -1,
                    "conditioning": -1,
                },
                "padding_values": {"text": 0, "wav": 0, "conditioning": 0},
            }
        )
        return datapipe

    @staticmethod
    def get_token_length(data):
        return data["text"].shape[0]

    @staticmethod
    def _process_wav(data):
        audio = data["speech"]
        data["wav"] = torch.from_numpy(audio)
        data["wav_lengths"] = torch.LongTensor([len(audio)])
        data["conditioning"] = data["wav"].unsqueeze(0)
        return data
    
    @staticmethod
    def _tokenize_t5(data, t5_tokenizer):
        text = data["text"]
        returns = t5_tokenizer(text, return_tensors="pt")
        data['original_text'] = text
        data['text'] = returns['input_ids'].squeeze(0)
        data['text_lengths'] = torch.LongTensor([len(returns['input_ids'])])
        return data

    @staticmethod
    def _tokenize(data, tokenizer):
        text = data["text"]
        #text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        data["original_text"] = text
        tokens = tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        data["text"] = tokens
        data["text_lengths"] = torch.LongTensor([len(tokens)])
        return data

    @staticmethod
    def _filter_unk_text(data):
        text = data["text"]
        if torch.any(text <= 1):
            # logger.warning(f"Found <unk> in {data['original_text']}")
            return False
        return True

    @staticmethod
    def _filter_long_sentence(data, max_text_tokens, max_audio_length):
        text = data["text"]
        wav = data["wav"]
        if text.shape[0] > max_text_tokens or wav.shape[0] > max_audio_length:
            return False
        return True

    @staticmethod
    def _release_unnecessary_data(data):
        necessary_keys = ["wav", "wav_lengths", "text", "text_lengths", "conditioning"]
        for key in list(data.keys()):
            if key not in necessary_keys:
                del data[key]
        return data


   
