import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, Classifier
from .gst import GST
from .multiscale_encoder import MultiScaleEncoder
from utils.tools import get_mask_from_lengths
from .reference import SpeakerEncoder


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        
        self.speaker_encoder_type = model_config["spk_encoder_type"] 
        print("Speaker Encoder Type: ", self.speaker_encoder_type)
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))

            if self.speaker_encoder_type == "onehot":
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            elif self.speaker_encoder_type == "gst":
                self.gst = GST(
                    gru_units=model_config["gst"]["gru_hidden"], 
                    conv_channels=model_config["gst"]["conv_filters"],
                    num_tokens=model_config["gst"]["n_style_token"], 
                    token_embed_dim=model_config["gst"]["token_size"], 
                    num_heads=model_config["gst"]["attn_head"]
                )

            elif self.speaker_encoder_type == "cls":
                self.multiscale_encoder = MultiScaleEncoder(
                    gru_units=model_config["multiscale_encoder"]["gru_hidden"],
                    conv_channels=model_config["multiscale_encoder"]["conv_filters"],
                    strides=model_config["multiscale_encoder"]["strides"],
                    global_out_dim=model_config["multiscale_encoder"]["global_style_dim"],
                    local_out_dim=model_config["multiscale_encoder"]["local_style_dim"]
                )
                self.speaker_classifier = Classifier(
                    in_dim=model_config["multiscale_encoder"]["global_style_dim"],
                    out_dim=n_speaker,
                    hidden_dims=model_config["classifier"]["cls_hidden"]
                )
            
            elif self.speaker_encoder_type == "newcls":
                self.speakerencoder = SpeakerEncoder()
                self.speaker_classifier = Classifier(
                    in_dim=256,
                    out_dim=n_speaker,
                    hidden_dims=model_config["classifier"]["cls_hidden"]
                )

    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        speakers=None,
        ref_mels=None, 
        ref_mel_lens=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        speaker_predict = None

        if self.speaker_encoder_type == "onehot":
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        elif self.speaker_encoder_type == "gst":
            output = output + self.gst(ref_mels).expand(
                -1, max_src_len, -1
            )
        elif self.speaker_encoder_type == "cls":
            global_emb, _ = self.multiscale_encoder(ref_mels)
            output = output + global_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            speaker_predict = self.speaker_classifier(global_emb)
        
        elif self.speaker_encoder_type == "newcls":
            global_emb = self.speakerencoder(ref_mels, ref_mel_lens)
            speaker_predict = self.speaker_classifier(global_emb)
            output = output + global_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            speaker_predict,
        )
