import os
import sys
import urllib.request
import torch
import torchaudio
import cuda_malloc
import numpy as np
import urllib
from .mss.utils import get_model_from_config,demix_track,demix_track_demucs

now_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(now_dir, "pretrained_models")
os.makedirs(models_dir,exist_ok=True)

base_url = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/"
configs_dir = os.path.join(now_dir,"mss","configs")
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
config_path_dict = {
    'htdemucs':["config_vocals_htdemucs.yaml","model_vocals_htdemucs_sdr_8.78.ckpt"],
    'mdx23c':["config_vocals_mdx23c.yaml","model_vocals_mdx23c_sdr_10.17.ckpt",],
    'segm_models':["config_vocals_segm_models.yaml","model_vocals_segm_models_sdr_9.77.ckpt"],
    'mel_band_roformer':["model_mel_band_roformer_ep_3005_sdr_11.4360.yaml","https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"],
    'bs_roformer':["model_bs_roformer_ep_317_sdr_12.9755.yaml","https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"],
    # 'swin_upernet':["config_vocals_swin_upernet.yaml","https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/model_swin_upernet_ep_56_sdr_10.6703.ckpt"]
}

def download_from_url(url,path):
    def progressbar(cur, cursize,totalsize):
        percent = '{:.2%}'.format(cur / 100)
        sys.stdout.write('\r')
        # sys.stdout.write("[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)),percent))
        sys.stdout.write("Downloading [%-50s] %s %s/%sMB" % ('=' * int(cur), percent,cursize//1024//1024,totalsize//1024//1024))
        sys.stdout.flush()


    def schedule(blocknum,blocksize,totalsize):
        """
        blocknum:当前已经下载的块
        blocksize:每次传输的块大小
        totalsize:网页文件总大小
        """
        if totalsize == 0:
            percent = 0
        else:
            percent = blocknum * blocksize / totalsize
        if percent > 1.0:
            percent = 1.0
        percent = percent * 100
        # print("download : %.2f%%" %(percent))
        progressbar(percent,blocknum * blocksize,totalsize)
    urllib.request.urlretrieve(url,path,schedule)

class VocalSeparationNode:
    def __init__(self) -> None:
        self.model_type = None
        self.model = None
        self.config = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "music":("AUDIO",),
                "model_type":(['htdemucs','mdx23c','segm_models',
                               'mel_band_roformer','bs_roformer'],{
                                   "default": 'bs_roformer'
                               }),
                "batch_size":("INT",{
                    "default": 4
                }),
                "if_mirror":("BOOLEAN",{
                    "default": True
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("vocals_AUDIO","instrumental_AUDIO",)

    FUNCTION = "separate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_VocalSeparation"

    def separate(self,music,model_type,batch_size,if_mirror):
        torch.backends.cudnn.benchmark = True
        if model_type in ['mel_band_roformer','bs_roformer']:
            config_path = os.path.join(configs_dir,"viperx",config_path_dict[model_type][0])
            model_url = ("https://mirror.ghproxy.com/" if if_mirror else "") + config_path_dict[model_type][1]
            model_path = os.path.join(models_dir,model_url.split("/")[-1])
        else:
            config_path = os.path.join(configs_dir,config_path_dict[model_type][0])
            model_url = ("https://mirror.ghproxy.com/" if if_mirror else "") + base_url + config_path_dict[model_type][1]
            model_path = os.path.join(models_dir,config_path_dict[model_type][1])
        
        if not os.path.isfile(model_path):
            print(f"Downloading {model_path} from {model_url}")
            download_from_url(model_url,model_path)

        if self.model_type != model_type:
            self.model_type = model_type
            self.model, self.config = get_model_from_config(model_type, config_path)
            print('Start from checkpoint: {}'.format(model_path))
            state_dict = torch.load(model_path)
            if model_type == 'htdemucs':
                # Fix for htdemucs pround etrained models
                if 'state' in state_dict:
                    state_dict = state_dict['state']
            self.model.load_state_dict(state_dict)

        print("Instruments: {}".format(self.config.training.instruments))
        self.model.to(device)
        self.model.eval()
        self.config.inference.batch_size = batch_size
        
        audio_data = music["waveform"].squeeze(0)
        audio_rate = music['sample_rate']
        target_sr = 44100
        if audio_rate != target_sr:
            audio_data = torchaudio.transforms.Resample(audio_rate,target_sr)(audio_data)
        
        mix = audio_data.numpy()[0]
        # print(mix.shape)

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)
        # print(mix.shape)
        mix_orig = mix.copy()
        if 'normalize' in self.config.inference:
            if self.config.inference['normalize'] is True:
                mono = mix.mean(0)
                mean = mono.mean()
                std = mono.std()
                mix = (mix - mean) / std
        mixture = torch.tensor(mix, dtype=torch.float32)

        if self.model_type == 'htdemucs':
            res = demix_track_demucs(self.config, self.model, mixture, device)
        else:
            res = demix_track(self.config, self.model, mixture, device)
        
        estimates = res['vocals'].T
        # print(estimates.shape)
        if 'normalize' in self.config.inference:
                if self.config.inference['normalize'] is True:
                    estimates = estimates * std + mean
        
        estimates_mono = estimates.mean(1)
        estimates_t = torch.Tensor(estimates_mono).unsqueeze(0).unsqueeze(0)
        # print(estimates_t.shape)
        vocals = {
            "waveform":estimates_t,
            "sample_rate": target_sr
        }
        instru_mono = (mix_orig.T-estimates).mean(1)
        instru_t = torch.Tensor(instru_mono).unsqueeze(0).unsqueeze(0)
        instrumental = {
            "waveform":instru_t,
            "sample_rate": target_sr
        }
        self.model.to("cpu")
        return (vocals,instrumental,)

NODE_CLASS_MAPPINGS = {
    "VocalSeparationNode": VocalSeparationNode
}
