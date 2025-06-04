import numpy as np
import os
import glob
import cv2
import yaml
from utils import video_augmentation
from Tokenizer import GlossTokenizer_S2G
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils
from utils.metrics import wer_list
import time # For timing
from thop import profile, clever_format # For GFLOPs calculation

###
gpu_id = 0 # The GPU to use
dataset = 'phoenix2014-T'  # support [phoenix2014, phoenix2014-T, CSL-Daily]
select_id = 13
model_weights = "./phoenix2014-t/best.pt"
### 
sparser = utils.get_parser()
p = sparser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        try:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            default_arg = yaml.load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    sparser.set_defaults(**default_arg)
args = sparser.parse_args()
with open(f"./configs/{dataset}.yaml", 'r') as f:
    args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

prefix = args.dataset_info['dataset_root']
# Load data and apply transformation
gloss_tokenizer = GlossTokenizer_S2G(args.dataset_info['gloss'])
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
img_folder = os.path.join(prefix, "features/fullFrame-210x260px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, "sentence/frames_512x512/" +inputs_list[select_id]['folder'])
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.resize(cv2.imread(img_path), (256, 256), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = gloss_tokenizer(inputs_list[select_id]['label'].split(" "))
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0), # This resize might be redundant if CenterCrop already gives 224x224
                video_augmentation.ToTensor(),
            ])
vid = transform(img_list)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0) # Add batch dimension: (1, T, C, H, W)

# Padding calculations
left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = args.kernel_size if hasattr(args, 'kernel_size') and args.kernel_size else ['K5', "P2", 'K5', "P2"]

for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

video_length_original_frames = vid.size(1) # Number of frames before padding
padded_len_feature = int(np.ceil(vid.size(1) / total_stride)) # Length after model's temporal striding
video_length = torch.LongTensor([padded_len_feature]) # This is often what models expect for CTC or attention mechanisms


current_len = vid.size(1)
target_len_for_stride = int(np.ceil(current_len / total_stride)) * total_stride
right_pad_for_stride = target_len_for_stride - current_len

max_len_before_conv_pad = vid.size(1)
right_pad_conv = int(np.ceil(max_len_before_conv_pad / total_stride)) * total_stride - max_len_before_conv_pad + left_pad
max_len_after_conv_pad = max_len_before_conv_pad + left_pad + right_pad_conv

vid_padded = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1).clone(), # Use .clone() to avoid modifying original tensor views
        vid[0],
        vid[0,-1][None].expand(max_len_after_conv_pad - vid.size(1) - left_pad, -1, -1, -1).clone(),
    )
    , dim=0).unsqueeze(0)
vid = vid_padded # Use the padded video

video_length_for_model = torch.LongTensor([int(np.ceil(vid.size(1) / total_stride))])

if not hasattr(args, 'slowfast_config'): args.slowfast_config = None # Or path to default config
if not hasattr(args, 'slowfast_args'): args.slowfast_args = None # Or default dict

model_for_flops = SLRModel(
    num_classes=len(gloss_tokenizer),
    c2d_type=args.c2d_type if hasattr(args, 'c2d_type') else 'slowfast101', # Make sure c2d_type is available
    conv_type=2, # Ensure this matches your model's conv_type
    gloss_dict=gloss_tokenizer,
    loss_weights=args.loss_weights,
    load_pkl=args.load_pkl if hasattr(args, 'load_pkl') else True, # Default if not in args
    slowfast_config=args.slowfast_config,
    slowfast_args=args.slowfast_args
)
model_for_flops.eval()

dummy_vid_for_flops = vid.clone().cpu() # Use the padded video shape
dummy_vid_lgt_for_flops = video_length_for_model.clone().cpu()
dummy_label_for_flops = None # For inference, labels are not used by the network path for FLOPs
dummy_label_lgt_for_flops = None

try:
    macs, params = profile(model_for_flops,
                           inputs=(dummy_vid_for_flops, dummy_vid_lgt_for_flops,
                                   dummy_label_for_flops, dummy_label_lgt_for_flops),
                           verbose=False)
    # GFLOPs = MACs * 2 (approx for most common ops)
    gflops = (macs * 2) / 1e9
    params_m = params / 1e6
    print(f"--- Model Complexity ---")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Parameters (Millions): {params_m:.2f}")
except Exception as e:
    print(f"Could not calculate GFLOPs with thop: {e}")
    gflops = -1
    params_m = -1
del model_for_flops, dummy_vid_for_flops, dummy_vid_lgt_for_flops # Free memory
# --- End GFLOPs Calculation ---


# --- GPU Memory and Inference Speed ---
device_util = utils.GpuDataParallel() # Renamed to avoid conflict with torch.device
device_util.set_device(gpu_id)
target_device = device_util.output_device if isinstance(device_util.output_device, int) else device_util.output_device[0]

torch.cuda.reset_peak_memory_stats(target_device)

# Load the actual model for inference
model = SLRModel(
    num_classes=len(gloss_tokenizer),
    c2d_type=args.c2d_type if hasattr(args, 'c2d_type') else 'slowfast101',
    conv_type=2,
    gloss_dict=gloss_tokenizer,
    loss_weights=args.loss_weights,
    load_pkl=args.load_pkl if hasattr(args, 'load_pkl') else True,
    slowfast_config=args.slowfast_config,
    slowfast_args=args.slowfast_args
)

state_dict = torch.load(model_weights, map_location='cpu')['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
s_dict = model.state_dict()
# Filter out incompatible keys (e.g., from a different head size if pretraining)
new_state_dict = {}
for name1, param1 in state_dict.items():
    if name1 in s_dict:
        if s_dict[name1].shape == param1.shape:
            new_state_dict[name1] = param1
        else:
            print(f"Skipping loading {name1}: shape mismatch ({s_dict[name1].shape} vs {param1.shape})")
    else:
        print(f"Skipping loading {name1}: not found in model")

model.load_state_dict(new_state_dict, strict=False) # Use strict=False if some keys are intentionally missing
model = model.to(target_device)
# model.cuda() # .to(device) is generally preferred and sufficient
model.eval()

# Prepare inputs for GPU
vid_gpu = device_util.data_to_device(vid) # vid is already padded
vid_lgt_gpu = device_util.data_to_device(video_length_for_model) # Use the length for the model
label_gpu = device_util.data_to_device([torch.LongTensor(label_list['gloss_labels'])])
label_lgt_gpu = device_util.data_to_device(torch.LongTensor(label_list['gls_lengths']))

# Perform a single inference pass to ensure all necessary memory is allocated on GPU
with torch.no_grad():
    _ = model(vid_gpu, vid_lgt_gpu, label=label_gpu, label_lgt=label_lgt_gpu)
torch.cuda.synchronize(target_device)

max_mem_bytes = torch.cuda.max_memory_allocated(target_device)
max_mem_gb = max_mem_bytes / (1024 ** 3)
print(f"\n--- Performance Metrics ---")
print(f"Peak GPU Memory Usage: {max_mem_gb:.2f} GB ({max_mem_bytes / (1024**2):.2f} MB)")

# Inference speed test
N_WARMUP = 5
N_RUNS = 20
print(f"Warming up for {N_WARMUP} iterations...")
for _ in range(N_WARMUP):
    with torch.no_grad():
        _ = model(vid_gpu, vid_lgt_gpu, label=label_gpu, label_lgt=label_lgt_gpu)
    torch.cuda.synchronize(target_device)

print(f"Measuring inference speed over {N_RUNS} iterations...")
total_time = 0
all_times = []
for i in range(N_RUNS):
    torch.cuda.synchronize(target_device) # Ensure previous CUDA ops are done
    start_time = time.perf_counter()
    with torch.no_grad():
        ret_dict = model(vid_gpu, vid_lgt_gpu, label=label_gpu, label_lgt=label_lgt_gpu)
    torch.cuda.synchronize(target_device) # Wait for model inference to complete on GPU
    end_time = time.perf_counter()
    iter_time = (end_time - start_time)
    total_time += iter_time
    all_times.append(iter_time)
    # print(f"Run {i+1}/{N_RUNS}: {iter_time*1000:.2f} ms")


avg_inference_time_s = total_time / N_RUNS
avg_inference_time_ms = avg_inference_time_s * 1000
std_dev_time_ms = np.std(all_times) * 1000

videos_per_second = 1 / avg_inference_time_s
# video_length_original_frames is T from vid.unsqueeze(0) -> (1, T, C, H, W)
# This is the number of frames in the input video *before* any padding for convolutions.
frames_per_second = video_length_original_frames / avg_inference_time_s


print(f"Average inference time per video: {avg_inference_time_ms:.2f} ms (std: {std_dev_time_ms:.2f} ms)")
print(f"Inference speed: {videos_per_second:.2f} videos/sec")
print(f"Effective processing speed: {frames_per_second:.2f} frames/sec (based on {video_length_original_frames} original frames)")

# Original output
print('\n--- Recognition Results ---')
print('Video Name:            ', name)
print('Label is:              ', inputs_list[select_id]['label'])
predicted_glosses = " ".join(ret_dict['recognized_sents'][0]) if ret_dict['recognized_sents'] and ret_dict['recognized_sents'][0] else ""
print('The predicted value is:', predicted_glosses)
wer = wer_list(inputs_list[select_id]['label'].split(" ")," ".join(ret_dict['recognized_sents'][0]).split(" "))['wer']
print('The wer value:         ', wer)