# AI_Illustrator
[MM'22 Oral] AI Illustrator: Translating Raw Descriptions into Images by Prompt-based Cross-Modal Generation

This reposity is the official implementation of <a href="https://arxiv.org/abs/2209.03160">AI Illustrator: Translating Raw Descriptions into Images by Prompt-based Cross-Modal Generation</a>.

The proposed pipeline is shown below.

<img src="/figs/overview.png" width="100%">  

## Usage
<b> Pretrained Models </b>

Currently, we support pretrained models on 3 domains: Human face, Cat and Church. The download urls are:

| |Human face|Cat|Church|
|----|----|----|----|
|Baidu Disk|p|p|p|
|Google Drive|p|p|p|
|One Drive|p|p|p|

The default path of pretrained models is ```./pretrained_projectors```.

<b> Generating </b>

After downloading the pretrained models, you can simply generate images by command

     python single_generate.py --kind <domain> --projector_path <path/to/the/pretrained_projector> --save_path <path/to/the/save_dir> --strength 1.75 --prompt_path <path/to/the/text_prompt>

One example is

     python single_generate.py --kind 'human' --projector_path './pretrained_projector/c2s_human.pth' --save_path './outputs' --strength 1.75 --prompt_path './prompts/ffhq_text_prompt.pth'

The values of argument "kind", "projector_path" and "prompt_path" should match.
