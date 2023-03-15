# AI_Illustrator
[MM'22 Oral] AI Illustrator: Translating Raw Descriptions into Images by Prompt-based Cross-Modal Generation

This reposity is the official implementation of <a href="https://arxiv.org/abs/2209.03160">AI Illustrator: Translating Raw Descriptions into Images by Prompt-based Cross-Modal Generation</a>.

The proposed pipeline is shown below.

<img src="/figs/overview.png" width="100%">  

## Usage
### Pretrained Models

Currently, we support pretrained models on 3 domains: Human face, Cat and Church. The download urls are:

| |Human face|Cat|Church|
|----|----|----|----|
|Baidu Disk|https://pan.baidu.com/s/1wDVj_YGYQoQlRDk5tTtL8A, <br /> extracting code: huzx|https://pan.baidu.com/s/1UIhCBL2Cl9CenjjCNsbHJw, <br /> extracting code: 7ul6 |https://pan.baidu.com/s/1WTrWgrjs9FD8o4ZyoQalxg, <br /> extracting code: hffh|
|Google Drive|TBD|TBD|TBD|
|One Drive|https://1drv.ms/u/s!Aq9epwaFQGaOgQpUdtd81YWh_TVe?e=5vuPjD|https://1drv.ms/u/s!Aq9epwaFQGaOgQwjbTUXjYgw-g_z?e=lTg9lX|https://1drv.ms/u/s!Aq9epwaFQGaOgQ1foRYeby5jHR96?e=g4yq8G|

The default path of pretrained models is ```./pretrained_projectors```.

### Generating

After downloading the pretrained models, you can simply generate images by command

     python single_generate.py --kind <domain> --projector_path <path/to/the/pretrained_projector> --save_path <path/to/the/save_dir> --strength 1.75 --prompt_path <path/to/the/text_prompt>

One example is

     python single_generate.py --kind 'human' --projector_path './pretrained_projector/c2s_human.pth' --save_path './outputs' --strength 1.75 --prompt_path './prompts/ffhq_text_prompt.pth'

The values of argument "kind", "projector_path" and "prompt_path" should match.
By default, "kind" should be one of "human", "cat" and "church".


### Benchmarking

100 raw descriptions can be found in ```./benchmark/description_benchmark_100.txt```

Corresponding generation results of our method can be download at

|Baidu Disk|Google Drive|One Drive|
|----|----|----|
|TBD|TBD|TBD|
