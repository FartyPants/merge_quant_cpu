# merge_quant_cpu
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

Merge and quatize models for Text WebUI using CPU.

New:
Reworked the terminal progress:

![image](https://github.com/FartyPants/merge_quant_cpu/assets/23346289/7778ca9c-02ab-4c4b-b1f0-f1260c84d667)

Merge can now merge checkpoint (subfolders in Lora)

![image](https://github.com/FartyPants/merge_quant_cpu/assets/23346289/1fc8ce22-90f7-42d5-a93e-1c1aab65116e)


You should be able to figure it out.
Don't try to be smart if you have potato GPU (that includes 24GB, hahaha) and just use CPU. It takes about minute, two to merge to HF and 30 min to quantize to GPTQ

![image](https://github.com/FartyPants/merge_quant_cpu/assets/23346289/b59fe564-3c04-4688-938c-85618b87bca1)

## In general, you don't need to touch any of the settings.

Just select model and Lora and Do Merge

Then Select the merged model and Quantize with default params.
