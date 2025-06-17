# Awesome AI-generated Image Detection
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A collection list of AI-generated image detection related datasets, tools, papers, and code.

If you want to contribute to this list, welcome to send me a pull request or contact me :)

+ :globe_with_meridians: Project Page
+ :octocat: Code
+ :bricks: Datasets Download Link

## TODOs

- [ ] Add ICML 2025 papers
- [x] Add CVPR 2025 paper links & code links 
- [x] Add CVPR 2025 paper titles


## Latest Updates 🔥


+ [Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2411.15633) (Zhiyuan Yan et al., ICML 2025) [:octocat:](https://github.com/YZY-stack/Effort-AIGI-Detection)

+ [Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection](https://arxiv.org/abs/2502.07778) (Anirudh Sundara Rajan et al., ICML 2025) [:octocat:](https://github.com/AniSundar18/AlignedForensics) [:globe_with_meridians:](https://anisundar18.github.io/Stay-Positive/)

+ [Are High-Quality AI-Generated Images More Difficult for Models to Detect?] (Yao Xiao et al., ICML 2025)

+ [Few-Shot Learner Generalizes Across AI-Generated Image Detection] (Shiyu Wu et al., ICML 2025)

+ [PiD: Generalized AI-Generated Images Detection with Pixelwise Decomposition Residuals] (Xinghe Fu et al., ICML 2025)

## Contents

- [Awesome AI-generated Image Detection](#awesome-ai-generated-image-detection)
  - [TODOs](#todos)
  - [Latest Updates 🔥](#latest-updates-)
  - [Contents](#contents)
  - [Benchmark/Survey](#benchmarksurvey)
  - [Datasets](#datasets)
  - [Reconstruction Error](#reconstruction-error)
  - [Vision foundation model related](#vision-foundation-model-related)
  - [Dataset bias](#dataset-bias)
  - [Training-free](#training-free)
  - [Transformation](#transformation)
  - [knowledge distillation](#knowledge-distillation)
  - [Patch](#patch)
  - [Multimodal Large Language Model](#multimodal-large-language-model)
  - [Noise fingerprint](#noise-fingerprint)
  - [Data-driven](#data-driven)
  - [Frequency domain](#frequency-domain)
- [Shadows](#shadows)
  - [Tools](#tools)
  - [Citing Awesome AI-generated Image Detection](#citing-awesome-ai-generated-image-detection)
  - [Acknowledgments](#acknowledgments)


## Benchmark/Survey

+ [Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images](https://arxiv.org/abs/2304.13023) (Zuyu Lu et al., NeurIPS 2023) [:octocat:](https://github.com/Inf-imagine/Sentry)

+ [Recent Advances on Generalizable Diffusion-generated Image Detection](https://arxiv.org/abs/2502.19716) (Qijie Xu et al., arxiv 2025) [:octocat:](https://github.com/zju-pi/Awesome-Diffusion-generated-Image-Detection)


## Datasets

<!-- 
+ []() (, ) [:globe_with_meridians:]() [:octocat:]() [:bricks:]()
-->

+ [Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation](https://arxiv.org/abs/2503.14905v1) (Siwei Wen et al., arxiv 2025) [:octocat:](https://github.com/opendatalab/FakeVLM) [:bricks:](https://huggingface.co/datasets/lingcco/FakeClue)

+ [**CO-SPY**: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](https://arxiv.org/abs/2503.18286) (Siyuan Cheng et al., CVPR2025) [:octocat:](https://github.com/Megum1/Co-Spy) [:bricks:](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench)

+ [A Sanity Check for AI-generated Image Detection](https://arxiv.org/abs/2406.19435) (Shilin Yan et al., ICLR 2025) [:octocat:](https://github.com/shilinyan99/AIDE) [:bricks:](https://github.com/shilinyan99/AIDE#-chameleon)

+ [**ImagiNet**: A Multi-Content Dataset for Generalizable Synthetic Image Detection via Contrastive Learning](https://arxiv.org/abs/2407.20020v3) (Delyan Boychev et al., ECCV 2024) [:octocat:](https://github.com/delyan-boychev/imaginet) [:bricks:](https://huggingface.co/datasets/delyanboychev/imaginet)

+ [**DRCT**: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images](https://openreview.net/forum?id=oRLwyayrh1) (Baoying Chen et al., ICML 2024) [:octocat:](https://github.com/beibuwandeluori/DRCT) [:bricks:](https://modelscope.cn/datasets/BokingChen/DRCT-2M/files)

+ [**Synthbuster**: Towards detection of diffusion model generated images.](https://ieeexplore.ieee.org/document/10334046) [:bricks:](https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images)

+ [**GenImage**: A Million-Scale Benchmark for Detecting AI-Generated Image](https://arxiv.org/abs/2306.08571) (Mingjian Zhu et al., NeurIPS 2023) [:globe_with_meridians:](https://genimage-dataset.github.io/) [:octocat:](https://github.com/GenImage-Dataset/GenImage) [:bricks:](https://github.com/GenImage-Dataset/GenImage#dataset)

+ [**CNNDetect**: CNN-generated images are surprisingly easy to spot...for now](https://arxiv.org/abs/1912.11035) (Sheng-Yu Wang et al., CVPR 2020) [:globe_with_meridians:](https://peterwang512.github.io/CNNDetection/) [:octocat:](https://github.com/peterwang512/CNNDetection) [:bricks:](https://github.com/peterwang512/CNNDetection?tab=readme-ov-file#3-dataset)


## Reconstruction Error

+ [FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error](https://arxiv.org/abs/2412.07140v2) (Beilin Chu et al., CVPR 2025) [:octocat:](https://github.com/Chuchad/FIRE)

+ [ReTD: Reconstruction-Based Traceability Detection for Generated Images](https://ieeexplore.ieee.org/abstract/document/10890492) (Weizhou Chen et al., ICASSP 2025) [:octocat:](https://github.com/chenweizhuo/ReTD)

+ [Spatial-Temporal Reconstruction Error for AIGC-based Forgery Image Detection](https://ieeexplore.ieee.org/abstract/document/10890455) (Chengji Shen et al., ICASSP 2025)

+ [Time Step Generating: A Universal Synthesized Deepfake Image Detector](https://arxiv.org/abs/2411.11016v2) (Ziyue Zeng et al., arxiv 2024) [:octocat:](https://github.com/NuayHL/TimeStepGenerating)

+ [DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images](https://openreview.net/forum?id=oRLwyayrh1) (Baoying Chen et al., ICML 2024) [:octocat:](https://github.com/beibuwandeluori/DRCT)

+ [FakeInversion: Learning to Detect Images from Unseen Text-to-Image Models by Inverting Stable Diffusion](https://arxiv.org/abs/2406.08603) (George Cazenavette et al., CVPR 2024) [:globe_with_meridians:](https://fake-inversion.github.io/)

+ [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_LaRE2_Latent_Reconstruction_Error_Based_Method_for_Diffusion-Generated_Image_Detection_CVPR_2024_paper.html) (Yunpeng Luo et al., CVPR 2024) [:octocat:](https://github.com/luo3300612/LaRE)

+ [AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html) (Jonas Ricker et al., CVPR 2024) [:octocat:](https://github.com/jonasricker/aeroblade)

+ [DIRE for Diffusion-Generated Image Detection](https://arxiv.org/abs/2303.09295) (Zhendong Wang et al., ICCV 2023) [:octocat:](https://github.com/ZhendongWang6/DIRE)

## Vision foundation model related

+ [CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](https://arxiv.org/abs/2503.18286) (Siyuan Cheng et al., CVPR2025) [:octocat:](https://github.com/Megum1/Co-Spy) 

+ [A Sanity Check for AI-generated Image Detection](https://arxiv.org/abs/2406.19435) (Shilin Yan et al., ICLR 2025) [:octocat:](https://github.com/shilinyan99/AIDE) 

+ [C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection](https://arxiv.org/abs/2408.09647) (Chuangchuang Tan et al. AAAI 2025) [:octocat:](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) 

+ [Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2411.15633) (Zhiyuan Yan et al., arxiv 2025) [:octocat:](https://github.com/YZY-stack/Effort-AIGI-Detection)

+ [Mixture of Low-rank Experts for Transferable AI-Generated Image Detection](https://arxiv.org/abs/2404.04883) (Zihan Liu et al., arxiv 2024) [:octocat:](https://github.com/zhliuworks/CLIPMoLE)

+ [Guided and Fused: Efficient Frozen CLIP-ViT with Feature Guidance and Multi-Stage Feature Fusion for Generalizable Deepfake Detection](https://arxiv.org/abs/2408.13697) (Yingjian Chen et al., arxiv 2024)

+ [Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection](https://arxiv.org/abs/2402.19091) (Christos Koutlis et al., ECCV 2024) [:octocat:](https://github.com/mever-team/rine) 

+ [Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection](https://arxiv.org/abs/2312.16649) (Huan Liu et al., CVPR 2024) [:octocat:](https://github.com/Michel-liu/FatFormer) 

+ [Clipping the deception: Adapting vision-language models for universal deepfake detection](https://arxiv.org/pdf/2402.12927) (Sohail Ahmed Khan et al., ICMR 2024) [:octocat:](https://github.com/sohailahmedkhan/CLIPping-the-Deception) 

+ [De-fake: Detection and attribution of fake images generated by text-to-image generation models](https://arxiv.org/abs/2210.06998) (Zeyang Sha et al., CCS 2024) [:octocat:](https://github.com/zeyangsha/De-Fake) 

+ [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174) (Utkarsh Ojha et al., CVPR 2023) [:octocat:](https://github.com/WisconsinAIVision/UniversalFakeDetect) 

## Dataset bias

+ [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://arxiv.org/abs/2411.04125v2) (Jeongsoo Park et al., CVPR 2025) [:global_with_meridians:](https://jespark.net/projects/2024/community_forensics/) [:octocat:](https://github.com/JeongsooP/Community-Forensics)

+ [Aligned Datasets Improve Detection of Latent Diffusion-Generated Images](https://openreview.net/pdf?id=doBkiqESYq) (Anirudh Sundara Rajan et al., ICLR 2025) [:globe_with_meridians:](https://anisundar18.github.io/AlignedForensics/) [:octocat:](https://github.com/AniSundar18/AlignedForensics) 

+ [SFLD: Reducing the content bias for AI-generated Image Detection](https://arxiv.org/abs/2502.17105) (Seoyeon Gye et al., arxiv 2025)

+ [Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/abs/2403.17608) (Patrick Grommelt et al., arxiv 2024) [:globe_with_meridians:](https://www.unbiased-genimage.org/) [:octocat:](https://github.com/gendetection/UnbiasedGenImage) 


## Training-free

+ [Manifold Induced Biases for Zero-shot and Few-shot Detection of Generated Images](https://openreview.net/pdf?id=7gGl6HB5Zd) (Jonathan Brokman et al., ICLR 2025)

+ [HFI: A unified framework for training-free detection and implicit watermarking of latent diffusion model generated images](https://arxiv.org/abs/2412.20704) (Sungik Choi et al, arxiv 2024)

+ [Zero-Shot Detection of AI-Generated Images](https://arxiv.org/abs/2409.15875) (Davide Cozzolino et al., ECCV 2024)[:globe_with_meridians:](https://grip-unina.github.io/ZED/) [:octocat:](https://github.com/grip-unina/ZED/)

+ [Your diffusion model is an implicit synthetic image detector](https://hal.science/hal-04713283v1) (Xi Wang et al, ECCV Workshop 2024)

+ [AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html) (Jonas Ricker et al., CVPR 2024) [:octocat:](https://github.com/jonasricker/aeroblade)

+ [RIGID: A Training-Free and Model-Agnostic Framework for Robust AI-Generated Image Detection](https://arxiv.org/abs/2405.20112) (Davide Cozzolino et al., arxiv 2024)


## Transformation

+ [Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective](https://arxiv.org/abs/2408.06741) (Ouxiang Li, KDD 2025) [:octocat:](https://github.com/Ouxiang-Li/SAFE)

+ [Learning on Less: Constraining Pre-trained Model Learning for Generalizable Diffusion-Generated Image Detection](https://arxiv.org/abs/2412.00665) (Yingjian Chen et al, arxiv 2024)

+ [Frequency Masking for Universal DeepFake Detection](https://arxiv.org/abs/2401.06506) (Chandler Timm Doloriel et al., ICASSP 2024) [:octocat:](https://github.com/chandlerbing65nm/FakeImageDetection)

## knowledge distillation

+ [GenDet: Towards Good Generalizations for AI-Generated Image Detection](https://arxiv.org/abs/2312.08880) (Mingjian Zhu, arxiv 2023)

## Patch

+ [All Patches Matter, More Patches Better: Enhance AI-Generated Image Detection via Panoptic Patch Learning](https://arxiv.org/abs/2504.01396) (Zheng Yang et al., arxiv 2025)

+ [Breaking Semantic Artifacts for Generalized AI-generated Image Detection](https://proceedings.neurips.cc/paper_files/paper/2024/file/6dddcff5b115b40c998a08fbd1cea4d7-Paper-Conference.pdf) (Chende Zhang et al., NeurIPS 2024) [:octocat:](https://github.com/Zig-HS/FakeImageDetection?tab=readme-ov-file)

+ [A Single Simple Patch is All You Need for AI-generated Image Detection](https://arxiv.org/abs/2402.01123) (Jiaxuan Chen, arxiv 2024)[:octocat:](https://github.com/bcmi/SSP-AI-Generated-Image-Detection)

+ [PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection](https://arxiv.org/abs/2311.12397v3) (Nan Zhong et al., arxiv 2023) [:globe_with_meridians:](https://fdmas.github.io/AIGCDetect/) [:octocat:](https://github.com/Ekko-zn/AIGCDetectBenchmark)

+ [Fusing global and local features for generalized ai-synthesized image detection](https://ieeexplore.ieee.org/abstract/document/9897820) (Yan Ju et al., ICIP 2022) [:octocat:](https://github.com/littlejuyan/FusingGlobalandLocal)

+ [What makes fake images detectable? Understanding properties that generalize](https://arxiv.org/abs/2008.10588) (Lucy Chai et al., ECCV 2020) [:globe_with_meridians:](https://chail.github.io/patch-forensics/) [:octocat:](https://github.com/chail/patch-forensics)

## Multimodal Large Language Model

+ [Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation](https://arxiv.org/abs/2503.14905v1) (Siwei Wen et al., arxiv 2025) [:octocat:](https://github.com/opendatalab/FakeVLM)

+ [Explainable Synthetic Image Detection through Diffusion Timestep Ensembling](https://arxiv.org/pdf/2503.06201) (Yixin Wu et al., arxiv 2025)

+ [FakeBench: Probing Explainable Fake Image Detection via Large Multimodal Models](https://arxiv.org/abs/2404.13306) (Yixuan Li at al., arxiv 2024)

## Noise fingerprint

+ [Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html) (Chuangchuang Tan et al., CVPR 2024) [:octocat:](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)

+ [Learning on gradients: Generalized artifacts representation for gan-generated images detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.pdf) (Chuangchuang Tan et al., CVPR 2023) [:octocat:](https://github.com/chuangchuangtan/LGrad)

+ [Detecting generated images by real images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740089.pdf) (Bo Liu et al., ECCV 2022) [:octocat:](https://github.com/Tangsenghenshou/Detecting-Generated-Images-by-Real-Images)

## Data-driven

<!-- 
+ []() (, ) [:globe_with_meridians:]() [:octocat:]()
-->

+ [On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) (Riccardo Corvi et al., ICASSP 2023) [:globe_with_meridians:](https://grip-unina.github.io/DMimageDetection/) [:octocat:](https://github.com/grip-unina/DMimageDetection) 

+ [Global Texture Enhancement for Fake Face Detection in the Wild](https://arxiv.org/abs/2002.00133) (Zhengzhe Liu et al., CVPR 2020) [:octocat:](https://github.com/liuzhengzhe/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild) 

+ [CNNDetect: CNN-generated images are surprisingly easy to spot...for now](https://arxiv.org/abs/1912.11035) (Sheng-Yu Wang et al., CVPR 2020) [:globe_with_meridians:](https://peterwang512.github.io/CNNDetection/) [:octocat:](https://github.com/peterwang512/CNNDetection) 

## Frequency domain

+ [Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](https://arxiv.org/abs/2505.14359) (Ruoxin Chen et al., arxiv 2025)

+ [Any-Resolution AI-Generated Image Detection by Spectral Learning](https://arxiv.org/abs/2411.19417) (Dimitrios Karageorgiou et al., CVPR 2025) 
[:octocat:](https://github.com/mever-team/spai)

+ [Towards More Accurate Fake Detection on Images Generated from Advanced Generative and Neural Rendering Models](https://arxiv.org/abs/2411.08642) (Chengdong Dong et al., arxiv 2024)

+ [Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning](https://ojs.aaai.org/index.php/AAAI/article/view/28310) (Chuangchuang Tan et al., AAAI 2024) [:octocat:](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) 

+ [Leveraging frequency analysis for deep fake image recognition](https://proceedings.mlr.press/v119/frank20a) (Joel Frank et al., ICML 2020) [:octocat:](https://github.com/RUB-SysSec/GANDCTAnalysis) 

+ [Detecting and simulating artifacts in gan fake images](https://arxiv.org/pdf/1907.06515) (Xu Zhang et al., WIFS 2019) [:octocat:](https://github.com/ColumbiaDVMM/AutoGAN) 

# Shadows

+ [Shadows Don't Lie and Lines Can't Bend! Generative Models don't know Projective Geometry...for now](https://arxiv.org/abs/2311.17138) (Ayush Sarkar et al., CVPR24) [:globe_with_meridians:](https://projective-geometry.github.io/) [:octocat:](https://github.com/hanlinm2/projective-geometry/)

## Tools

+ [SIDBench: A Python framework for reliably assessing synthetic image detection methods](https://dl.acm.org/doi/abs/10.1145/3643491.3660277) (Manos Schinas et al., MAD Workshop 2024) [:octocat:](https://github.com/mever-team/sidbench)

+ [PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection](https://arxiv.org/abs/2311.12397v3) (Nan Zhong et al., arxiv 2023) [:globe_with_meridians:](https://fdmas.github.io/AIGCDetect/) [:octocat:](https://github.com/Ekko-zn/AIGCDetectBenchmark)


## Citing Awesome AI-generated Image Detection

If you find this project useful for your research, please use the following BibTeX entry.

```bibtex
@misc{zhai2025awesome,
  author={Zhai, Naixin},
  title={Awesome AI-generated Image Detection},
  howpublished={\url{https://https://github.com/nxZhai/Awesome-AI-generated-Image-Detection}},
  year={2025}
}

```

## Acknowledgments

The page is borrowed from [Awesome Dataset Distillation](https://github.com/Guang000/Awesome-Dataset-Distillation) and [Awesome-Deepfakes-Detection](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection). Thanks for their great work!
