<!-- omit in toc -->
# Multi-Modal-LLMs-in-Agriculture

[![Awesome](https://awesome.re/badge.svg)](https://github.com/JiajiaLi04/Multi-Modal-LLMs-in-Agriculture
) ![](https://img.shields.io/github/stars/JiajiaLi04/Multi-Modal-LLMs-in-Agriculture?style=social)

![](https://img.shields.io/github/last-commit/JiajiaLi04/Multi-Modal-LLMs-in-Agriculture?color=#00FA9A) ![](https://img.shields.io/badge/PaperNumber-67-blue) ![](https://img.shields.io/badge/PRs-Welcome-red) 

A curated list of awesome **Multi-Modal-LLMs-in-Agriculture** papers 🔥🔥🔥. 

Currently maintained by <ins>[Jiajia Li](xx) @ MSU</ins>. 


<!-- omit in toc -->
## How to contribute?

If you have any suggestions or find any missed papers, feel free to reach out or submit a [pull request](https://github.com/JiajiaLi04/Agriculture-Foundation-Models/pulls):

1. Use following markdown format.

```markdown
*Author 1, Author 2, and Author 3.* **Paper Title.**  <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
```
<!-- >1. **Paper Title.** *Author 1, Author 2, and Author 3.* Conference/Journal/Preprint Year. [[pdf](link)]. -->

2. If one preprint paper has multiple versions, please use **the earliest submitted year**.
   
3. Display the papers in a **year descending order** (the latest, the first).


<!-- omit in toc -->
## 🔍 Table of Contents 

- [1. 💁🏽‍♀️ Introduction](#1-️-introduction)
- [2. 💁🏽‍♀️ LLMs](#2-️-llms)
  - [2.1 Amazon](#21-Amazon)
  - [2.2 OpenAI](#22-Openai)
  - [2.3 MetaAI](#23-Metaai)
  - [2.4 Apple](#24-Apple)
  - [2.5 Nvidia](#25-Nvidia)
  - [2.6 Google Deepmind](#22-Google-deepmind)
  - [2.7 Microsoft](#27-Microsoft)
  - [2.8 Alibaba](#28-Alibaba)

- [3. 💁🏽‍♀️ Applications](#2--applications)
  - [3.1 LLMS in agriculture information diagnosis and human-machine interaction peer reviewed](#31-LLMS-in-agriculture-information-diagnosis-and-human-machine-interaction-peer-reviewed)
  - [3.2 LLMS in agriculture information diagnosis and human-machine interaction arXiv](#32-LLMS-in-agriculture-information-diagnosis-and-human-machine-interaction-arXiv)


## 1. 💁🏽‍♀️ Introduction
Why foundation models instead of traditional deep learning models?
- 👉 **Pre-trained Knowledge.** By training on vast and diverse datasets, FMs possess a form of "general intelligence" that encompasses knowledge of the world, language, vision, and their specific training domains.
- 👉 **Fine-tuning Flexibility.** FMs demonstrate superior performance to be fine-tuned for particular tasks or datasets, saving the computational and temporal investments required to train extensive models from scratch.
- 👉 **Data Efficiency.** FMs harness their foundational knowledge, exhibiting remarkable performance even in the face of limited task-specific data, which is effective for scenarios with data scarcity issues. 

## 2. 💁🏽‍♀️ LLMs
### 2.1 Amazon
1. Fan, Haozheng, et al. "HLAT: High-quality Large Language Model Pre-trained on AWS Trainium." arXiv preprint arXiv:2404.10630 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=HLAT%3A+High-quality+Large+Language+Model+Pre-trained+on+AWS+Trainium&btnG=) [[Paper]](https://arxiv.org/abs/2404.10630)
2. Zhang, Zhuosheng, et al. "Multimodal chain-of-thought reasoning in language models." arXiv preprint arXiv:2302.00923 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Multimodal+chain-of-thought+reasoning+in+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2302.00923)
3. Soltan, Saleh, et al. "Alexatm 20b: Few-shot learning using a large-scale multilingual seq2seq model." arXiv preprint arXiv:2208.01448 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Alexatm+20b%3A+Few-shot+learning+using+a+large-scale+multilingual+seq2seq+model&btnG=) [[Paper]](https://arxiv.org/abs/2208.01448)

### 2.2 OpenAI
1. Radford, Alec, et al. "Improving language understanding with unsupervised learning." (2018): 4. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Improving+language+understanding+with+unsupervised+learning&btnG=) [[Website]](https://openai.com/index/language-unsupervised/)
2. Radford, Alec, et al. "Better language models and their implications." OpenAI blog 1.2 (2019). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Better+language+models+and+their+implications&btnG=) [[Website]](https:// openai.com/index/better-language-models/)
3. Brown, Tom B. "Language models are few-shot learners." arXiv preprint ArXiv:2005.14165 (2020). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Language+models+are+few-shot+learners&btnG=) [[Paper]](https://splab.sdu.edu.cn/GPT3.pdf)
4. Achiam, Josh, et al. "Gpt-4 technical report." arXiv preprint arXiv:2303.08774 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Gpt-4+technical+report&btnG=) [[Paper]](https://arxiv.org/abs/2303.08774)
5. OPenAI, “Hello GPT-4o.” https://openai.com/index/hello-gpt-4o/, 2024. [[Website]](https://openai.com/index/hello-gpt-4o/)
6. McAleese, Nat, et al. "Llm critics help catch llm bugs." arXiv preprint arXiv:2407.00215 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Llm+critics+help+catch+llm+bugs&btnG=) [[Paper]](https://arxiv.org/abs/2407.00215)

### 2.3 MetaAI
1. Team, Chameleon. "Chameleon: Mixed-modal early-fusion foundation models." arXiv preprint arXiv:2405.09818 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Chameleon%3A+Mixed-modal+early-fusion+foundation+models&btnG=) [[Paper]](https://arxiv.org/abs/2405.09818)
2. Meta, A. I. "Introducing meta llama 3: The most capable openly available llm to date." Meta AI (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Introducing+meta+llama+3%3A+The+most+capable+openly+available+llm+to+date&btnG=) [[Website]](https://ai.meta.com/blog/meta-llama-3/)
3. Xiong, Wenhan, et al. "Effective long-context scaling of foundation models." arXiv preprint arXiv:2309.16039 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Effective+long-context+scaling+of+foundation+models&btnG=) [[Paper]](https://arxiv.org/abs/2309.16039)
4. Zhou, Chunting, et al. "Lima: Less is more for alignment." Advances in Neural Information Processing Systems 36 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Lima%3A+Less+is+more+for+alignment&btnG=) [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)
5. Iyer, Srinivasan, et al. "Opt-iml: Scaling language model instruction meta learning through the lens of generalization." arXiv preprint arXiv:2212.12017 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Opt-iml%3A+Scaling+language+model+instruction+meta+learning+through+the+lens+of+generalization&btnG=) [[Paper]](https://arxiv.org/abs/2212.12017)
6. Xu, Jing, et al. "Improving open language models by learning from organic interactions." arXiv preprint arXiv:2306.04707 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Improving+open+language+models+by+learning+from+organic+interactions&btnG=) [[Paper]](https://arxiv.org/abs/2306.04707)
7. Izacard, Gautier, et al. "Atlas: Few-shot learning with retrieval augmented language models." Journal of Machine Learning Research 24.251 (2023): 1-43. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Atlas%3A+Few-shot+learning+with+retrieval+augmented+language+models&btnG=#d=gs_cit&t=1724435301294&u=%2Fscholar%3Fq%3Dinfo%3ADThmVHR_I6wJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) [[Paper]](https://www.jmlr.org/papers/v24/23-0037.html)
8. Zhang, Susan, et al. "Opt: Open pre-trained transformer language models." arXiv preprint arXiv:2205.01068 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Opt%3A+Open+pre-trained+transformer+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2205.01068)
9. Fried, Daniel, et al. "Incoder: A generative model for code infilling and synthesis." arXiv preprint arXiv:2204.05999 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Incoder%3A+A+generative+model+for+code+infilling+and+synthesis&btnG=) [[Paper]](https://arxiv.org/abs/2204.05999)

### 2.4 Apple
1. Bachmann, Roman, et al. "4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities." arXiv preprint arXiv:2406.09406 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=4M-21%3A+An+Any-to-Any+Vision+Model+for+Tens+of+Tasks+and+Modalities&btnG=) [[Paper]](https://arxiv.org/abs/2406.09406)
2. Mehta, Sachin, et al. "OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework." arXiv preprint arXiv:2404.14619 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=OpenELM%3A+An+Efficient+Language+Model+Family+with+Open-source+Training+and+Inference+Framework&btnG=) [[Paper]](https://arxiv.org/abs/2404.14619)
3. McKinzie, Brandon, et al. "Mm1: Methods, analysis & insights from multimodal llm pre-training." arXiv preprint arXiv:2403.09611 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Mm1%3A+Methods%2C+analysis+%26+insights+from+multimodal+llm+pre-training&btnG=) [[Paper]](https://arxiv.org/abs/2403.09611)
4. Moniz, Joel Ruben Antony, et al. "ReALM: Reference Resolution As Language Modeling." arXiv preprint arXiv:2403.20329 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=ReALM%3A+Reference+Resolution+As+Language+Modeling&btnG=) [[Paper]](https://arxiv.org/abs/2403.20329)
5. You, Keen, et al. "Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs." arXiv preprint arXiv:2404.05719 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Ferret-UI%3A+Grounded+Mobile+UI+Understanding+with+Multimodal+LLMs&btnG=) [[Paper]](https://arxiv.org/abs/2404.05719)
6. Fu, Tsu-Jui, et al. "Guiding instruction-based image editing via multimodal large language models." arXiv preprint arXiv:2309.17102 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Guiding+instruction-based+image+editing+via+multimodal+large+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2309.17102)
7. You, Haoxuan, et al. "Ferret: Refer and ground anything anywhere at any granularity." arXiv preprint arXiv:2310.07704 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Ferret%3A+Refer+and+ground+anything+anywhere+at+any+granularity&btnG=) [[Paper]](https://arxiv.org/abs/2310.07704)

### 2.5 Nvidia
1. Adler, Bo, et al. "Nemotron-4 340B Technical Report." arXiv preprint arXiv:2406.11704 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Nemotron-4+340B+Technical+Report&btnG=) [[Paper]](https://arxiv.org/abs/2406.11704)
2. Jiang, Yunfan, et al. "Vima: Robot manipulation with multimodal prompts." (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Vima%3A+Robot+manipulation+with+multimodal+prompts&btnG=) [[Paper]](https://openreview.net/forum?id=nkDMZ8yqBt)
3. Wang, Boxin, et al. "Instructretro: Instruction tuning post retrieval-augmented pretraining." arXiv preprint arXiv:2310.07713 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Instructretro%3A+Instruction+tuning+post+retrieval-augmented+pretraining&btnG=) [[Paper]](https://arxiv.org/abs/2310.07713)
4. Huang, Jie, et al. "Raven: In-context learning with retrieval augmented encoder-decoder language models." arXiv preprint arXiv:2308.07922 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Raven%3A+In-context+learning+with+retrieval+augmented+encoder-decoder+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2308.07922)
5. Smith, Shaden, et al. "Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model." arXiv preprint arXiv:2201.11990 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Using+deepspeed+and+megatron+to+train+megatron-turing+nlg+530b%2C+a+large-scale+generative+language+model&btnG=) [[Paper]](https://arxiv.org/abs/2201.11990)

### 2.6 Google Deepmind
1. Reid, Machel, et al. "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context." arXiv preprint arXiv:2403.05530 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Gemini+1.5%3A+Unlocking+multimodal+understanding+across+millions+of+tokens+of+context&btnG=) [[Paper]](https://arxiv.org/abs/2403.05530)
2. Saab, Khaled, et al. "Capabilities of gemini models in medicine." arXiv preprint arXiv:2404.18416 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Capabilities+of+gemini+models+in+medicine&btnG=) [[Paper]](https://arxiv.org/abs/2404.18416)
3. De, Soham, et al. "Griffin: Mixing gated linear recurrences with local attention for efficient language models." arXiv preprint arXiv:2402.19427 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Griffin%3A+Mixing+gated+linear+recurrences+with+local+attention+for+efficient+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2402.19427)
4. Team, Gemma, et al. "Gemma: Open models based on gemini research and technology." arXiv preprint arXiv:2403.08295 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Gemma%3A+Open+models+based+on+gemini+research+and+technology&btnG=) [[Paper]](https://arxiv.org/abs/2403.08295)
5. Chen, Xi, et al. "Pali-3 vision language models: Smaller, faster, stronger." arXiv preprint arXiv:2310.09199 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Pali-3+vision+language+models%3A+Smaller%2C+faster%2C+stronger&btnG=) [[Paper]](https://arxiv.org/abs/2310.09199)
6. Padalkar, Abhishek, et al. "Open x-embodiment: Robotic learning datasets and rt-x models." arXiv preprint arXiv:2310.08864 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Open+x-embodiment%3A+Robotic+learning+datasets+and+rt-x+models&btnG=) [[Paper]](https://arxiv.org/abs/2310.08864)
7. Tu, Tao, et al. "Towards generalist biomedical AI." NEJM AI 1.3 (2024): AIoa2300138. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Towards+generalist+biomedical+AI&btnG=) [[Paper]](https://ai.nejm.org/doi/abs/10.1056/AIoa2300138)

### 2.7 Microsoft
1. Reuter, “Microsoft readies new ai model to compete with google, openai, the information reports,” May 2024. 
2. Sun, Yutao, et al. "You only cache once: Decoder-decoder architectures for language models." arXiv preprint arXiv:2405.05254 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=You+only+cache+once%3A+Decoder-decoder+architectures+for+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2405.05254)
3. Abdin, Marah, et al. "Phi-3 technical report: A highly capable language model locally on your phone." arXiv preprint arXiv:2404.14219 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Phi-3+technical+report%3A+A+highly+capable+language+model+locally+on+your+phone&btnG=) [[Paper]](https://arxiv.org/abs/2404.14219)
4. Yu, Zhaojian, et al. "Wavecoder: Widespread and versatile enhanced instruction tuning with refined data generation." arXiv preprint arXiv:2312.14187 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Wavecoder%3A+Widespread+and+versatile+enhanced+instruction+tuning+with+refined+data+generation&btnG=) [[Paper]](https://arxiv.org/abs/2312.14187)
5. Mitra, Arindam, et al. "Orca 2: Teaching small language models how to reason." arXiv preprint arXiv:2311.11045 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Orca+2%3A+Teaching+small+language+models+how+to+reason&btnG=) [[Paper]](https://arxiv.org/abs/2311.11045)
6. Xiao, Bin, et al. "Florence-2: Advancing a unified representation for a variety of vision tasks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Florence-2%3A+Advancing+a+unified+representation+for+a+variety+of+vision+tasks&btnG=) [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.html)

### 2.8 Alibaba
1. Bai, Jinze, et al. "Qwen technical report." arXiv preprint arXiv:2309.16609 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Qwen+technical+report&btnG=) [[Paper]](https://arxiv.org/abs/2309.16609)
2. Nguyen, Xuan-Phi, et al. "SeaLLMs--Large Language Models for Southeast Asia." arXiv preprint arXiv:2312.00738 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=SeaLLMs--Large+Language+Models+for+Southeast+Asia&btnG=) [[Paper]](https://arxiv.org/abs/2312.00738)


## 3. 💁🏽‍♀️ Applications
### 3.1 LLMS in agriculture information diagnosis and human-machine interaction (peer-reviewed)
1. Qing, Jiajun, et al. "GPT-aided diagnosis on agricultural image based on a new light YOLOPC." Computers and electronics in agriculture 213 (2023): 108168. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=GPT-aided+diagnosis+on+agricultural+image+based+on+a+new+light+YOLOPC&btnG=) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0168169923005562)
2. Zhang, Yongheng, et al. "Building Natural Language Interfaces Using Natural Language Understanding and Generation: A Case Study on Human–Machine Interaction in Agriculture." Applied Sciences 12.22 (2022): 11830. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Building+Natural+Language+Interfaces+Using+Natural+Language+Understanding+and+Generation%3A+A+Case+Study+on+Human%E2%80%93Machine+Interaction+in+Agriculture&btnG=) [[Paper]](https://www.mdpi.com/2076-3417/12/22/11830)
3. Yadav, Sargam, and Abhishek Kaushik. "Comparative study of pre-trained language models for text classification in smart agriculture domain." Advances in Data-driven Computing and Intelligent Systems: Selected Papers from ADCIS 2022, Volume 2. Singapore: Springer Nature Singapore, 2023. 267-279. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Comparative+study+of+pre-trained+language+models+for+text+classification+in+smart+agriculture+domain&btnG=) [[Paper]](https://link.springer.com/chapter/10.1007/978-981-99-0981-0_21)
4. Rezayi, Saed, et al. "AgriBERT: Knowledge-Infused Agricultural Language Models for Matching Food and Nutrition." IJCAI. 2022. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=AgriBERT%3A+Knowledge-Infused+Agricultural+Language+Models+for+Matching+Food+and+Nutrition.&btnG=) [[Paper]](https://www.researchgate.net/profile/Amulya-Yadav-2/publication/362052926_Forecasting_the_Number_of_Tenants_At-Risk_of_Formal_Eviction_A_Machine_Learning_Approach_to_Inform_Public_Policy/links/642eef0320f25554da139319/Forecasting-the-Number-of-Tenants-At-Risk-of-Formal-Eviction-A-Machine-Learning-Approach-to-Inform-Public-Policy.pdf)
5. Veena, G., Vani Kanjirangat, and Deepa Gupta. "AGRONER: An unsupervised agriculture named entity recognition using weighted distributional semantic model." Expert Systems with Applications 229 (2023): 120440.[[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=AGRONER%3A+An+unsupervised+agriculture+named+entity+recognition+using+weighted+distributional+semantic+model&btnG=) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417423009429)
6. Palma, Raul, et al. "Agricultural information model." Information and Communication Technologies for Agriculture—Theme III: Decision. Cham: Springer International Publishing, 2022. 3-36. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Palma%2C+Raul%2C+et+al.+%22Agricultural+information+model.%22+Information+and+Communication+Technologies+for+Agriculture%E2%80%94Theme+III%3A+Decision.+Cham%3A+Springer+International+Publishing%2C+2022.+3-36.&btnG=) [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-84152-2_1)
7. Zhao, Biao, et al. "ChatAgri: Exploring potentials of ChatGPT on cross-linguistic agricultural text classification." Neurocomputing 557 (2023): 126708. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=ChatAgri%3A+Exploring+potentials+of+ChatGPT+on+cross-linguistic+agricultural+text+classification&btnG=) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231223008317)
8. Tzachor, Asaf, et al. "Large language models and agricultural extension services." Nature food 4.11 (2023): 941-948. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Large+language+models+and+agricultural+extension+services&btnG=) [[Paper]](https://www.nature.com/articles/s43016-023-00867-x.pdf)
9. Ray, Partha Pratim. "AI-assisted sustainable farming: Harnessing the power of ChatGPT in modern agricultural sciences and technology." ACS Agricultural Science & Technology 3.6 (2023): 460-462. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=AI-assisted+sustainable+farming%3A+Harnessing+the+power+of+ChatGPT+in+modern+agricultural+sciences+and+technology&btnG=) [[Paper]](https://pubs.acs.org/doi/full/10.1021/acsagscitech.3c00145)
10. Moustafa, Ahmed. "Smart-Insect Monitoring System Integration and Interaction via AI Cloud Deployment and GPT." (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Smart-Insect+Monitoring+System+Integration+and+Interaction+via+AI+Cloud+Deployment+and+GPT&btnG=) [[Paper]](https://scholarworks.uark.edu/csceuht/127/)
11. Weichelt, Bryan P., et al. "The potential of AI and ChatGPT in improving agricultural injury and illness surveillance programming and dissemination." Journal of agromedicine 29.2 (2024): 150-154. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=The+potential+of+AI+and+ChatGPT+in+improving+agricultural+injury+and+illness+surveillance+programming+and+dissemination&btnG=) [[Paper]](https://www.tandfonline.com/doi/full/10.1080/1059924X.2023.2284959)
12. Liang, Zijun, et al. "Harnessing the Power of GPT-3 and LSTM for Natural Language Processing in Agricultural Product News: Focus on Soybeans." 2023 4th International Conference on Computer Engineering and Intelligent Control (ICCEIC). IEEE, 2023. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Harnessing+the+Power+of+GPT-3+and+LSTM+for+Natural+Language+Processing+in+Agricultural+Product+News%3A+Focus+on+Soybeans&btnG=) [[Paper]](https://ieeexplore.ieee.org/abstract/document/10426673)
13. Kumar, S. Selva, et al. "Overcoming LLM Challenges using RAG-Driven Precision in Coffee Leaf Disease Remediation." 2024 International Conference on Emerging Technologies in Computer Science for Interdisciplinary Applications (ICETCS). IEEE, 2024. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Overcoming+llm+challenges+using+rag-driven+precision+in+coffee+leaf+disease+remediation&btnG=) [[Paper]](https://ieeexplore.ieee.org/abstract/document/10543859)
14. Ting, W. A. N. G., et al. "Agricultural technology knowledge intelligent question-answering system based on large language model." Smart agriculture 5.4 (2023): 105. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Agricultural+technology+knowledge+intelligent+question-answering+system+based+on+large+language+model&btnG=) [[Paper]](https://www.smartag.net.cn/EN/abstract/abstract22206.shtml)
15. Stoyanov, Stanimir, et al. "Using LLMs in Cyber-Physical Systems for Agriculture-ZEMELA." 2023 International Conference on Big Data, Knowledge and Control Systems Engineering (BdKCSE). IEEE, 2023. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Using+llms+in+cyber-physical+systems+for+agriculturezemela&btnG=) [[Paper]](https://ieeexplore.ieee.org/abstract/document/10339738)
16. Li, Zehong, et al. "How far are green products from the Chinese dinner table?——Chinese farmers’ acceptance of green planting technology." Journal of Cleaner Production 410 (2023): 137141. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=How+far+are+green+products+from+the+Chinese+dinner+table%3F%E2%80%94%E2%80%94Chinese+farmers%27+acceptance+of+green+planting+technology&btnG=) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0959652623012994)
17. Hu, Yixin, Mansoor Ahmed Koondhar, and Rong Kong. "From traditional to smart: exploring the effects of smart agriculture on green production technology diversity in family farms." Agriculture 13.6 (2023): 1236. [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=From+traditional+to+smart%3A+exploring+the+effects+of+smart+agriculture+on+green+production+technology+diversity+in+family+farms&btnG=) [[Paper]](https://www.mdpi.com/2077-0472/13/6/1236)
18. Rezayi, Saed, et al. "Exploring new frontiers in agricultural nlp: Investigating the potential of large language models for food applications." IEEE Transactions on Big Data (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Exploring+new+frontiers+in+agricultural+nlp%3A+Investigating+the+potential+of+large+language+models+for+food+applications&btnG=) [[Paper]](https://ieeexplore.ieee.org/abstract/document/10637955)




### 3.2 LLMS in agriculture information diagnosis and human-machine interaction (arXiv)
1. De Clercq, Djavan, et al. "Large language models can help boost food production, but be mindful of their risks." arXiv preprint arXiv:2403.15475 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Large+language+models+can+help+boost+food+production%2C+but+be+mindful+of+their+risks&btnG=) [[Paper]](https://arxiv.org/abs/2403.15475)
2. Silva, Bruno, et al. "GPT-4 as an agronomist assistant? Answering agriculture exams using large language models." arXiv preprint arXiv:2310.06225 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=GPT-4+as+an+agronomist+assistant%3F+Answering+agriculture+exams+using+large+language+models&btnG=) [[Paper]](https://arxiv.org/abs/2310.06225)
3. Yang, Xianjun, et al. "Pllama: An open-source large language model for plant science." arXiv preprint arXiv:2401.01600 (2024).[[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Pllama%3A+An+open-source+large+language+model+for+plant+science&btnG=) [[Paper]](https://arxiv.org/abs/2401.01600)
4. Gupta, Aman, et al. "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture." arXiv preprint arXiv:2401.08406 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=RAG+vs+Fine-tuning%3A+Pipelines%2C+Tradeoffs%2C+and+a+Case+Study+on+Agriculture&btnG=) [[Paper]](https://arxiv.org/abs/2401.08406)
5. Zhang, Nan, et al. "Large Language Models for Explainable Decisions in Dynamic Digital Twins." arXiv preprint arXiv:2405.14411 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Large+Language+Models+for+Explainable+Decisions+in+Dynamic+Digital+Twins&btnG=) [[Paper]](https://arxiv.org/abs/2405.14411)
6. Jiang, Shufan, et al. "Fine-tuning BERT-based models for plant health bulletin classification." arXiv preprint arXiv:2102.00838 (2021). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Fine-tuning+BERT-based+models+for+plant+health+bulletin+classification&btnG=) [[Paper]](https://arxiv.org/abs/2102.00838)
7. Peng, Ruoling, et al. "Embedding-based retrieval with llm for effective agriculture information extracting from unstructured data." arXiv preprint arXiv:2308.03107 (2023).[[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=bedding-based+retrieval+with+llm+for+effective+agriculture+information+extracting+from+unstructured+data&btnG=) [[Paper]](https://arxiv.org/abs/2308.03107)
8. Wu, Jing, et al. "Extended agriculture-vision: An extension of a large aerial image dataset for agricultural pattern analysis." arXiv preprint arXiv:2303.02460 (2023). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Extended+agriculture-vision%3A+An+extension+of+a+large+aerial+image+dataset+for+agricultural+pattern+analysis&btnG=#d=gs_cit&t=1724433549683&u=%2Fscholar%3Fq%3Dinfo%3AgjvJokI44XYJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) [[Paper]](https://arxiv.org/abs/2303.02460)
9. Deforce, Boje, Bart Baesens, and Estefanía Serral Asensio. "Leveraging Time-Series Foundation Models in Smart Agriculture for Soil Moisture Forecasting." arXiv preprint arXiv:2405.18913 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=Leveraging+Time-Series+Foundation+Models+in+Smart+Agriculture+for+Soil+Moisture+Forecasting&btnG=) [[Paper]](https://arxiv.org/abs/2405.18913)
10. Wu, Yiqi, et al. "GPT-4o: Visual perception performance of multimodal large language models in piglet activity understanding." arXiv preprint arXiv:2406.09781 (2024). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=GPT-4o%3A+Visual+perception+performance+of+multimodal+large+language+models+in+piglet+activity+understanding&btnG=) [[Paper]](https://arxiv.org/abs/2406.09781)
11. Darapaneni, Narayana, et al. "LSTM-RASA Based Agri Farm Assistant for Farmers." arXiv preprint arXiv:2204.09717 (2022). [[Google Scholar]](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C23&q=LSTM-RASA+Based+Agri+Farm+Assistant+for+Farmers&btnG=) [[Paper]](https://arxiv.org/abs/2204.09717)


