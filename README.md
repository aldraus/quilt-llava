# Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos


We generated spatially grounded visual instruction tuning data from educational YouTube videos to train large language and vision assistant in histopathology that can localize the prominent medical regions and reason towards diagnosis.

[[Paper, Arxiv](https://arxiv.org/abs/2312.04746)] 


[Mehmet Saygin Seyfioglu*](https://mehmetsayginseyfioglu.github.io/), [Wisdom Ikezogwo*](https://wisdomikezogwo.github.io/), [Fateme Ghezloo*](https://fghezloo.github.io/), [Ranjay Krishna](https://www.ranjaykrishna.com/index.html), [Linda Shapiro](https://homes.cs.washington.edu/~shapiro/) (*Equal Contribution)


<p align="center">
    <img src="images/quiltllama_fav.png" width="25%"> <br>
</p>




<p align="center">
    <img src="images/quilt_llava2.png" width="90%"> <br>
 
  *Quilt-LLaVA was initialized with the general-domain LLaVA and then continuously trained in a curriculum learning fashion (first biomedical concept alignment then full-blown instruction-tuning). We evaluated LLaVA-Med on standard visual conversation and question answering tasks. We release both stage 1 (Quilt) and stage 2(Quilt-Instruct) training sets as well as our evaluation dataset Quilt-VQA*
</p>


## Release
- Quilt-LLaVA is open-sourced under the X release policy, which does not allow any commercial use. Checkout the [paper](https://arxiv.org/pdf/2312.04746.pdf)
- Alongside Quilt-LLaVA, we also release Quilt-Instruct, our instruction-tuning data generated from educational videos. It is also protected by Y license.
- We also release Quilt-VQA, an evaluation dataset to evaluate generative multi modal histopathology models. 




<p align="center">
    <img src="images/pipeline_clustering.png" width="90%"> <br>
 
  *We have created a grounded image-text dataset from educational histopathology videos on YouTube. The bottom row displays an illustrative example. First, we detect frames that have a stable background. Then we extract the narrators' mouse cursors. Then, we perform spatio-temporal clustering on the mouse pointer locations to obtain dense visual groundings for the narrators' speech. Using this method, we create grounded image-text dataset, from which we generate Quilt-Instruct to train our visual Language Learning Model, Quilt-LLaVA.*
</p>



## Contents
- [Data Download](#data-download)
- [Data Generation](#Data Generation)
- [Training](#training)
- [Evaluation](#evaluation)


### Data Download
| Instruction-Tuning data | Size |
| --- | ---: |
| [Quilt-Instruct](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K) | X MiB |

| Evaluation files | Size |
| --- | ---: |
| [Quilt-VQA](https://huggingface.co/datasets/wisdomik/Quilt_VQA) | 	X MiB |
| [Quilt-VQA Red Circle](https://huggingface.co/datasets/wisdomik/QuiltVQA_RED) | X MiB |

| Raw Mouse Cursor Data | Size |
| --- | ---: |
| [Cursors](some path) | N MiB |

| Image URLS | Size |
| --- | ---: |
| [Images](some path) | N MiB |





### Data Generation
In case if you want to generate the instruction tuning data from scratch, please see quilt-instruct folder.


See quilt-VQA folder for the prompt and helper code to generate the evaluation Quilt-VQA data.

### Training
If you want to skip training and access to our checkpoint, etc. 


# Wiz maybe place some stuff here that you have on NOTES.MD about training etc.








### Evaluation

#Wiz 

Eval on pathvqa
eval on quiltvqa
etc. 





 

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex

@article{saygin2023quilt,
  title={Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos},
  author={Saygin Seyfioglu, Mehmet and Ikezogwo, Wisdom O and Ghezloo, Fatemeh and Krishna, Ranjay and Shapiro, Linda},
  journal={arXiv e-prints},
  pages={arXiv--2312},
  year={2023}
}

@article{ikezogwo2023quilt,
  title={Quilt-1M: One Million Image-Text Pairs for Histopathology},
  author={Ikezogwo, Wisdom Oluchi and Seyfioglu, Mehmet Saygin and Ghezloo, Fatemeh and Geva, Dylan Stefan Chan and Mohammed, Fatwir Sheikh and Anand, Pavan Kumar and Krishna, Ranjay and Shapiro, Linda},
  journal={arXiv preprint arXiv:2306.11207},
  year={2023}
}
```


## Related Projects
- Our model is based on 🌋 LLaVA: Large Language and Vision Assistant so model architecture and training scripts are heavily borrowed from https://github.com/haotian-liu/LLaVA.
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)


[![Code and Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
**Usage and License Notices**: The data, code, and model checkpoints are intended and licensed for research use only. They are also subject to additional restrictions dictated by the Terms of Use: LLaMA, Vicuna and GPT-4 respectively. The data is made available under CC BY NC 4.0. The data, code, and model checkpoints may be used for non-commercial purposes and any models trained using the dataset should be used only for research purposes. It is expressly prohibited for models trained on this data to be used in clinical care or for any clinical decision making purposes.

