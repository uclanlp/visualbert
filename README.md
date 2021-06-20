This repository contains code for the following two papers:

+ [VisualBERT: A Simple and Performant Baseline for Vision and Language (arxiv)](https://arxiv.org/abs/1908.03557) with a short version titiled [What Does BERT with Vision Look At?](https://www.aclweb.org/anthology/2020.acl-main.469/) published on ACL 2020.

   Under the folder `visualbert` is code (the original VisualBERT), where we pre-train a Transformer for vision-and-language (V&L) tasks on image-caption data.

+ [Unsupervised Vision-and-Language Pre-training Without Parallel Images and Captions](https://arxiv.org/abs/2010.12831) published on NAACL 2021.

   Under the folder `unsupervised_visualbert` is code (Unsupervised VisualBERT), where we pre-train a V&L transformer without aligned image-captions pairs. Rather, we pre-training only using unaligned images and text, and achieve competitive performance with many models supervised with aligned data.

The model VisualBERT has been also integrated into several libararies such as [Huggingface Transformer](https://huggingface.co/transformers/model_doc/visual_bert.html) (many thanks to [Gunjan Chhablani](https://github.com/gchhablani) who made it work) and [Facebook MMF](https://github.com/facebookresearch/mmf).

Thanks~