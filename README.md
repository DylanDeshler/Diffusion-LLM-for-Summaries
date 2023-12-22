# PyTorch Diffusion

I explain this in the pytorch_diffusion notebook, but basically I ran into a lot of issues trying to work with text diffusion and realized I did not have the theoretical or practical understanding of diffusion I needed to learn anything from working with them. So instead I decided to implement continuous denoising diffusion in pytorch without the help of any fancy diffusion libraries or repos. I hoped it would help me learn a better understanding of the process, that I could apply to text generation later, and it certainly did.

sources:
https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing
which itself cites:
https://github.com/lucidrains/denoising-diffusion-pytorch
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023


I was also confused by the optional assignment on blackboard, didn't know if that was a final project thing so I tossed my code for that in here as well (train.py). I was able to get around 88% accuracy, which was disapointing. Scalling up the efficient net I use along with the image size provides a bonus but that didn't seem like a great solution (shouldn't need 20M+ parameters). The B0 model does well enough. I want to know what the approach that got over 90% was.

The other scripts are related to diffusion LLMs, you do not need to look at them.

KEEPING BELOW FOR CONTINUITY------

# Diffusion-LLM-for-Summaries

I want to take a pre-trained encoder-decoder LLM and update it to an encoder-diffusion-decoder 
LLM as described in the CodeFusion paper and compare its summarization results to the original
architecture. The idea being that allowing the network to “ruminate” on its thoughts in latent 
space should produce better summarizations (or results in general). I will use the pretrained 
BERT encoder-decoder network from HuggingFace with Keras so that all I need to implement is
the diffusion network and possibly finetune the decoder network for summarization.
The tutorial I attached uses the XSum dataset, but I will use the much larger reddit summaries 
dataset (also available in keras). I will do a brief initial study with the pretrained BERT model to 
see which size has the best trade-off between accuracy and training speed, here I will also 
investigate using LoRA for fine tuning.

sources:
https://github.com/XiangLi1999/Diffusion-LM
https://keras.io/examples/nlp/t5_hf_summarization/
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/t5_hf_summarization.ipynb

The CodeFusion paper has a substantial amount of information about training the full encoder-
diffusion-decoder network and finding optimal hyperparameters for generating good results, like 
the number of denoising steps. But I expect to spend the bulk of my project optimizing the 
diffusion network and getting it training properly. Setting up the pretrained network and dataset 
should be easy enough because they are available through HuggingFace and Keras.
