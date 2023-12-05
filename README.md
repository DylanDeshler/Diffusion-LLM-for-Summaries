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
The CodeFusion paper has a substantial amount of information about training the full encoder-
diffusion-decoder network and finding optimal hyperparameters for generating good results, like 
the number of denoising steps. But I expect to spend the bulk of my project optimizing the 
diffusion network and getting it training properly. Setting up the pretrained network and dataset 
should be easy enough because they are available through HuggingFace and Keras.
