This is the project of compression text using visual encoders. The core idea is to use visual embedder for text embeddings from text encoder to compress the text tokens and save the accuracy for text reconstruction. You can use cvlm conda env.

Current achitecture use unfreezed text encoder, vision encoder and decoder with learnable projectors from different hidden size dims. After some experiments i use for stability training curriculum by cr, use more aggresive lr.

Also we need to write the text based on reults in directory /home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm. I need that text will be present all results with strongest side, minimize the questions for reviewers and look correctness. Also write on russian language. 

