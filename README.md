# A Transformer made fully from scratch
### An attempt to build and train a transformer model from scratch using pytorch.

This was  a REAALLLY FUN thing to do. I've implemented two types of transformers here:

- An Encoder-Decoder model
- A Decoder-only model

Encoder decoder models are used where the input and output are related to each other in terms of semantics but differ in length or structure, like text summarization or translation!!
- Here, the encoder processes the input tokens using the self-attention mechanism and passes it on to the decoder 
- The decoder combines it along with the previously generated output tokens (through masked self-attention) using the cross-attention mechanism to give us the output. 

(Masked self attention means that the decoder can only "see" the previously generated output tokens and not the future ones, hence preventing it from "cheating".)

Decoder only models are used in cases where the input and output data are closely related, like text generation.
- ChatGPT is a decoder only model.

#### Tokenization
- Tokenization was done using 2 tools - SpaCy and AutoTokenizer. Probably the only thing thats not made entirely from scratch is the tokenizer.

#### Dataset
 - The Encoder-Decoder model was trained on the opus books dataset for english to french translation, and the Decoder only model I trained with a bunch of Kendrick Lamar song lyrics
- I only trained the models with a minimal amount of data for sanity checking
- Since transformers are super data hungry, I'd reccommend using a gpu on the cloud or something to train it with as much data as you can for the best results.

#### Loss, and stuff
- The loss values for both the models were around 5, which is good provided that I didnt train them with much data.

#### Why Pytorch??
-  Because I prefer it over TensorFlow, no other reason. I'm sure TensorFlow is great and stuff, but im confortable with torch. 

Thats it byeeee