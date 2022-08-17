## Transformer Implementation from ([Vaswani et al.](https://arxiv.org/abs/1706.03762))


## What Are Transformers
Transformers are typically utilized in natural language processing models such as BERT and GPT-3. Similar to LSTMs and Residual Neural Networks, transformers can reference information from previous outputs and inputs so that it allows the model to derive a level of context. This context weighs in on the transformer's future predictions and improves the overall output. 
* The key difference between transformers and previous models such as ResNet and LSTMS, is that the previous models would suffer from vanishing gradients. 
  * Vanishing gradients essentially descibes the phenomenone in which the model is unable to retrieve valuable context from previous data outiside of a certain range. 
  * For example: In the sentence `What kind of trees do you think would grow best in Oregon?` previous models would only be able to derive context from the last 7 words.
* Transformers on the other hand don't suffer from this problem. They can pull a constant flow of previous inputs and outputs that will aid in it determining the best prediction.
  * This gives it a much better long-range dependency

Advantages:
* Transformer models are highly parallelizable which means much better efficiency
* It makes no assumptions about the temporal / spatial relationships across data which makes it ideal for processing a set of objects
* Distant items can affect each other's output without passing through many RNN-steps or through many convolutional layers

Disadvantages:
* For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and the current hidden-state
* If the input has a temporal / spatial relationship (text for example) some positional encoding must be added or the model will pretty much only see a bag of words 


<p align="left">
<img src="data/PIC 1.png"/>
</p>


# Attention Mechanism

The key to understanding transformer models is to understand the attention mechanisms behind it

As the model generates text, word by word, it has the ability to attend to previous words that are related to the generated word (context). 
* Backpropagation is used to teach the model what words to attend to 

Attention mechanisms, in theory and given enough resources, have an infinite attention span and can reference words from the start of the passage. 

<p align="left">
<img src="data/PIC 2.png"/>
</p>

Depiction of Referencing in transformers 

<p align="left">
<img src="data/PIC 3.png"/>
</p>
Depiction of Referencing in ResNets (Note the inability to reference further than the window due to vanishing gradients

# Transformers use an `Encoder` / `Decoder` type architecture

*The* `Encoder` (red) maps an input sequence into a continous vector representation that holds all the learned representations of that input

The `Decoder` (blue) takes that representation and step-by-step generates an output while also being fed the previous outputs recurently until the 'end of sentence' token is generated

<p align="left">
<img src="data/PIC 5.png"/>
</p>

# Step-Wise Breakdown of Transformer Mechanics

## Step 1: Feeding Inputs Into A Word Embedded Layer

This layer converts the input words into learned vectors with continous values that represent an individual word

## Step 2: Inject Positional Information Into The Embeddings

Since transformers have no reccurance like recurrent neural networks, information about the positions of each word relative to one another must be added. 
* The positional encodings are vectors that are calculated with `sine` and `cosine` functions as they are linearly related and thus their relationships are easier for the model to learn
 * For every odd time step, the positional vector is calculated with a `cosine` function 
 * For every even time step, the positional vector is calculated with a `sine` function

These vectors are then added to the input embedding vecotrs from `Step 1`

## Step 3 and 4: The Encoder Layer

The encoder layer's job is to map out the embedded input sequence to an abstract vector representation that holds the learned information for the entire sequence 

The encoder layer contains two submodules:
* `Muli-Headed Attention` which is followed by a `fully connected network`
 * There are also residual connections around the two sub modules and normalization layers after each module

## `Multi-Headed Attention Module`

<p align="left">
<img src="data/PIC 6.png"/>
</p>

The multi-headed attention module applies `self-attention` which allows it to associate each individual word in the input with every other word in the input 
* These associations are represented in a matrix in which each word's relation to another is measured by a numerical value. The higher the value, the better the association

To get the self-attention, the embedded inputs from `Step 2` are fed into three distinct fully-connected layers
* These layers will output the `Query`, `Key`, and `Value` vectors (as seen in the image above `Q`, `K`, `V`)
 * The `Query`, `Key`, and `Value` concept comes from retrieval systems
 * For example: If you search for a video on Youtube, the search engine will map the query against a set of keys (video title, description, etc.) that are associated to potential videos in its database. Then it will present the best matched videos to you (values)
* The queries and keys undergo a dot-product multiplication to generate the matrix that holds the 'related-ness' between the words
* The matrix is then scaled down by dividing it by the square-root of the dimensions of the queries and keys
 * This helps stabilize the model becasue if the matrix values are too large, explosive effects can occur which will over bias certain relationships
* A soft-max layer is then used to get the attention weights. These give probability scores between 0 and 1 and emphasizes the largest values and suppress the lowest ones
 * This will aid in the model's certainty when predicting words based on relations
* The output of the soft-max layer (`Query and Key` matrix) will then be multiplied by the `Value` vector from earlier to get the output vector
 * The higher attention weights from the matrix will keep the value of the words learned as more important while the lower attention weights will decrease the probability that these words will be predicted if they are not related to what is being generated by the decoder model

For this to be a `Multi-Headed` attention process, the `Query`, `Key`, and `Value` need to be split into `N`-vectors. These vectors will go through their own self-attention processes (as described above) seperately. They will generate their own output vectors that will then be concatenated before going through a final linear layer
* In theory, each attention-head would learn something different which in the end would give the whole model more representational ability

In summary, multi-headed attention takes in the positiional input embedding from `Step 2` and gives an ouput vector that has encoded information on how each word should attend to all other words in a sequence

## `Fully Connected Network`

The next step takes the `multi-headed attentiion` output and adds it to the input through a `residual connection`

<p align="left">
<img src="data/PIC 7.png"/>
</p>
The ouput of the concatenation is then fed into a normalization layer and then fed into a point-wise feed forward sequence of layers
* This sequence of layers is a couple of linear layers with ReLu activations in between

The ouput of the point-wise feed forward sequence is then normalized again and combined with the previous normalized output of the concatenated `Positional Input Embeddings` and `Multi-Headed Attention Output` vectors through another residual connection:

<p align="left">
<img src="data/PIC 8.png"/>
</p>



* The residual connection helps the model train by allowing the gradients to flow directly 
* The normalization layers help stabilize the network which in turn help decrease the training time necessariy 
* The point-wise feed forward layer helps further process the outputs, potentially giving it a richer representation


All of this is the `Encoder` section of the model. These representations will help the `Decoder` focus on appropriate words in the input during the decoding process. It's possible to stack the encoder `N` times which will further improve the representations and increase the decoder's power to output the best sequence. 

# The Decoder

## Step 5: Output Embedding and Positional Encoding

Similar to the Positional Input Embedding from the `Encoder` except that the ouputs of the entire model are fed in instead of the inputs

## Step 6: Decoder Multi-Headed Attention 1

The positional output embeddings are fed into the first multi-headed attention module in the decoder. 
* This module computes the attention score for the decoder's input
* Since the decoder is auto regressive, it must be prevented from conditioning to future tokens and should only focus on past tokens

<p align="left">
<img src="data/PIC 9.png"/>
</p>


For example: When computing attention scores on the word `am`, the decoder shouldn't have access to the word `fine` as that word comes after `am` in the sequence
* The word `am` should only have access to itself and the words before it (i.e. `am`, `I`, `<start>`)

The method that prevents the model from calculating attention scores for future words is called `Masking`
* A look-ahead mask is applied to the attention scores before the calculation is done:

<p align="left">
<img src="data/PIC 10.png"/>
</p>

* The masked scores are replaced with negative infinities
  * The reason for this is that when the soft-max layer is applied to the masked scores, the negative infinities are converrted to 0 which means the model won't use those future tokens 
<p align="left">
<img src="data/PIC 11.png"/>
</p>

This masking is the only difference in the first multi-headed attention module

<p align="left">
<img src="data/PIC 12.png"/>
</p>

## Step 7: Decoder Multi-Headed Attention 2

For this module, the inputs are the encoder's multi-headed attention output as the `Query` and `Key` and the decoder's multi-headed attention 1 output as the `Value`
* This process matches the encoder's input to the decoder's input and allows the decoder to choose which attention scores to put focus on more

## Step 8: Point-Wise Feed Forward Sequence

This is the same setup as in the encoder. The output of the Decoder's multi-headed attention 2 module is passed into a point-wise feed forward sequence of linear layers with ReLu activation in between to further process the relations between tokens.

## Step 9: Linear Classifier

The final output of the feed-forward sequence is passed through a linear layer that functions as a classifier
* The classifier is a large as the number of classes you have (Example: 10k classes for 10k words)

The output of the linear classifier is passed onto a soft-max layer that outputs the same size as the classifier but the values  are instead between 0 and 1

The decoder takes the `index` of the highest probability score and that is the predicted word

It then takes that output and adds it to the list of decoder inputs and continues the process until the `<end>` token is predicted

The decoder can also be stacked `N` layers high which will also improve the representational ability of the network. 



## License

MIT License

Copyright (c) 2022 Otavio Pailo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


