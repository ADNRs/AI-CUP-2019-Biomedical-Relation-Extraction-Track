# AI-CUP-2019-Biomedical-Relation-Extraction-Track

AI CUP 2019 Biomedical Relation Extraction Track 5th place Solution.

## Dependency
+ Python 3.6 or higher
+ tensorflow-gpu 1.13.1
+ keras 2.2.4
+ numpy 1.16.3
+ pandas 0.24.2
+ sklearn 0.21.3
+ keras-bert 0.79.0
  + Once keras-bert is installed, it is needed to go to its installation directory and replace "pretrained.py" with the one in this repository.
  + If your version of keras-bert is not 0.79.0, you may have to solve the problem which I didn't encounter by yourself.
+ nltk 3.4.1

## Usage
1. Move **training.tsv** and **test.tsv** to **datasets/**.
2. Run **main.py**.
3. You will see the results in **prediction.tsv**.

## Preprocessing
+ Convert all greek letters to the corresponding one in English.
+ Convert all uppercase letters to lowercase.
+ Rename target genes to genea and geneb.

## Model
| Layer (type)                    | Output Shape      | Param #   | Connected to                                                 |
| ------------------------------- | ----------------- | --------- | ------------------------------------------------------------ |
| input_1 (InputLayer)            | (None, 100)       | 0         | -                                                            |
| input_2 (InputLayer)            | (None, 100)       | 0         | -                                                            |
| bert_embedding (Model)          | (None, 100, 768)  | 108851880 | input_1\[0]\[0]<br />input_2\[0]\[0]                          |
| remove_mask_1 (RemoveMask)      | (None, 100, 768)  | 0         | bert_embedding\[0]\[0]                                       |
| bidirectional_1 (Bidirectional) | (None, 100, 1024) | 5251072   | remove_mask_1\[0]\[0]                                        |
| conv1d_1 (Conv1D)               | (None, 100, 384)  | 1180032   | bidirectional_1\[0]\[0]                                      |
| conv1d_2 (Conv1D)               | (None, 100, 384)  | 1573248   | bidirectional_1\[0]\[0]                                      |
| conv1d_3 (Conv1D)               | (None, 100, 384)  | 1966464   | bidirectional_1\[0]\[0]                                      |
| max_pooling1d_1 (MaxPooling1D)  | (None, 25, 384)   | 0         | conv1d_1\[0]\[0]                                             |
| max_pooling1d_2 (MaxPooling1D)  | (None, 25, 384)   | 0         | conv1d_2\[0]\[0]                                             |
| max_pooling1d_3 (MaxPooling1D)  | (None, 25, 384)   | 0         | conv1d_3\[0]\[0]                                              |
| concatenate_1 (Concatenate)     | (None, 25, 1152)  | 0         | max_pooling1d_1\[0]\[0]<br />max_pooling1d_2\[0]\[0]<br />max_pooling1d_3\[0]\[0] |
| flatten_1 (Flatten)             | (None, 28800)     | 0         | concatenate_1\[0]\[0]                                        |
| dropout_1 (Dropout)             | (None, 28800)     | 0         | flatten_1\[0]\[0]                                            |
| dense_1 (Dense)                 | (None, 25)        | 720025    | dropout_1\[0]\[0]                                            |

## Trained model weights

You can download the trained weights [here](https://drive.google.com/file/d/1uqp5E-ODTPFHtMLiBXQOf2OfUH9TYHkJ/view). Once the download finished, move the **bert-bilstm512-textcnn384.3.4.5-100-weights.h5** file to **models/**.

## Note
+ I trained this model on my GTX 1060 6G very hard. It's better to use GPUs with more memory capacity.
+ Because of the NDA, I'm not allowed to provide datasets on GitHub.
