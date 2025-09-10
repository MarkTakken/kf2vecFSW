
# kf2vec 
k-mer frequency to vector

<!-- k-mer frequency to distance-->

This repository contains the code of the software.

Data and scripts are available in https://github.com/noraracht/kf2vec_data.

INSTALLATION
-----------

## kf2vec is available on Bioconda and can be installed: 

~~~bash
# To create a clean  environment
conda create -n kf2vec_env python=3.11
conda activate kf2vec_env

# To install package
conda install -c bioconda kf2vec
~~~

## To run kf2vec as a module:

1. Clone github repo
2. Navigate to the directory kf2vec
3. Run to create an environment
4. Run kf2vec as a module

~~~bash 
conda env create -f=kf2vec_osx64_v2.yml -n kf2vec_env
OR
conda env create -f=kf2vec_osx64_v2_nobuilds.yml -n kf2vec_env
~~~

Then, each time you want to run, activate an environment using
~~~bash
conda activate kf2vec_env
python -m kf2vec.main <function>
~~~
Below, we assume you are in the `kf2vec` directory of the repository.
We tested our software on macOS and Linux systems.

COMMANDS
-----------
<!--
Combination function to perform backbone preprocessing and training classifier and distance models 
------------
It's a wraper function that consequtively runs computation of k-mer frequences for backbone sequences, splits backbone tree into subtrees and produce corresponding true distance matrices as well as trains classifier and distance models. 
```
 python main.py build_library -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -size 800 -tree $INPUT_PHYLOGENY -mode subtrees_only -cl_epochs 1 -di_epochs 1
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is **-k** which is a k-mer length, set to 7 by default. This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy. Optional parameter is **-p** corresponds to number of processors that Jellyfish can utilize to preprocess input sequences.

**$INPUT_PHYLOGENY** is an input backbone phylogenetic tree in .newick/.nwk format that should be split into multiple smaller subtrees. **-size** parameteter is the user spacified subtree size. We set -size default to 850 but in practice we recommend user to define it. **-mode** parameter can take values full_only, hybrid (default), subtrees_only and specifies whether distance matrices should be computed only for a full backbone tree, subtrees or both. This command requires [TreeCluster](https://github.com/niemasd/TreeCluster) to be installed as a dependancy.

Next set of optional parameters are dealing with conditions for training classifier model. These parametrs are equivalent to parameters used by `train_classifier` function. Thus **-cl_epochs** specifies maximum number of training epochs (default is 2000), **-cl_hidden_sz** is a dimension of hidden layer in a model (default is 2048), **-cl_batch_sz** identifies batch size (default values is 16), **-cl_lr**, **-cl_lr_min** and **-cl_lr_decay** refer to starting learning rate, minimum allowed learning rate and learning rate decay values. We suggest to keep learning rate paramaters at their default values unless user has a specific need to modify them. **-cl_seed** is random seed for training classifier (default is 16).

Final set of optional parameters are related to conditions for training distance model. These parametrs are equivalent to parameters used by `train_model_set` function. Thus **-di_epochs** specifies maximum number of training epochs (default is 8000), **-di_hidden_sz** is a dimension of hidden layer in the model (default is 2048), **-di_embed_sz** is embedding dimension (default is 1024), **-di_batch_sz** identifies batch size (default values is 16), **-di_lr**, **-di_lr_min** and **-di_lr_decay** refer to starting learning rate, minimum allowed learning rate and learning rate decay values. We suggest to keep learning rate paramaters at their default values unless user has a specific need to modify them. **-di_seed** is random seed for training distance models (default is 16).
###### Output: 
All output files from this command are stored in **$OUTPUT_DIR**. 

This command generates normalized k-mer frequencies for every entry in the **$INPUT_DIR**. For every entry it outputs corresponding single file (comma delimited) with extention `.kf`. Next this command will compute subtrees (file with extension `.subtrees` that lists every leaf of a phylogeny and its corresponding subtree number) and corresponding true distance matrices (files named `*subtree_INDEX.di_mtrx`). Output includes a classifier model called `classifier_model.ckpt` and distance models for every subtree.

Combination function to perform query preprocessing, classification and distance computation
------------
It's a wraper function that consequtively runs computation of k-mer frequences for query sequences, classifies query into subtress and computes distances between queries and a corresponding backbone sequences in a subtree.
```
 python main.py process_query_data -input_dir $INPUT_DIR  -output_dir $OUTPUT_DIR -classifier_model $CL_MODEL_DIR -distance_model $DI_MODEL_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is **-k** which is a k-mer length, set to 7 by default. Obviously k-mer length for backbone and query sequences should be the same. This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy. Optional parameter is **-p** corresponds to number of processors that Jellyfish can utilize to preprocess input sequences.
**-classifier_model** corresponds to directory where classifier model is stored and **-distance_model** is a location of distance models. If used in combination with `build_library` all model files should be preserved in the same output directory precified in `build_library` command. **-cl_seed** and **-di_seed** are random seeds classification and query (default is 16).
###### Output: 
All output files from this command are stored in **$OUTPUT_DIR**. 

Output includes results of classification and distance computation. Thus `classes.out` tab delimited file contains information about each query sequence, assigned subtree number and probability values for top as well as all other classes. Distance values are summarized in as query per backbone sequences distance matrix for each subtree.

-->

<!--## Main commands -->

Version number and help
------------
To obtain the version number or invoke the description of commands:
```
 kf2vec --version
 kf2vec --help
 OR
 python -m kf2vec.main --version
 python -m kf2vec.main --help
```

Extracting k-mer frequencies
------------
To obtain k-mer frequencies for backbone species and a query set, the user can execute the get_frequencies command:
```
 kf2vec get_frequencies -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. The optional parameter is **-k**, which is a k-mer length, set to 7 by default. The optional parameter **-p** corresponds to a count of processors that the software can utilize to preprocess input sequences. Optional parameters include **-pseudocount** that adds 0.5 count to each k-mer count before normalization and **-raw_cnt** that outputs k-mer frequencies without normalization. At its core, kf2vec  uses [Jellyfish](https://github.com/gmarcais/Jellyfish) to efficiently count k-mers in sequence data.
###### Output: 
This command generates normalized k-mer frequencies for every entry in the **$INPUT_DIR**. For every entry, it outputs a corresponding single file (comma-delimited) with extension `.kf` that is stored in **$OUTPUT_DIR**.

Split phylogeny into subtrees 
------------
We recommend generating subtrees for a phylogeny with a number of leaves > 4000 using the `divide_tree` command:
```
 kf2vec divide_tree -size $SUBTREE_SIZE -tree $INPUT_PHYLOGENY
```
###### Input: 
**$INPUT_PHYLOGENY** is an input phylogenetic tree in .newick/.nwk format that should be split into multiple smaller subtrees. **-size** parameter is the user-specified subtree size. We set **-size** default to 850, but in practice, we recommend that the user define it. Internally, this command relies on [TreeCluster](https://github.com/niemasd/TreeCluster).
###### Output: 
The output is a text file (extension `.subtrees`) that lists every leaf of a phylogeny and its corresponding subtree number.

Ground truth distance matrix computation 
------------
To compute the distance matrix for the backbone phylogeny:
```
kf2vec get_distances -tree $INPUT_PHYLOGENY  -subtrees $FILE.subtrees
```
###### Input: 
**$INPUT_PHYLOGENY** is an input phylogenetic tree in .newick/.nwk format. **$FILE.subtrees** is the file where each input genome has an assigned subtree number. If a distance matrix corresponds to a single tree, it can be treated as a single clade (clade 0) and provided as input to this command. Under the hood, the distance computation command uses [TreeSwift](https://github.com/niemasd/TreeSwift). 
###### Output: 
The output is saved in a directory where the phylogeny is located.

Scale phylogeny 
------------
To scale phylogeny by multiplying all branch lengths by a user-specified factor:
```
kf2vec scale_tree -tree $INPUT_PHYLOGENY  -factor $SCALE_VALUE
```
###### Input: 
**$INPUT_PHYLOGENY** is an input phylogenetic tree in .newick/.nwk format. **$SCALE_VALUE** is a scaling factor by which all branch lengths will be multiplied. Internally, it uses [TreeSwift](https://github.com/niemasd/TreeSwift). 
###### Output: 
The output is a file with the suffix `r$FACTOR`, saved in the same directory as the phylogeny file.

Training a classifier model
------------
To train a classifier model, one can use the following command:
```
 kf2vec train_classifier -input_dir $INPUT_DIR -subtrees $FILE.subtrees -e 2000 -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain a k-mer frequency count file for backbone species in `.kf` format (output of the `get_frequencies` command). **$FILE.subtrees** is the file where each input genome has an assigned target subtree number. Optional model training parameters include: **-e** number of epochs (default is 2000), **-hidden_sz** dimension of hidden layer (default value is 2048), and **-batch_sz** identifies batch size (default value is 16). **-lr**, **-lr_min** and **-lr_decay** refer to starting learning rate, minimum allowed learning rate, and learning rate decay values. We suggest keeping learning rate parameters at their default values unless the user has a specific need to modify them. **-seed** is the random seed (default 28). **$OUTPUT_DIR** is the directory where the classifier model will be saved once training is complete. 
###### Output: 
The output is a classifier model called `classifier_model.ckpt` stored in a user-defined output repository.

Classification of queries into subtrees
------------
Command to classify query sequences into subtrees:
```
 kf2vec classify -input_dir $INPUT_DIR -model $MODEL_DIR -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain a k-mer frequency count file for the query species in `.kf` format (output of the `get_frequencies` command). **$MODEL_DIR** is the folder where model named `classifier_model.ckpt` is located. **$OUTPUT_DIR** is the directory where `classes.out` will be stored. **-seed** is the random seed (default 28). 
###### Output: 
The output is a `classes.out` tab-delimited file stored in a user-defined repository. The file contains information about each query sequence, assigned subtree number, and probability values for top as well as all other classes.

Train models for subtrees
------------
To train:
```
kf2vec train_model_set -input_dir $INPUT_DIR  -true_dist $TRUE_DIST_MATRIX_DIR  -subtrees $FILE.subtrees -e 4000 -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain a k-mer frequency count file for backbone species in `.kf` format (output of the `get_frequencies` command). **$TRUE_DIST_MATRIX_DIR** is a directory where true distance matrices are located (the location where `*subtree_INDEX.di_mtrx` files are). **$FILE.subtrees** is the file where each input genome has an assigned subtree number. Model training parameters include: **-e** number of epochs (default is 8000), **-hidden_sz** is a dimension of the hidden layer in the model (default is 2048), **-embed_sz** is the embedding dimension (default is 1024),  **-batch_sz** identifies batch size (default value is 16). **-lr**, **-lr_min**, and **-lr_decay** refer to starting learning rate, minimum allowed learning rate, and learning rate decay values. We suggest keeping learning rate parameters at their default values unless the user has a specific need to modify them. **-clade** is the clade number to train the model for. If the clade number is not provided, the models are trained for all clades consecutively. **-seed** is the random seed (default is 28). **$OUTPUT_DIR** is the directory where `model_subtree_INDEX.ckpt` will be stored. 
###### Output: 
The output is a set of trained models for each input subtree.

Query subtree models
------------
To query models:
```
kf2vec query -input_dir $INPUT_DIR  -model $MODEL_DIR  -classes $CLASSES_DIR -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain k-mer frequency count files for the query species in `.kf` format (output of `get_frequencies` command). **$MODEL_DIR** is the folder where model named `model_subtree_INDEX.ckpt` is located. **$CLASSES_DIR** is the directory where `classes.out` is located. **$OUTPUT_DIR** is the directory where `apples_input_di_mtrx_query_INDEX.csv` will be stored. **-seed** is the random seed (default is 28).
###### Output: 
The output is a query per backbone sequences distance matrix for subtrees.

Generate chunked inputs for backbone species in the training set
------------
To obtain k-mer frequencies for backbone species and a query set, the user can execute the get_frequencies command:
```
 kf2vec get_chunks -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. The optional parameter is **-k**, which is a k-mer length, set to 7 by default. The optional parameter **-p** corresponds to a count of processors that the software can utilize to preprocess input sequences. Optional parameters include **-pseudocount** that adds 0.5 count to each k-mer count before normalization.
###### Output: 
This command generates a single chunked sample for every entry in the **$INPUT_DIR**. Each output is a matrix where rows correspond to generated chunks and columns are k-mer count for a corresponding chunk (not normalized). Every output file has an extension `.kf` added and is stored in **$OUTPUT_DIR**.

Training a classifier model for chunked input
------------
To train a classifier model for chunked input, one can use the following command:
```
 kf2vec train_classifier_chunks -input_dir $INPUT_DIR -input_dir_fullgenomes $INPUT_DIR_FULL -subtrees $FILE.subtrees -e 2000 -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain a k-mer counts file (chunked input) for backbone species in `.kf` format (output of the `get_chunks` command). **$INPUT_DIR_FULL** is an input directory that should contain the k-mer frequencies for full genomes of backbone species in `.kf` format (output of the `get_frequencies` command). **$FILE.subtrees** is the file where each input genome has an assigned target subtree number. Optional model training parameters include: **-e** number of epochs (default is 2000), **-hidden_sz** dimension of hidden layer (default value is 2048) and **-batch_sz** identifies batch size (default value is 16). **-lr**, **-lr_min** and **-lr_decay** refer to starting learning rate, minimum allowed learning rate, and learning rate decay values. We suggest keeping learning rate parameters at their default values unless the user has a specific need to modify them. **-seed** is the random seed (default 28). **-cap** reads input values as an unsigned 8-bit integer to reduce memory consumption while training. **$OUTPUT_DIR** is the directory where the classifier model will be saved once training is complete. 
###### Output: 
The output is a classifier model called `classifier_model.ckpt` stored in a user-defined output repository.

Training embedder models for chunked input
------------
To train:
```
kf2vec train_model_set_chunks -input_dir $INPUT_DIR -input_dir_fullgenomes $INPUT_DIR_FULL -true_dist $TRUE_DIST_MATRIX_DIR  -subtrees $FILE.subtrees -e 4000 -o $OUTPUT_DIR
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain k-mer counts files (chunked input) for backbone species in `.kf` format (output of the `get_chunks` command). **$INPUT_DIR_FULL** is an input directory that should contain the k-mer frequencies for full genomes of backbone species in `.kf` format (output of the `get_frequencies` command). **$TRUE_DIST_MATRIX_DIR** is a directory where true distance matrices are located (the location where `*subtree_INDEX.di_mtrx` files are). **$FILE.subtrees** is the file where each input genome has an assigned subtree number. Model training parameters include: **-e** number of epochs (default is 8000), **-hidden_sz** is a dimension of the hidden layer in the model (default is 2048), **-embed_sz** is the embedding dimension (default is 1024),  **-batch_sz** identifies batch size (default value is 16). **-lr**, **-lr_min**, and **-lr_decay** refer to starting learning rate, minimum allowed learning rate, and learning rate decay values. We suggest keeping learning rate parameters at their default values unless the user has a specific need to modify them. **-clade** is the clade number to train the model for. If the clade number is not provided, the models are trained for all clades consecutively. **-seed** is the random seed (default is 28). **-cap** reads input values as an unsigned 8-bit integer to reduce memory consumption while training. **$OUTPUT_DIR** is the directory where `model_subtree_INDEX.ckpt` will be stored. 
###### Output: 
The output is a set of trained models for each input subtree.



TOY EXAMPLE
-----------

To test the step-by-step workflow on a toy dataset:
------------
While located in the code directory

1. To extract k-mer frequencies from backbone and query sequences:
```
kf2vec get_frequencies -input_dir toy_example/train_tree_fna -output_dir toy_example/train_tree_kf
kf2vec get_frequencies -input_dir toy_example/test_fna -output_dir toy_example/test_kf
```

2. To split the tree into subtrees and compute ground truth distance matrices:
```
kf2vec divide_tree -tree /toy_example/train_tree_newick/train_tree.nwk -size 2
kf2vec get_distances -tree /toy_example/train_tree_newick/train_tree.nwk  -subtrees  /toy_example/train_tree_newick/train_tree.subtrees
```
The `divide tree` command generates a file with extension `.subtrees` where the clade number for each sample is specified. Columns are space seperated and can be modified manually.

Get distances takes as input phylogeny `.nwk` and subtree information `.subtrees` files, and generates corresponding distance matrices in the same folder where the phylogeny is. Distance matrices are named with the suffix `subtree_cladeNumber`.

If a distance matrix is required for the entire phylogeny, we suggest increasing the `size` parameter in a divide tree such that entire tree is represented as a single clade 0 and compute the distance matrix. See example file `train_tree_single_clade.subtrees`.

3. To train the classifier model:
```
kf2vec train_classifier -input_dir /toy_example/train_tree_kf -subtrees /toy_example/train_tree_newick/train_tree.subtrees -e 10 -o /toy_example/train_tree_models
```

4. To classify query sequences:
```
kf2vec classify -input_dir /toy_example/test_kf -model /toy_example/train_tree_models -o /toy_example/test_results
```

5. To train distance models:
```
kf2vec train_model_set -input_dir /toy_example/train_tree_kf -true_dist /toy_example/train_tree_newick  -subtrees /toy_example/train_tree_newick/train_tree.subtrees -e 10 -o /toy_example/train_tree_models
```

   Single clade example
```
kf2vec train_model_set -input_dir /toy_example/train_tree_kf -true_dist /toy_example/train_tree_newick  -subtrees /toy_example/train_tree_newick/train_tree.subtrees -e 10 -clade 0 -o /toy_example/train_tree_models
```

6. To compute distances from backbone to query sequences:
```
kf2vec query -input_dir ../toy_example/test_kf  -model ../toy_example/train_tree_models -classes ../toy_example/test_results  -o ../toy_example/test_results
```

To scale the backbone phylogeny by a factor before splitting into subtrees: 
------------
This step is OPTIONAL, but might be helpful in practice
```
kf2vec scale_tree -tree ../toy_example/train_tree_newick/train_tree.nwk  -factor 100
```
This step scales all branch lengths in backbone phylogeny by a specific factor (x100 in the example above). Software adds the suffix `rFACTOR` (_r100.0) to the original phylogeny filename and saves the output into the same directory. 

To test with chunked input on toy dataset:
------------
While located in code directory

1. To generate chunked input for backbone training sequences (this step takes a couple of minutes, run with -p 20 to speed up):
```
kf2vec get_chunks -input_dir ../toy_example/train_tree_fna -output_dir ../toy_example/train_tree_chunks
```
2. To train a classifier model for chunked input:
```
kf2vec train_classifier_chunks -input_dir ../toy_example/train_tree_chunks -input_dir_fullgenomes ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -o ../toy_example/train_tree_models -cap
```
3. To train the embedder model for chunked input:
```
# Train on clade 1 and 0
kf2vec train_model_set_chunks -input_dir ../toy_example/train_tree_chunks -input_dir_fullgenomes ../toy_example/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -o ../toy_example/train_tree_models -clade 1 0

# Train only on clade 1
kf2vec train_model_set_chunks -input_dir ../toy_example/train_tree_chunks -input_dir_fullgenomes ../toy_example/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -o ../toy_example/train_tree_models -clade 1
```


<!--
To test wrapper functions on toy dataset:
------------
While located in code directory
1. To preprocess data and train models:
```
python main.py build_library -input_dir ../toy_example/train_tree_fna -output_dir ../toy_example/combo_models -size 2 -tree ../toy_example/train_tree_newick/train_tree.nwk -mode subtrees_only -cl_epochs 10 -di_epochs 1
```
2. To preprocess queries and compute distances:
```
python main.py process_query_data -input_dir ../toy_example/test_fna -output_dir ../toy_example/combo_results   -classifier_model ../toy_example/combo_models -distance_model ../toy_example/combo_models
```
-->


