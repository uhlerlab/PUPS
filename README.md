# Pretrained models using the Human Protein Atlas:
- ./checkpoints has the model parameters for all the 22 epochs.
- All results in the paper are based on the model parameters in “splice_isoform_dataset_cell_line_and_gene_split_full-epoch=01-val_combined_loss=0.18.ckpt”

# Compiling & accessing the dataset:
The repository is configured to use the free MongoDB Service to store and serve all data for training and validation. Not only does MongoDB scale well for the hundreds of thousands of immunofluorescent images & ESM-2 feature binaries, it also natively supports hierarchical data structures which is crucial for clean data handling.
- Whenever the dataset is updated or accessed the mongo server must be up and running. To do so open run the command `sudo mongod --dbpath <MONGO_PATH>`
- It is recommended to also download Mongo Compass to visualize the datasets directly from a GUI (screenshots below of the hierarchical structure of the dataset as seen from Compass)
- From the compass UI it is possible to directly create a database https://www.mongodb.com/docs/compass/current/databases/. Before any data can be downloaded or accessed you must first create a database named ‘hpa’.
## Dataset preparation: 
- Down the public tabular subcellular_localization dataset from HPA (https://www.proteinatlas.org/download/subcellular_location.tsv.zip). Note: the tabular dataset is used only for its index of genes for the rest of the pipeline to reference. The rest of the information in the tabular dataset including the localization labels are not used as they only describe the coarse gene level information.
- To populate the proteoform level information (localization labels, cell images, amino acid sequence, ESM-2 representation) refer to dataset/download_data.py. The python code is all set up to run but may take a few days to finish pulling data from the web.
- To create dataset splits for training & evaluation refer to dataset/create_datasets.py. A number of different training / evaluation splits are offered.
## Visualization of train/test data
datasplits_matrix_visualization.ipynb (Figure 2a)

# Model training:
The model is defined in src/model:
- Image inpainting: src/model/nn_unet.py
- Localization prediction using the sequence representation: src/model/nn_multilabel_mlp.py
- Learning protein sequence representation: src/model/nn_light_attention.py
- Full model: src/model/full_model.py
For model training, run ‘python train.py’. Model training progress can monitored with TensorBoard 

# Visualize model performance and protein localization variability
- The following plots are generated using plotLoss.ipynb and plotLoss_holdout2.ipynb (same procedure separately applied to training/Holdout 1 and Holdout 2)
  - Computing protein image prediction losses for all held-out proteins in the test set of Holdout 1 and in Holdout 2 (Figure 2b)
  - Plotting examples of protein image predictions (Figure 2c)
  - Ploting predicted intra-nuclear proportions (Figure 3a)
  - Computing the variability of intra-nuclear proportions across cell lines (Figure 3)
  - Computing the variability of intra-nuclear proportions across single cells of the same cell lines (Figure 4)
  - Gene ontology of the most variable proteins: go_variableProteins.ipynb (Training and Holdout 1); go_variableProteins_holdout2 (Holdout 2)
- spectral_bleed_through.ipynb; guided backprop attribution of model attention and a visualization of model predictions across different proteoforms and cell lines (Supplementary Figure 4b)
 
# Evaluations of the experimental validation
plotLoss_experiment.ipynb (Figure 5)

# Visualization of cell and protein representations (Figure 6)
- latent_proteoform_representation_visualization.ipynb - PCA of protein sequence representations (Figure 6a left panel, Supplemental Figure 12)
- latent_proteoform_nucCytosol.ipynb - PCA of proteins in nucleoplasm, cytosol, or both (Figure 6a right panel)
- shap.ipynb - Shapley analysis (Figure 6b)
- jointEmbedding.ipynb - Image representation (Figure 6c) and joint representation (Figure 6d)
- latent_cell_representation_visualization.ipynb - PCA of image representations for 36 cell lines (Supplemental Figure 13)

