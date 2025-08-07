# ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attentive Recommendation

 This is the PyTorch implementation by Anonymous for ReFORM framework

 >**ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attentive Recommendation**  
 >Anonymous




## 📝 Environment
You can run the following command to download the codes faster:
```bash
git clone --depth 1 https://github.com/Anonymous/ReFORM.git
```

Then run the following commands to create a conda environment:

```bash
conda create -y -m reform python=3.10
conda activate reform

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

pip install pyyaml tqdm
pip install transformers
```

😉 The codes are developed based on the RLMRec framework.  
🔎 Other Contents-based GCN baselines in the paper used the MMRec framework.

## 📚 Text-attributed Recommendation Dataset

We utilized two public datasets to evaluate ReFORM:  *Yelp* and *Google Restaurants*.


First of all, please **download the data** (Yelp/Google Restaurants) from [Google Drive](https://drive.google.com/drive/folders/17WYUnoX0SGo3bFN0w5Sbt23rEUjIjnYC?usp=sharing).  
Please put these data under the **data** directory.

The data contains the coo matrix separated into train, validate, and test, as well as the transformed text embeddings.  
During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.
```
- yelp(google restaurants)
|--- trn_mat.pkl            # training set (sparse matrix)
|--- val_mat.pkl            # validation set (sparse matrix)
|--- tst_mat.pkl            # test set (sparse matrix)
|--- usr_emb_factors.pkl    # user text embeddings
|--- itm_emb_factors.pkl    # user text embeddings
```

### Semantic Representation
- Each user and item has a semantic embedding encoded from its own profile using **BERT**.
- The encoded semantic embeddings are stored in `usr_emb_factors.pkl` and `itm_emb_factors.pkl`.
- `usr_emb_factors.pkl` and `itm_emb_factors.pkl` are two-dimensional lists containing embeddings for each factor.

### Original Data

The original data of our dataset can be found from following links:
- Yelp: https://business.yelp.com/data/resources/open-dataset/
- Google Restaurants: https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants

<!-- 🤗 Welcome to use our processed data to improve your research! -->

## 🚀 Examples to run the codes

The command to evaluate the backbone models and ReFORM is as follows. 

  - Backbone **(LightGCN)**

    ```python encoder/train_encoder.py --model lightgcn --dataset {dataset} --cuda 0```   
  
  - **ReFORM**

    ```python encoder/train_encoder.py --model reform --dataset {dataset} --cuda 0```
  
Supported models/datasets:

* model_name:  `gccf`, `lightgcn`, `sgl`, `simgcl`, `rlmrec`, `reform`
* dataset: `yelp`, `google`

Hypeparameters:

* The hyperparameters of each model are stored in `encoder/config/modelconf`.

 **For advanced usage of arguments, run the code with --help argument.**

## 🔮 Factor-specific Profile Generation Profile Encoding
Here we provide some examples with *Yelp* Data to generate user/item profiles and semantic representations.  
You can also download the input files for profile creation and the generated profile from the [Google Drive](https://drive.google.com/drive/folders/17WYUnoX0SGo3bFN0w5Sbt23rEUjIjnYC?usp=sharing).  
Place the files in the **generation** folder in their respective locations.

Firstly, we need to complete the following three steps.
- Install the openai library `pip install openai`
- Prepare your **OpenAI API Key**
- Enter your key on `Line 8` of these files: `generation\profile\{item/user}\generate_{item/user}_profile.py`.

Then, here are the commands to generate the desired output with examples:

  - **Item Profile Generation**:

    ```python generation/profile/item/generate_item_profile.py```  

  - **User Profile Generation**:

    ```python generation/profile/user/generate_user_profile.py```

  - **Profile Encoding**:

    ```python generation/emb/gen_factor_emb.py --name {usr/itm}```

For profile encoding, you can also try other text embedding models.

😀 The **factor descriptions** we designed are in the `generate_{item/user}_profile.py` files. You can modify them according to your requirements and generate the desired output!

<!-- ## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@inproceedings{TBD
}
``` -->
