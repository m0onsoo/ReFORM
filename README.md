# ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attentive Recommendation
![Image](https://github.com/user-attachments/assets/c64af119-80d2-40e8-a224-f084c08dbdae)
 This is the PyTorch implementation by <a href='https://github.com/m0onsoo'>@m0onsoo</a> for ReFORM framework proposed in this [paper]():

 >**ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attentive Recommendation**  
 >Moonsoo Park\
 >*IJCAI 2025*




## 📝 Environment
You can run the following command to download the codes faster:
```bash
git clone --depth 1 https://github.com/m0onsoo/ReFORM.git
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

😉 The codes are developed based on the [RLMRec](https://github.com/HKUDS/RLMRec) framework.

## 📚 Text-attributed Recommendation Dataset (TBD)

We utilized two public datasets to evaluate ReFORM:  *Yelp* and *Google Restaurant*.

Each user and item has a generated text description.

First of all, please **download the data** by running following commands.
 ```
 cd data/
 wget https://archive.org/download/reform_data/data.zip
 unzip data.zip
 ```


Each dataset consists of a training set, a validation set, and a test set. During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.
```
- yelp(google restaurant)
|--- trn_mat.pkl            # training set (sparse matrix)
|--- val_mat.pkl            # validation set (sparse matrix)
|--- tst_mat.pkl            # test set (sparse matrix)
|--- usr_emb_factors.pkl    # text description of users
|--- itm_emb_factors.pkl    # text description of items
|--- usr_emb_mean.pkl       # user text embeddings
|--- itm_emb_mean.pkl       # item text embeddings
```

### User/Item Profile
- Each profile is a **high quality text description for each factor** of a user/item.
- Both user and item profiles are generated from **Large Language Models** from their own reviews.
<!-- - The `user profile` (in `usr_prf.pkl`) shows the particular types of items that the user tends to prefer. 
- The `item profile` (in `itm_prf.pkl`) articulates the specific types of users that the item is apt to attract.  -->


### Semantic Representation
- Each user and item has a semantic embedding encoded from its own profile using **Text Embedding Models**.
- The encoded semantic embeddings are stored in `usr_emb_factors.pkl`, `itm_emb_factors.pkl`, `usr_emb_mean.pkl` and `itm_emb_mean.pkl`.
- `usr_emb_factors.pkl` and `itm_emb_factors.pkl` are two-dimensional lists containing embeddings for each factor, and `usr_emb_mean.pkl` and `itm_emb_mean.pkl` are the means of embeddings for each factor.

### Original Data

The original data of our dataset can be found from following links (thanks to their work):
- Yelp: https://business.yelp.com/data/resources/open-dataset/
- Google Restaurant: https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants

🤗 Welcome to use our processed data to improve your research!

## 🚀 Examples to run the codes

The command to evaluate the backbone models and ReFORM is as follows. 

  - Backbone **(LightGCN)**

    ```python encoder/train_encoder.py --model lightgcn --dataset {dataset} --cuda 0```   
  
  - ReFORM

    ```python encoder/train_encoder.py --model reform --dataset {dataset} --cuda 0```
  
Supported models/datasets:

* model_name:  `gccf`, `lightgcn`, `sgl`, `simgcl`, `rlmrec_con`
* dataset: `yelp`, `google`

Hypeparameters:

* The hyperparameters of each model are stored in `encoder/config/modelconf`.

 **For advanced usage of arguments, run the code with --help argument.**

## 🔮 Profile Generation and Semantic Representation Encoding
Here we provide some examples with *Yelp* Data to generate user/item profiles and semantic representations.

Firstly, we need to complete the following three steps.
- Install the openai library `pip install openai`
- Prepare your **OpenAI API Key**
- Enter your key on `Line 8` of these files: `generation\profile\{item/user}\generate_{item/user}_profile.py`.

Then, here are the commands to generate the desired output with examples:

  - **Item Profile Generation**:

    ```python generation/profile/item/generate_item_profile.py```  

  - **User Profile Generation**:

    ```python generation/profile/user/generate_user_profile.py```

  - **Semantic Representation**:

    ```python generation/emb/gen_factor_emb.py --name {usr/itm}```

For semantic representation encoding, you can also try other text embedding models.

😀 The **factor descriptions** we designed are in the `generate_{item/user}_profile.py` files. You can modify them according to your requirements and generate the desired output!

## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@inproceedings{TBD
}
```
