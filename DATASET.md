## Datasets for few-shot learning

Simply execute:
```bash
sh dataset/get_tier_and_mini.sh
```
You are all set! 

----
The following are just some documents to get you more comfortable about datasets.

- miniImagenet
    
    - Thrid-party instructions (OpenAI's Reptile) 
    [here](https://github.com/openai/supervised-reptile/blob/master/fetch_data.sh) (update:
    do **NOT** use it as it is too large; it will download the original large images from ImageNet).
  Google drive file [here](
https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view) to directly 
download the `mini-imagenet.zip` file (we use this). 

    - The splits  `test.csv, train.csv, val.csv` (**already there if you clone our repo**) can be 
downloaded from [Ravi and Larochelle - splits](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet). 
For more information on how to obtain the images check the original source 
[Ravi and Larochelle - github](https://github.com/twitter/meta-learning-lstm).




- tierImagenet
    
    - Original info [here](https://github.com/renmengye/few-shot-ssl-public#tieredimagenet).

### Structure

The `dataset` folder looks like this:
    
    few-shot-ctm
    ├── ...
    └── dataset
       |__ data_loader.py
       |__ ...
       
       |__ miniImagenet                
          └── images            # (ignored in repo)
             ├── n0153282900000006.jpg
             ├── n1313361300001299.jpg
             └── ...
          |__ tes.csv
          |__ train.csv
          |__ val.csv
          
       |__ tier_split
          |__ train.csv
          ...
          
       |__ tier_imagenet        # (ignored in repo)
          |__ train_images_png.pkl 
          |__ train_labels.pkl
          |__ ...

