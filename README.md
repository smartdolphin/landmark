# Korea landmark classification AI competition 
Development of Artificial Intelligence Algorithm to Classification Landmark Images from Large-Scale Images [[link]](https://dacon.io/competitions/official/235585/overview/).

## Extract korea landmark data
Please download and extract image data in the `public`<br>

    $ cd landmark
    $ mkdir public
    $ wget https://dacon-datasets.s3.ap-northeast-2.amazonaws.com/landmark/data.zip
    $ unzip data.zip -d public

Input data

    └── landmark
        └── public
            └── train
                └── 부산시
                    └── 사직야구장
                        └── xxx.JPG (landmark name)
            └── test
                └── x (name doesn't matter)
                    └── xxx.JPG (landmark name)
            └── category.csv
            └── category.json
            └── train.csv
            └── sample_submission.csv

## Usage
#### Training

Run commands as follows:

````bash
$ cd landmark
$ python main.py --gpu 0 --epochs 100 --batch_size 80 --image_size 224 --model_dir save/test_01 --depth 0
````

#### Testing

Run commands as follows:

````bash
$ cd landmark
$ python main.py --gpu 0 --load_epoch 99 --batch_size 80 --image_size 224 --test --model_dir save/test_01 --test_csv_submission_dir my_submission.csv
````

## Requirements
- python 3.6
- torch 1.4
- albumentations
- torch_optimizer
- efficientnet_pytorch
- matplotlib
- numpy
- pandas
- sklearn
- tqdm
- argparse
