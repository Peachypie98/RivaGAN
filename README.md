# RivaGAN PyTorch (Unofficial)

## :hushed: Before We Start...
This repository is created to assist people encountering difficulties running the offical repository from DAI-Lab, as the official one has not received updates for the past several years.
Additionally, I have optimized the official version to enhance execution speed while maintaining overall performance integrity.
Lastly, I have conducted testing on Windows 11 using the latest Python 3.11 and PyTorch 2.0.1

## :grinning: Prerequisites
1. Install PyTorch, Numpy, OpenCV, Pandas, ArgParse
1. Install GitBash from `https://git-scm.com/`
2. Install wget from `https://gnuwin32.sourceforge.net/packages/wget.htm`
3. Install Torch DCT using `pip install torch_dct`

## :zany_face: Let's Get Started!
1. Clone this repository
2. Open GitBash Terminal and Download Hollywood2 Training Dataset  
   Acquiring the dataset may require several hours, depending upon the speed of your internet connection
   ```
   cd data
   bash download.sh
   ```
4. Train RivaGAN Model  
   The hyperparameter settings align with the official specifications and are currently configured to their default values
   ```
   python train.py 
   python train.py --epochs 200 --lr 0.001 --data_dim 64 
   ```
   Default Hyperparameters Details: 
   * --epochs: 300
   * --train_batch: 12
   * --lr: 0.0005
   * --num_workers: 16
   * --data_dim: 32
   * --use_critic: True
   * --use_adversary: True
   * --use_noise: True
   * --use_bit_inverse: True
5. Inference RivaGAN Model  
   After completing the model training, our objective is to encode a data watermark onto a video and subsequently extract it from the encoded footage. After the inference process, it will generate `output_log.txt` file, providing a detailed record of the extracted data from each frame in the video and a watermarked video that contains the data
   
   ```
   python inference.py --model_weight your_weight_path/model.pt
   python inference.py --model_weight your_weight_path/model.pt --random_data No --your_data "1100 1001 0011 0000 1111 0101 1100 0011" --fps 30
   ```
   Default Hyperparameters Details:    
   * --data_dim: 32 
      * The data dimensions must correspond with the dimensions used during the model training
   * --model_weight: `None` 
     * Must be added
   * --random_data: Yes
   * --your_data: `None` 
     * Set `--random_data` to `No` to use your own data
   * --video_location: `./data/hollywood2/val/actioncliptest00002.avi`
   * --fps: 25 
      * Watermaked video output FPS

## Changelog
### 2024-05-23
* `make_pair` function has been reverted to its original code due to instability issues during training

### 2024-04-18
* Incorporated pre-trained RivaGAN model, which was trained using 32-bit data dimensions

### 2023-01-23
* Enhanced code optimization for encoding and decoding processes (3.5 ~ 4x speed increase)
