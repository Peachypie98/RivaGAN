# RivaGAN PyTorch (Unofficial)

## :hushed: Before We Start...
This repository is created to assist people encountering difficulties running the offical repository from DAI-Lab, as the official one has not received updates for the past several years.
Furthermore, I have implemented several modifications to the official version to ensure successful code execution, along with making specific adjustments to enhance overall performance.
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
   python inference.py --model_weight 'weight_path/model.pt'
   python inference.py --data_dim 64 --model_weight weight_path/model.pt --your_data (1,0,1,1)*8 --fps 30
   ```
   Default Hyperparameters Details:    
   * --data_dim: 32 (The data dimensions must correspond with the dimensions used during the model training)
   * --model_weight: None (Must insert)
   * --random_data: True
   * --your_data: None (Set `--random_data` to `False` to use your own data)
   * --video_location: ./data/hollywood2/val/actioncliptest00002.avi
   * --fps: 25 (Watermaked video fps)
