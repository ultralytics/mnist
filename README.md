<img src="https://storage.googleapis.com/ultralytics/logo/logoname1000.png" width="200">

# Introduction 🚀

Welcome to the software repository developed by Ultralytics! All the code provided here is open-source and available for redistribution under the AGPL-3.0 license. We are dedicated to sharing our work with the community and fostering innovation in the field of machine learning. For more details on our projects, please visit [Ultralytics](https://www.ultralytics.com).

# Description 📜

This repository, [Ultralytics/mnist](https://github.com/ultralytics/mnist), is our sandbox for experimenting with the MNIST dataset, a classic benchmark in the machine learning community. It contains code for training several neural network architectures to recognize handwritten digits, an essential task in the field of computer vision.

# Requirements 🛠️

To get started with this project, you'll need Python 3.7 or later. Install the necessary libraries by running the following command:

```bash
pip3 install -U -r requirements.txt
```

The required packages include:

- `numpy` for numerical computations,
- `torch` for building and training neural network models,
- `torchvision` for accessing pre-trained models and computer vision datasets,
- `opencv-python` for image processing tasks.

# Running the Code 🏃‍♂️

To train the neural network models on the MNIST digit database, execute the `train.py` script through your Python environment:

```python
# Run this command to train the model using train.py
python train.py
```

The training and testing data are located in the `data/` folder, originally sourced from Yann LeCun's website [MNIST database](http://yann.lecun.com/exdb/mnist/).

![MNIST Dataset Examples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "MNIST Dataset Examples")   
_Image credit: [Josef Steppan](https://commons.wikimedia.org/wiki/User:Jost_swd15) - Own work, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), [Link](https://commons.wikimedia.org/w/index.php?curid=64810040)_

# Support 🤝

For any issues or contributions, please open an issue directly in the [GitHub repository](https://github.com/ultralytics/mnist/issues). We appreciate your input and collaboration in improving this project. While we do not provide email support, our GitHub repository is actively maintained, and your feedback is always welcome there.

Thank you for exploring our MNIST sandbox, and happy coding! 🎉
