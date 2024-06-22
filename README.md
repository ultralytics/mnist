<br>
<a href="https://ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Introduction

Welcome to the repository containing innovative software developed by Ultralytics 🧠. Our code is 🌟 **open-sourced and freely available for redistribution under the AGPL-3.0 license**. For more insight into our work and impact, head over to https://www.ultralytics.com.

[![Ultralytics Actions](https://github.com/ultralytics/mnist/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/mnist/actions/workflows/format.yml)

# 📗 Description

The repository at https://github.com/ultralytics/mnist is our dedicated playground for the MNIST dataset. 🖐 This repository houses sandbox code that allows for experimentation and training of different neural network architectures on the famous MNIST digit database.

# 📦 Requirements

Ensure you have Python 3.7 or later installed on your machine. The following packages are required, and you can install them using pip with the provided command: `pip3 install -U -r requirements.txt`.

- `numpy`: A fundamental package for scientific computing in Python.
- `torch`: PyTorch, an open-source machine learning library for Python.
- `torchvision`: A PyTorch package that includes datasets and model architectures for computer vision.
- `opencv-python`: An open-source computer vision and machine learning software library.

# 🏃‍♂️ Run

To start training on the MNIST digits dataset, execute `train.py` from your Python environment. The training and test data are located in the `data/` folder and were initially curated by Yann LeCun (http://yann.lecun.com/exdb/mnist/).

```python
# Example snippet of train.py to showcase its usage.
# This will set up the environment for training a model on MNIST dataset.

# Import necessary libraries (Make sure they are installed as per requirements)
import torch

# Your training script will start here, initialize models, load data, etc.
# ...

# Start the training process
# ...

# Save your trained model
torch.save(model.state_dict(), "path_to_save_model.pt")

# Add suitable comments to each segment of your code for better understanding.
```

![MNIST Examples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "MNIST digits example")\
_Image credit: [Josef Steppan](//commons.wikimedia.org/w/index.php?title=User:Jost_swd15&action=edit&redlink=1 "User:Jost swd15 (page does not exist)") - [Own work](//commons.wikimedia.org/wiki/File:MnistExamples.png), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0 "Creative Commons Attribution-Share Alike 4.0"), [Link](https://commons.wikimedia.org/w/index.php?curid=64810040)_

# 🤝 Contribute

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your input is invaluable. Take a look at our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started. Also, we'd love to hear about your experience with Ultralytics products. Please consider filling out our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge 🙏 and thank you to all of our contributors!

<!-- Ultralytics contributors -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

# ©️ License

Ultralytics is excited to offer two different licensing options to meet your needs:

- **AGPL-3.0 License**: Perfect for students and hobbyists, this [OSI-approved](https://opensource.org/licenses/) open-source license encourages collaborative learning and knowledge sharing. Please refer to the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for detailed terms.
- **Enterprise License**: Ideal for commercial use, this license allows for the integration of Ultralytics software and AI models into commercial products without the open-source requirements of AGPL-3.0. For use cases that involve commercial applications, please contact us via [Ultralytics Licensing](https://ultralytics.com/license).

# 📬 Contact Us

For bug reports, feature requests, and contributions, head to [GitHub Issues](https://github.com/ultralytics/mnist/issues). For questions and discussions about this project and other Ultralytics endeavors, join us on [Discord](https://ultralytics.com/discord)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
