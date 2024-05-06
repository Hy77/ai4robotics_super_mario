# Mario Environment Deep Learning Project

This project implements a deep learning framework using a Convolutional Neural Network (CNN) and a Deep Q-Network (DQN) to navigate and make decisions within the Mario environment. The CNN extracts features from the game states, which are then used by the DQN to decide the best actions to perform.

## Prerequisites

- **Python**: Python 3.7 or later.
- **PyTorch**: The project is built using the PyTorch library. Install it using pip:

  ```
  pip install torch torchvision
  ```

- **OpenAI Gym**: This project uses OpenAI Gym for the Mario environment. Install it with:

  ```
  pip install gym
  ```

- **TensorBoard**: For logging and visualizing the training process, TensorBoard is used:

  ```
  pip install tensorboard
  ```
  
- **nes_py**: NES emulator

  ```
  pip install nes_py
  ```

- **gym_super_mario_bros**: Gym environment for Super Mario Bros

  ```
  pip install gym_super_mario_bros
  ``

## Usage

To use this project, follow these steps:

1. To train the model, run the training script:

   ```
   python train_model.py
   ```

2. To test the trained model, run the test script:

   ```
   python test_model.py
   ```

3. To visualize the training process, start TensorBoard:

   ```
   tensorboard --logdir=runs
   ```

   Open your web browser and go to `http://localhost:6006/` to view the TensorBoard dashboard.
