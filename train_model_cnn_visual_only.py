from mario_env import MarioEnv
from cnn_model import MarioCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

# Set up TensorBoard
writer = SummaryWriter('runs/mario_experiment')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment and the model
env = MarioEnv('SuperMarioBros-1-1-v0')
cnn = MarioCNN().to(device)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Define image preprocessing
transform = T.Compose([
    T.ToPILImage(),  # Convert numpy array to PIL image
    T.Resize((84, 84)),  # Resize image to match CNN input
    T.ToTensor(),  # Convert PIL image to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Training loop
state = env.reset()
state = transform(state).unsqueeze(0).to(device)  # Preprocess and add batch dimension

for step in range(5000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = transform(next_state).unsqueeze(0).to(device)  # Preprocess the next state

    # Use CNN to extract features
    state_features = cnn(state)
    next_state_features = cnn(next_state)

    # Assume the target features are randomly initialized, for demonstration
    target = torch.rand_like(state_features)

    # Calculate loss
    loss = criterion(state_features, target)
    writer.add_scalar('Loss', loss.item(), step)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Prepare for the next step
    state = next_state

    if terminated or truncated:
        state = env.reset()
        state = transform(state).unsqueeze(0).to(device)

    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item()}')

env.close()
writer.close()

'''
OUTPUTS:

Step 0, Loss: 0.3099452555179596
Step 100, Loss: 0.09610064327716827
Step 200, Loss: 0.0919470563530922
Step 300, Loss: 0.09442157298326492
Step 400, Loss: 0.08084771782159805
Step 500, Loss: 0.09403703361749649
Step 600, Loss: 0.08207416534423828
Step 700, Loss: 0.09366093575954437
Step 800, Loss: 0.09716380387544632
Step 900, Loss: 0.08888061344623566
Step 1000, Loss: 0.08279009163379669
Step 1100, Loss: 0.08116091787815094
Step 1200, Loss: 0.08707574754953384
'''