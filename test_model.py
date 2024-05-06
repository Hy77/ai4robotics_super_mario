import torch
import torchvision.transforms as T
from mario_env import MarioEnv
from cnn_model import MarioCNN
from dqn_model import MarioDQN

# Initialize Mario environment
env = MarioEnv('SuperMarioBros-1-1-v0')

# Set device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CNN and DQN models
cnn = MarioCNN()
dqn = MarioDQN(input_dim=256, output_dim=env.action_space.n)  # Adjust as necessary

# Load model weights
cnn.load_state_dict(torch.load('cnn_models/cnn_model_ver1.pth'))
dqn.load_state_dict(torch.load('dqn_models/dqn_model_ver1.pth'))
cnn.to(device)
dqn.to(device)

# Define image preprocessing pipeline
transform = T.Compose([
    T.ToPILImage(),  # Convert numpy array to PIL image
    T.Resize((84, 84)),  # Resize image to match CNN input dimensions
    T.ToTensor(),  # Convert PIL image to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Initialize environment
state = env.reset()
state = transform(state).unsqueeze(0).to(device)

# Simulate a test loop
done = False
while not done:
    with torch.no_grad():
        # Extract features from the current state using CNN
        state_features = cnn(state)

        # TODO: Determine action from DQN (currently using random actions as a placeholder)
        action = dqn(state_features).max(1)[1].item()
        # action = env.action_space.sample()  # Placeholder for actual DQN decision

        # Execute the chosen action in the environment
        next_state, reward, done, info = env.step(action)
        next_state = transform(next_state).unsqueeze(0).to(device)

        # Update state
        state = next_state

    if done:
        # Reset the environment for the next test
        state = env.reset()
        state = transform(state).unsqueeze(0).to(device)

# Close the environment after testing
env.close()
