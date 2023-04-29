import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class ColorPredictor(nn.Module):
    def __init__(self):
        super(ColorPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.output_layers = nn.ModuleList(
            [nn.Linear(128, 4) for _ in range(6)])

    def forward(self, x):
        x = self.fc(x)
        return [output_layer(x) for output_layer in self.output_layers]


def train(model, data, num_epochs=100, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        random.shuffle(data)

        # input_tensor = torch.tensor((255, 0, 0), dtype=torch.float32) / 255.0
        # writer.add_graph(model, input_tensor)
        for rgb, hex_str in data:
            input_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
            target_tensors = [torch.tensor(list(map(int, format(
                int(char, 16), '04b'))), dtype=torch.float32) for char in hex_str]

            optimizer.zero_grad()

            outputs = model(input_tensor)
            loss = sum([criterion(output, target_tensor)
                       for output, target_tensor in zip(outputs, target_tensors)])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data)}")
        writer.add_scalar("Loss/train", loss, epoch)


writer.flush()
writer.close()


def rgb_to_hex(rgb):
    return ''.join([format(val, '02X') for val in rgb])


def hex_to_rgb(hex_str):
    return tuple(int(hex_str[i:i+2], 16) for i in range(0, 6, 2))


def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        rgb = [random.randint(0, 255) for _ in range(3)]
        hex_str = rgb_to_hex(rgb)
        data.append((rgb, hex_str))
    return data


if __name__ == "__main__":
    model = ColorPredictor()
    data = generate_data()
    train(model, data)
    torch.save(model.state_dict(), 'color_predictor.pth')
