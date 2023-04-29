import torch
import torch.nn as nn
from color_predictor import ColorPredictor, hex_to_rgb, rgb_to_hex


def load_model(model_path):
    model = ColorPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_color(model, rgb):
    input_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
    input_tensor = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0) / 255.0
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

    hex_digits = ''.join([format(val.item(), 'X') for val in predicted])
    return hex_digits


if __name__ == "__main__":
    model_path = 'color_predictor.pth'
    model = load_model(model_path)

    # Test the model with a sample input
    rgb_input = (127, 200, 100)
    hex_output = predict_color(model, rgb_input)
    actual_hex = rgb_to_hex(rgb_input)
    print(
        f"Input RGB: {rgb_input}\nPredicted Hex: {hex_output}\nActual Hex: {actual_hex}")
