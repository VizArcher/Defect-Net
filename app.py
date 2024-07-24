
# 1. Imports and class names setup ###
import gradio as gr
import os
import torch
from torchvision import transforms
from functions import model_builder
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['defected', 'not_defected']

# 2. Model and transforms preparation ###
defectnet = model_builder.CNN(input_shape=3,
                              hidden_units=10,
                              output_shape=len(class_names)).to("cpu")

defectnet_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor()])

# Load save weights
defectnet.load_state_dict(
    torch.load(
        f="cnn_model.pth",
        map_location=torch.device("cpu") # load the model to the cpu
    )
)

# 3. Predict function 

def predict(img) -> Tuple[Dict, float]:
  # Start a timer
  start_time = timer()

  img = defectnet_transforms(img).unsqueeze(0) # add batch dimension

  # Put model into eval mode, make prediction
  defectnet.eval()
  with torch.inference_mode():
    # Pass the transformed image through the model and turn the prediction logits into probability
    pred_probs = torch.softmax(defectnet(img), dim=1)

  # Creating a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

# 4. Gradio app 

# Create title, description and article strings
title = "Defect - Net"
description = "A CNN based 3D printer defect detection"
# article = "Created."

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")], # fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description) # , article=article)

# Launch 
demo.launch(debug=False)
