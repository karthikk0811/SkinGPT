import torch
from PIL import Image
import sys
from langchain_community.llms import CTransformers
from transformers import AutoModel

# Load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local', force_reload=True)

# Function to perform object detection using YOLOv5
def perform_object_detection(image_path):
    img = Image.open(image_path).convert("RGB")
    results = model(img)
    return results

# Function to extract the disease with the highest probability
def extract_diseases(yolov5_results):
    if len(yolov5_results.xyxy[0]) > 0:
        sorted_detections = sorted(yolov5_results.xyxy[0], key=lambda x: x[4], reverse=True)
        highest_confidence_disease = model.names[int(sorted_detections[0][5])]
        return highest_confidence_disease
    else:
        return "No disease detected"

# Function to load the LLaMA model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.Q5_0.gguf",
        # model = AutoModel.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF"),
        model_type="llama",
        max_new_tokens=150,
        temperature=0.5
    )
    return llm
# model = AutoModel.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF")

# Function to generate a response to a query using LLaMA
def generate_response(context, text_query):
    llm = load_llm()
    combined_query = f"{context} {text_query}"
    response = llm(combined_query)
    return response

# Start of the chatbot interaction
if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")  # User provides the image path
    yolov5_results = perform_object_detection(image_path)
    disease = extract_diseases(yolov5_results)
    context = f"The detected disease is {disease}."  # Initial context based on disease detection

    print(context)  # Inform the user about the detected disease

    # Enter a loop for the chatbot interaction
    while True:
        text_query = input("Ask me anything about the disease or type 'exit' to quit: ")
        if text_query.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        response = generate_response(context, text_query)
        print("Response:", response)