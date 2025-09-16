from nicegui import ui, app
from matplotlib import pyplot as plt
import base64
from deepdish import DeepDish
import torch
from torchvision import transforms
from PIL import Image
import io

CONFIG = {
    "run_name": "UnfreezeLastTwoLayers250Epochs",
    "backbone": "dinov2",
    "checkpoint": "ressouces/test_run_cont.pt",
    "device": "cuda",
    "epochs": 250,
    "learning_rate": 4e-4,
    "batch_size": 32,
    "head_hidden_size": 384,
    "head_dropout_p": 0.1,
    "unfreeze_backbone_block_after_n": 10,
    "train_split_percentage": 0.85,
    "validation_split_percentage": 0.05,
}
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

uploaded_img = {'src': None}
result = None

@ui.page('/')
def main_page():
    def handle_upload(e):
        print("handling upload")
        # read the uploaded file into memory
        data = e.content.read()
        # encode as base64 so we can display without saving to disk
        b64 = base64.b64encode(data).decode('utf-8')
        uploaded_img['src'] = f'data:image/*;base64,{b64}'

        make_prediction(data)
        
        ui.navigate.to('/display')

    with ui.header().classes('justify-center items-center'):
        ui.image('ressouces/DeepDishLogo.png').props('fit=contain').classes('mx-auto max-w-xs max-h-16')

    with ui.column().classes('w-full items-center gap-4 mt-8'):
        ui.upload(
            on_upload=handle_upload,
            auto_upload=True,
        ).props('accept="image/*"').classes('bg-blue-500 text-white px-6 py-3 rounded cursor-pointer text-center')

@ui.page('/display')
def display_page():
    with ui.header().classes('justify-center items-center'):
        ui.image('ressouces/DeepDishLogo.png').props('fit=contain').classes('mx-auto max-w-xs max-h-16')
    with ui.column().classes('w-full items-center gap-4 mt-8'):
        ui.image(uploaded_img['src']).classes('max-w-sm max-h-96')
        if result:
            
            # Pie chart
            fat_kcal_ratio = round(result["fat"] * 9 / result["calories"], ndigits=2)
            carb_kcal_ratio = round(result["carb"] * 4 / result["calories"], ndigits=2)
            protein_kcal_ratio = round(result["protein"] * 4 / result["calories"], ndigits=2)
            plot_labels = [f"P: {protein_kcal_ratio*100}%", f"C: {carb_kcal_ratio*100}%", f"F: {fat_kcal_ratio*100}%"]
            with ui.pyplot(figsize=(3, 2)):
                #plt.pie([33,33,33])
                plt.pie([protein_kcal_ratio,carb_kcal_ratio,fat_kcal_ratio], labels=plot_labels)
                ui.label(f'kcal: {result["calories"]}').classes('text-lg font-medium text-center')
                ui.label(f'Protein: {result["protein"]}g  Carbs: {result["carb"]}g  Fats: {result["fat"]}g')
                plt.title("Relative Calorie Distribution")
        ui.button('Back', on_click=lambda: ui.navigate.to('/'))

def make_prediction(image):
    # convert to PIL for PyTorch
    pil_img = Image.open(io.BytesIO(image)).convert('RGB')

    # apply test transform
    tensor = test_transform(pil_img)
    # add batch dimension
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model.forward(tensor)
        print(output)
        values = output.squeeze(0).detach().cpu().numpy()
        class_names = ["calories", "mass", "fat", "carb", "protein"]
        global result
        result = {name: round(abs(float(val)), ndigits=2) for name, val in zip(class_names, values)}
        print(result)
    return result

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*20)
print(device)
print("="*20)
model = DeepDish(5, CONFIG)
model.to(device)
if CONFIG['checkpoint']:
    print("loading checkpoint " + CONFIG['checkpoint'])
    model.load_state_dict(torch.load(CONFIG["checkpoint"], map_location=device))

ui.run()
