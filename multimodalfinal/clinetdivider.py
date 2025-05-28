import os
import random
import shutil

def copy_selected_images(src_folder, dst_folder, keep_count=300):
    os.makedirs(dst_folder, exist_ok=True)
    all_images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(all_images) <= keep_count:
        selected = all_images
        print(f"[NOTE] Only {len(all_images)} images in {src_folder}, copying all.")
    else:
        selected = random.sample(all_images, keep_count)

    for img in selected:
        shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

def trim_client_folder(client_folder):
    client_name = os.path.basename(client_folder)
    reduced_folder = os.path.join(os.path.dirname(client_folder), f"reduced_{client_name}")
    print(f"\nProcessing {client_name} → reduced_{client_name}")

    for class_name in ['normal', 'pneumonia']:
        src_class_folder = os.path.join(client_folder, class_name)
        dst_class_folder = os.path.join(reduced_folder, class_name)
        copy_selected_images(src_class_folder, dst_class_folder)

# Absolute paths for your clients
client_paths = [
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/client1",
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/client2",
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/client3"
]

# Run for each client
for client in client_paths:
    trim_client_folder(client)
