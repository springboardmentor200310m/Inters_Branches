import os

dataset_path = "spectrogram_output2"   # change if your folder name is different

classes = os.listdir(dataset_path)

print("Class Distribution:\n")
for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        print(cls, ":", len(os.listdir(cls_path)))
