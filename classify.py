import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, transforms
import torchvision.models as models
from sklearn import svm
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
from sklearn.metrics import accuracy_score
import pickle


# ----------------------ML model trainning--------------------------------
def heq(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # _Convert_to_HSV_colorspace
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    Meq_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Meq_color


def tranform_image(DATA_DIR):
    target = []
    images = []
    flatten_data = []

    CATEGORIES = os.listdir(DATA_DIR)

    for categories in CATEGORIES:
        class_num = CATEGORIES.index(categories)  # label encoding
        path = os.path.join(DATA_DIR, categories)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_array = heq(img_array)
            img_resized = resize(img_array, (160, 160, 3))
            flatten_data.append(img_resized.flatten())  # Normalizes values
            images.append(img_resized)
            target.append(class_num)

    flatten_data = np.array(flatten_data)
    target = np.array(target)
    images = np.array(images)

    return flatten_data, target


def build_ml_model():
    path_ = "train"
    X_train, y_train = tranform_image(f"Images/{path_}")
    X_test, y_test = tranform_image("Images/test")
    model = svm.SVC(
        C=100, class_weight=None, gamma="auto", kernel="rbf", probability=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if "data.pkl" not in os.listdir("Artifacts"):
        pickle.dump(model, open("Artifacts/data.pkl", "wb"))
    return acc * 100


# ------------------------------trainning DL model---------------------


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(weights=models.vgg.VGG19_Weights.DEFAULT)
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vgg19(x)
        return x


def build_model(num_epochs, lr):
    def test_model(model, test_loader, criterion):
        model.eval()  # Set the model to evaluation mode
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Get the device of the model parameters
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                model = model.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)

        average_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100.0
        return average_loss, accuracy

    def train_and_test(
        model, train_loader, test_loader, criterion, optimizer, num_epochs
    ):
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Get the device of the model parameters

        global average_loss
        global test_loss
        global test_acc
        global accuracy
        global epoch

        # progress bar
        progress_text = "Training... Please wait."
        bar = st.progress(0, text=progress_text)

        for epoch in range(1, num_epochs + 1):
            torch.cuda.empty_cache()
            gc.collect()
            model.train()  # Set the model to training mode
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for inputs, labels in train_loader:
                model = model.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Clear the gradients

                try:
                    outputs = model(inputs)
                except:
                    inputs = inputs.squeeze(dim=0)
                    outputs = model(inputs)

                try:
                    loss = criterion(outputs.logits, labels)
                except:
                    loss = criterion(outputs, labels)

                try:
                    _, predicted = torch.max(outputs, 1)
                except:
                    _, predicted = torch.max(outputs.logits, 1)

                loss.backward()  # Backpropagation
                optimizer.step()  # Update the model parameters

                total_loss += loss.item() * inputs.size(0)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)

            average_loss = total_loss / total_samples
            accuracy = (total_correct / total_samples) * 100.0

            # Test the model
            test_loss, test_acc = test_model(model, test_loader, criterion)

            print(
                f"epoch={epoch}\ntrain={accuracy:.3f},{average_loss:.3f}\ntest={test_acc:.3f},{test_loss:.3f}"
            )

            bar.progress(epoch * 2)

            st.sidebar.write(
                f"epoch={epoch}\ntrain={accuracy,average_loss}\ntest={test_acc,test_loss}"
            )

        return model

    path_ = "train"

    tranformer = Compose([transforms.Resize([299, 299]), transforms.ToTensor()])

    train_dataset = ImageFolder(f"Images/{path_}/", transform=tranformer)
    val_dataset = ImageFolder("Images/test/", transform=tranformer)

    BATCH_SIZE = len(os.listdir(f"Images/test/{os.listdir('Images/test/')[0]}"))

    if BATCH_SIZE > 32:
        BATCH_SIZE = 16

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

    torch.cuda.empty_cache()
    gc.collect()
    model = VGG19(len(os.listdir(f"Images/{path_}")))
    train_loader = train_dataloader
    test_loader = test_loader
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_epochs = num_epochs

    # Train and test the model
    if "data.pt" not in os.listdir("Artifacts/"):
        model_ = train_and_test(
            model, train_loader, test_loader, criterion, optimizer, num_epochs
        )

    if model_ != None and "data.pt" not in os.listdir("Artifacts/"):
        try:
            torch.save(model_, "Artifacts/data.pt")
        except:
            model_ = train_and_test(
                model, train_loader, test_loader, criterion, optimizer, num_epochs
            )
            torch.save(model_, "Artifacts/data.pt")

    return accuracy, test_acc, average_loss, test_loss


# ------------------------------------roi------------------------------------


def roi(img_array):
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img_array = img_array[y : y + h, x : x + w]
        return img_array


# -----------------Inferencing DL model---------------------------

tranformer2 = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize([299, 299]), transforms.ToTensor()]
)


def Classifier(img):
    model = torch.load("Artifacts/data.pt").eval()
    img = tranformer2(img)
    res = model((img).unsqueeze(dim=0).to("cuda"))
    return (
        torch.argmax(res, dim=1).to("cpu").detach().numpy(),
        torch.softmax(res, dim=1).to("cpu").detach().numpy(),
    )


# -----------------Inferencing DL model---------------------------
def Classifier_ml(img):
    flatten_data = []
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = roi(img_array)
    img_array = heq(img_array)
    img_resized = resize(img_array, (160, 160, 3))
    flatten_data.append(img_resized.flatten())
    model = pickle.load(open("Artifacts/data.pkl", "rb"))
    res = model.predict_proba(flatten_data)
    return res[0]
