# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader


class Classifier():
    def __init__(self,image_size,min_face_size,dropout_prob):
        """
         Initialize MTCNN ResNeXt and InceptionResnet for face img to embeding conversion
         
         @param image_size - size of image used for mtcnn
         @param min_face_size - minimum face size used for resnet
         @param dropout_prob - probability of embedding of face img to
        """
        self.data_path = "Artifacts/data.pt"

        self.mtcnn = MTCNN(image_size=image_size, margin=0, min_face_size=min_face_size)

        # initializing resnet for face img to embeding conversion
        self.resnet = InceptionResnetV1(pretrained="vggface2",dropout_prob=dropout_prob).eval()

        with open('.tmp.txt','r') as f:
            path_=f.read()
            if path_=='Web Cam':
                path_='train'

        self.dataset = datasets.ImageFolder(f"Images/{path_}")  # photos folder path


    def train_model(self):
        """
        Train resnet model to classify photos and peoples. Args : dataset : dataset class_to_idx : mapping from class id to
        """
        # initializing mtcnn for face detection

        idx_to_class = {
            i: c for c, i in self.dataset.class_to_idx.items()
        }  # accessing names of peoples from folder names

        loader = DataLoader(self.dataset, collate_fn=lambda X: X[0])

        face_list = []  # list of cropped faces from photos folder
        name_list = []  # list of names corrospoing to cropped photos
        embedding_list = (
            []
        )  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

        # returns the embedding matrix for each face in the loader
        for img, idx in loader:
            face, prob = self.mtcnn(img, return_prob=True)
            # if face is not None and prob 0. 90 and porbability 90% then the face is cropped to a resnet model and porbability 90%
            if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                emb = self.resnet(
                    face.unsqueeze(0)
                )  # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(
                    emb.detach()
                )  # resulten embedding matrix is stored in a list
                name_list.append(idx_to_class[idx])  # names are stored in a list

        # save

        data = [embedding_list, name_list]
        torch.save(data, self.data_path)  # saving data.pt file


    def face_match(self,img):
        """
        Match a face to the person. This is a wrapper around mtcnn to return the face and the prob of the match
        
        Args:
            img: image to be matched to person
        
        Returns: 
            tuple of name and distance of the face in the image as a tuple ( name distance ) or None if no
        """
        face, prob = self.mtcnn(img, return_prob=True)  # returns cropped face and probability
        # returns a tuple of name_list min distance distance for each person in the embedding data.
        if face != None:
            emb =self.resnet(
                face.unsqueeze(0)
            ).detach()  # detech is to make required gradient false

            saved_data = torch.load(self.data_path)  # loading data.pt file
            embedding_list = saved_data[0]  # getting embedding data
            name_list = saved_data[1]  # getting list of names
            dist_list = (
                []
            )  # list of matched distances, minimum distance is used to identify the person

            # Add a dist to the list of embedding distributions.
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
            return (name_list[idx_min], min(dist_list))

