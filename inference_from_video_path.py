import numpy as np

import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from pytorchvideo.data import labeled_video_dataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute,   
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomAdjustSharpness,
    Resize,
    RandomHorizontalFlip
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)



video_transform = Compose([
    ApplyTransformToKey(key="video",
    transform=Compose([
        UniformTemporalSubsample(25),
        Lambda(lambda x: x/255),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        # RandomShortSideScale(min_size=256, max_size=512),
        CenterCropVideo(256),
        RandomHorizontalFlip(p=0.5),
    ]),
    ),
])

class VideoModel(LightningModule):
    def __init__(self, ):
        super(VideoModel, self).__init__()

        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(400, 1)

        self.lr = 1e-3
        self.batch_size = 8
        self.num_worker = 4
        self.num_steps_train = 0
        self.num_steps_val = 0

        # self.metric = torchmetrics.classification.MultilabelAccuracy(num_labels=num_classes)
        self.metric = torchmetrics.Accuracy()
        
        #loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr = self.lr)
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.05, patience=2, min_lr=1e-6)
        # scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {'optimizer': opt,
                'lr_scheduler': scheduler, 
                "monitor": "val_loss"}

    def train_dataloader(self):
        dataset = labeled_video_dataset(train_path, clip_sampler=make_clip_sampler('random', 2),
                                         transform=video_transform, decode_audio=False)

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_worker, pin_memory=True)

        return loader

    def training_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        # label = F.one_hot(label, num_classes=num_classes).float()
        out = self(video)
        print(f"Label: {label}\nPred: {out}")
        print(f">>> Training step No.{self.num_steps_train}:")
        # print("Pred:", out)
        # print("GT:", label)
        # print(f"Pred:\n{out}")
        # print(f"Pred shape:\n{out.shape}")
        # print(f"Label:\n{label}")
        # print(f"Label shape:\n{label.shape}")
        # print(">>> INFO: Computing Training Loss")
        
        loss = self.criterion(out, label.float().unsqueeze(1))
        
        print(f"Loss: {loss}")
        self.num_steps_train += 1
        # print(">>> INFO: Training Loss Computed")
        # print(">>> INFO: Computing Training Metric")
        # metric = self.metric(out, label)
        
        # Below is for Accuracy
        metric = self.metric(out, label.to(torch.int64).unsqueeze(1))
        
        print(f"Accuracy: {metric}")

        values = {"loss": loss,
                "metric": metric.detach()}
        
        self.log_dict({"step_loss": loss,
                        "step_metric": metric.detach()})
        
        return values
        
        # return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        
        self.log('training_loss', loss)
        print(f">>> Epoch end loss: {loss}")
        self.log('training_metric', metric)
        

    def val_dataloader(self):
        dataset = labeled_video_dataset(val_path, clip_sampler=make_clip_sampler('random', 2),
                                         transform=video_transform, decode_audio=False)

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_worker, pin_memory=True)

        return loader

    def validation_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        # label = F.one_hot(label, num_classes=num_classes).float()
        out = self(video)
        print(f"Label: {label}\nPred: {out}")
        # print(">>> INFO: Computing Val Loss")
        print(f">>> Validation step No.{self.num_steps_val}:")

        loss = self.criterion(out, label.float().unsqueeze(1))
        print()
        print(f"Loss: {loss}")
        self.num_steps_val += 1
        # print(">>> INFO: Val Loss Computed")
        # print(">>> INFO: Computing Val Metric")
        # metric = self.metric(out, label)
        # Below is for Accuracy
        
        metric = self.metric(out, label.to(torch.int64).unsqueeze(1))
                
        print(f"Accuracy: {metric}")
        

        return {"loss": loss,
                "metric": metric.detach()}
        
        # return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('val_loss', loss)
        self.log('val_metric', metric)

    def test_dataloader(self):
        dataset = labeled_video_dataset(test_path, clip_sampler=make_clip_sampler('random', 2),
                                         transform=video_transform, decode_audio=False)

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_worker, pin_memory=True)

        return loader

    def test_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        # label = F.one_hot(label, num_classes=num_classes).float()
        out = self.forward(video)
        # metric = self.metric(out, label)

        return {"label": label,
                "pred": out.detach(),}

    def test_epoch_end(self, outputs):
        label=torch.cat([x['label'] for x in outputs]).cpu().numpy()
        pred = torch.cat([x['pred'] for x in outputs]).cpu().numpy()
        # self.log('test_loss', loss)
        # self.log('test_metric', metric)
        
        # Below is for MultiLabelClassification
        # class_labels = label.argmax(axis=1)
        # class_pred = pred.argmax(axis=1)
        
        
        # Below is for BinaryClassification
        class_labels = label
        class_pred = np.where(pred > 0, 1, 0)
        
        print(f">> Label: {class_labels}")
        print(f">> Pred: {class_pred.squeeze()}")
              
        # print(classification_report(class_labels, class_pred))
        
        return {"prediction": class_pred,
                "labels": class_labels}




def word_level_prediction(path_to_model, path_to_video):
    model = torch.load(path_to_model)
    video = EncodedVideo.from_path(path_to_video)

    video_data = video.get_clip(0, 2)
    video_data = video_transform(video_data)

    inputs = video_data["video"].cuda()
    inputs = inputs.unsqueeze(0)
    
    preds = model(inputs).detach().cpu().numpy()
    preds = np.where(preds > 0, 1, 0)
    
    return preds

