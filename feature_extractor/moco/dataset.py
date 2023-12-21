import torch.utils.data
from PIL import Image

class MoCoDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        """x: rows are genes and columns are samples"""
        self.image_list = [s.strip() for s in open(filename)]
        self.transform = transform

    def __getitem__(self, i):
        path = self.image_list[i]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = 0

        return img, label

    def __len__(self):
        return len(self.image_list)

    def crop(self, img, dets):
        if not dets:
            return img
        x, y, w, h = [float(e) for e in dets.split(',')[0:4]]

        W, H = img.size
        x1 = x * W - w * W / 2.0
        x2 = x * W + w * W / 2.0
        y1 = y * H - h * H / 2.0
        y2 = y * H + h * H / 2.0

        return img.crop((x1,y1,x2,y2))

class MyDatasetYoloROI(torch.utils.data.Dataset):
    def __init__(self, input_file, transform):
        self.image_list = []
        self.detections = []
        with open(input_file) as f:
            for s in f:
                a = s.strip().split("\t")
                if len(a) == 2:
                    image_file, yolo_roi = a
                elif len(a) == 1:
                    image_file, yolo_roi = a[0], None
                else:
                    raise ValueError(f"Unknown data: {s}")
                self.image_list.append(image_file)
                self.detections.append(yolo_roi)
        self.transform = transform
        assert len(self.image_list) == len(self.detections)
    def __getitem__(self, i):
        with open(self.image_list[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.crop(img, self.detections[i])
            img = self.transform(img)
        return img, -1
    def __len__(self):
        return len(self.image_list)

    def crop(self, img, dets):
        if not dets:
            return img
        x, y, w, h = [float(e) for e in dets.split(',')[0:4]]

        W, H = img.size
        x1 = x * W - w * W / 2.0
        x2 = x * W + w * W / 2.0
        y1 = y * H - h * H / 2.0
        y2 = y * H + h * H / 2.0

        return img.crop((x1,y1,x2,y2))

