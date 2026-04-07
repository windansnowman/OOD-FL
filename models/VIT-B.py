
from transformers import ViTForImageClassification, ViTFeatureExtractor
from models.simple import SimpleNet
class CustomViT(SimpleNet):
    def __init__(self,num_classes):
        super(CustomViT, self).__init__()
        # 加载预训练的ViT模型
        pretrained_model_name='google/vit-base-patch16-224-in21k'
        self.model = ViTForImageClassification.from_pretrained(pretrained_model_name, num_labels=num_classes)

    def forward(self, x):
        # 使用预训练模型进行前向传播
        outputs = self.model(x)
        return outputs.logits
def VITB(num_classes):

    return CustomViT(num_classes=num_classes)