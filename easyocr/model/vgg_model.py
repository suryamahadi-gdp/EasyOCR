import torch
import torch.nn as nn
from .modules import VGG_FeatureExtractor, BidirectionalLSTM

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((256, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, x, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
    
    def create_sample_input(self, device):
        return torch.rand((1, 1, 64, 896)).to(device), torch.rand((1, 40)).to(device)
    
    def export_to_onnx(self, output: str, device: str, opset_version: int = 11, verbose=False):
        """
        NOTE: this function only works on CUDA, there is a bug in pytorch
        see: https://github.com/JaidedAI/EasyOCR/issues/746#issuecomment-1186319659
        """
        # create sample input
        sample_input = self.create_sample_input(device)
        self.eval()

        if device != 'cuda':
            raise ValueError("Only CUDA device is supported for now.")

        # export
        torch.onnx.export(
            self.module, 
            sample_input, 
            f=output, 
            do_constant_folding=True,
            verbose=True, 
            opset_version=opset_version, 
            input_names=['input', 'text'], 
            output_names=['output'],
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                    3: 'width'
                },
                'output': {
                    0: 'batch_size',
                    2: 'width'
                }
            }
        )
