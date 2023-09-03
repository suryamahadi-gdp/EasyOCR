import torch
import torch.nn as nn
from .modules import ResNet_FeatureExtractor, BidirectionalLSTM

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
    
    def create_sample_input(self, device):
        return torch.rand((1, 1, 64, 320)).to(device)
    
    def export_to_onnx(self, output: str, device: str, opset_version: int = 11, verbose=False):
        # create sample input
        sample_input = self.create_sample_input(device)

        model = self
        if device == 'cuda':
            model = self.module

        # export
        torch.onnx.export(
            model, 
            sample_input, 
            f=output, 
            do_constant_folding=True,
            verbose=verbose, 
            opset_version=opset_version, 
            input_names=['input'], 
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
