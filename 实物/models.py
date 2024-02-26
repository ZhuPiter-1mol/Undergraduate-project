import torch.nn as nn
import torch
class AutoDriveNet(nn.Module):
    '''
    端到端自动驾驶模型
    '''
 
    def __init__(self):
        """
        初始化
        """
        super(AutoDriveNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Dropout(0.5)
        )
        
        # 计算线性层输入的特征数
        self.feature_size = self._get_conv_output_size()
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.feature_size, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.Linear(10, 1)
        )
 
    def _get_conv_output_size(self):
        '''
        计算卷积层输出的特征数
        '''
        test_input = torch.zeros(1, 3, 720, 1280)  # 创建一个测试输入张量
        test_output = self.conv_layers(test_input)  # 通过卷积层前向传播计算输出
        return test_output.view(-1).size(0)  # 返回特征数
 
    def forward(self, input):
        '''
        前向推理
        '''
        output = self.conv_layers(input)
        output = output.view(-1, self.feature_size)  # 将输出扁平化为一维张量
        output = self.linear_layers(output)
        return output
