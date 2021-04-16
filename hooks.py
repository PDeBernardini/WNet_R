import torch
import torch.nn as nn

### my first hook ###

def function_forward(self, input, output):
  if ("UEnc.final" in self.__name__):
    print(f"{self.__name__}: {output.shape}")
    print(f"  > Max output forward: {torch.max(torch.abs(output))}")
    print(f"  > Min output forward: {torch.min(torch.abs(output))}")
  if ("UEnc.dec4.dropout" == self.__name__):
    print(f"{self.__name__}: {output.shape}")
    print(f"  > Max output forward: {torch.max(torch.abs(output))}")
    print(f"  > Min output forward: {torch.min(torch.abs(output))}")

def function_backward(self, grad_input, grad_output):
  grad_input = grad_input[0]
  grad_output = grad_output[0]
  if ("UEnc.final" in self.__name__):
    print(f"{self.__name__} output_shape: {grad_output.shape}, input_shape: {grad_input.shape}")
    print("INPUT GRADIENTS CHECK:")
    print(f"  > Max input_grad backward: {torch.max(torch.abs(grad_input))}")
    print(f"  > Min input_grad backward: {torch.min(torch.abs(grad_input))}")
    print("OUTPUT GRADIENTS CHECK:")
    print(f"  > Max output_grad backward: {torch.max(torch.abs(grad_output))}")
    print(f"  > Min output_grad backward: {torch.min(torch.abs(grad_output))}")

    if(torch.prod(torch.isfinite(grad_output)) == False):
      grad_output = torch.randn_like(grad_output) * 1e-18
      grad_input = torch.randn_like(grad_input) * 1e-18
      print("check shape:", grad_input.shape)
      return (tuple([grad_input]))

  if ("UEnc.dec4.dropout" == self.__name__):
    print(f"{self.__name__}: {grad_output.shape}")
    print(f"  > Max output_grad backward: {torch.max(torch.abs(grad_output))}")
    print(f"  > Min output_grad backward: {torch.min(torch.abs(grad_output))}")

    

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_modules():
            self.layer = layer
            self.layer.__name__ = name
            layer.register_forward_hook(function_forward)
            layer.register_full_backward_hook(function_backward)

    def forward(self, x, returns='both'):
        if returns=='enc':
            return self.model(x, returns = 'enc')
        
        if returns=='dec':
            return self.model(x, returns = 'dec')
        
        if returns=='both':
            return self.model(x, returns = 'both')
        
        else:
            raise ValueError('Invalid returns, returns must be in [enc dec both]')

