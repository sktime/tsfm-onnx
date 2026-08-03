import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = torch.randn(4, 512)  # 4 series, 512 past timesteps
out = model.predict(context, horizon=64, quantiles=[0.1, 0.5, 0.9])
print(out.quantiles)  # (4, 64, 3)
print(out.median)    # (4, 64)
