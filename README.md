# ImageLocalization

Image localization using EfficientNet model

Run training 
```python
python train.py --batch_size=64 --lr=3e-4 --model_name="efficientnet_b3a"
```

Run inference
```python
python inference.py --model_path="best_model.pt"
```

![download](https://user-images.githubusercontent.com/50166164/208833218-ce916470-b9c7-457b-9549-ad5118330432.png)
