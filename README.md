# polygon-coloring
a cross attention unet for coloring polygons based on input


### Notebooks
I trained using notebooks, so just follow along with the train and infer notebooks for interactive workflos

Or

### Command Line
Setup config in config.py

and  run the scripts
```bash
python train.py

python infer.py
```

--- 

![10 loss curves](imgs/validation_loss)  
10 most recent loss curves

## Sample Predictions

Here’s how the model’s output evolves over training:

![Predictions at the Beginning](imgs/predictions%20in%20the%20beginning.png)  
*Figure 1: Predictions at the very start of training.*

![Predictions in the Middle](imgs/predictions%20in%20the%20middle.png)  
*Figure 2: Predictions halfway through training.*

![Final Predictions](imgs/predictions.png)  
*Figure 3: Predictions after full training.*

---

## Learning Rate Schedule

![Learning Rate Schedule](imgs/lr.png)  
*Figure 4: Cosine annealing learning rate over epochs.*

---

## Final Validation Loss of the winning model

![Final Validation Loss](imgs/val_loss_final.png)  
*Figure 5: Validation loss curve over epochs.*
