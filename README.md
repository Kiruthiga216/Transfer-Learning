# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

</br>

<p>Develop an image classification model using transfer learning with the pre-trained VGG19 model.</p>

</br>

<img width="578" height="165" alt="image" src="https://github.com/user-attachments/assets/6b95bc1c-13a4-45b3-bced-a64bb8c436a2" />

</br>

## DESIGN STEPS
### STEP 1:

<p>Import required libraries.Then dataset is loaded and define the training and testing dataset.</p>
</br>

### STEP 2:

<p>initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.</p>
</br>

### STEP 3:

<p>Train the model with training dataset.</p>
<br/>

### STEP 4:

<p>Evaluate the model with testing dataset.</p>
<br>

### STEP 5:

<p>Make Predictions on New Data.</p>
<br>

## PROGRAM

<pre><code>

# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match the dataset classes

  num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)



# Include the Loss function and optimizer

  criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)




# Train the model

  def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

    # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: KIRUTHIGA.B")
    print("Register Number: 212224040160")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


</pre>
</code>

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot


<img width="893" height="964" alt="image" src="https://github.com/user-attachments/assets/dcfd1a4e-3c74-4a97-959f-0345bdf0b10c" />

</br>
</br>
</br>

### Confusion Matrix

<img width="843" height="753" alt="image" src="https://github.com/user-attachments/assets/090d2c02-7a8c-4a9c-a989-d2ffe0144c3f" />

</br>
</br>
</br>

### Classification Report

<img width="614" height="300" alt="image" src="https://github.com/user-attachments/assets/f6933ead-f030-4612-a7f5-6404a3b98efb" />

</br>
</br>
</br>

### New Sample Prediction

<img width="478" height="567" alt="image" src="https://github.com/user-attachments/assets/bd9fd0e4-0a83-41a3-8f30-4c8bd4fefbb7" />

<br>

<img width="488" height="548" alt="image" src="https://github.com/user-attachments/assets/3d4a5bcb-fd0e-4c6d-aebb-0eb38d51337a" />

</br>
</br>
</br>

## RESULT
</br>
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
</br>
</br>
