# CIFAR-10-image-classification

The dataset is broken into batches to prevent your machine from running out of memory. The CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc.. Each batch contains the labels and images that are one of the following:

1. airplane

2. automobile

3. bird

4. cat

5. deer

6. dog

7. frog

8. horse

9. ship

10. truck

## Examine the data:

![image](https://cloud.githubusercontent.com/assets/24555661/25315870/b5639bda-2819-11e7-824f-69592c8f75f7.png)

Final Validation Accuracy: 50.34%, which is higher than randomly guessing (~10%). However, changes will need to be made in order to obtain a higher accuracy.

## Network Architecture
50.34% accuracy was obtained using the following network set-up:

1. Convolutional layer 32x32x18
2. Max Pool layer 2x2
3. Dropout layer with keep_prob = 60%
4. Flatten layer
5. Fully connected layer with 384 nodes
6. Output layer with 10 nodes (for the final prediction)
